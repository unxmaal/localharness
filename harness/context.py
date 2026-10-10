"""The context each GGUF is served at: trained length, capped by what its KV cache leaves room for. #498."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import NamedTuple

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
#: Arrays longer than this (tokenizer vocabularies) are skipped, not kept.
_KEEP_ARRAY = 4096


def _read(fh, fmt: str):
    size = struct.calcsize(fmt)
    raw = fh.read(size)
    if len(raw) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, raw)[0]


def _string(fh) -> str:
    n = _read(fh, "<Q")
    return fh.read(n).decode("utf-8", "replace")


def _value(fh, kind: int):
    if kind in _SCALAR:
        return _read(fh, _SCALAR[kind])
    if kind == _STRING:
        return _string(fh)
    if kind == _ARRAY:
        inner, count = _read(fh, "<I"), _read(fh, "<Q")
        if count > _KEEP_ARRAY:
            if inner in _SCALAR:
                fh.seek(count * struct.calcsize(_SCALAR[inner]), 1)
            else:
                for _ in range(count):
                    _value(fh, inner)
            return None
        return [_value(fh, inner) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {kind}")


#: Below this a served model fails ordinary agent work; refuse rather than serve it.
FLOOR = 8192
STEP = 1024
#: Default ceiling on one model's KV cache, which llama-server allocates whole at load.
KV_MAX_GIB = 8
#: Bytes per cached element for each llama-server cache type, as (numerator, denominator).
CACHE_BYTES = {"f32": (4, 1), "f16": (2, 1), "bf16": (2, 1), "q8_0": (34, 32),
               "q5_1": (24, 32), "q5_0": (22, 32), "q4_1": (20, 32),
               "q4_0": (18, 32), "iq4_nl": (18, 32)}


#: Qwen2.5-7B-Instruct-Q4_K_M (once eval-7b) serves infovore's bulk claims run in parallel: slots, and the context each slot is sized for. #665.
EVAL_7B_STEM = "Qwen2.5-7B-Instruct-Q4_K_M"
EVAL_7B_SLOTS = 8
EVAL_7B_SLOT_CTX = 4096
#: A claims request is a 6000-character window and a 400-token reply; no slot is sized under it. #665.
MIN_SLOT_CTX = 3072


def slot_plan(stem: str) -> tuple[int, int] | None:
    """(slots, context per slot) a stem is served at by name, else None for the memory plan."""
    if stem != EVAL_7B_STEM:
        return None
    if EVAL_7B_SLOT_CTX < MIN_SLOT_CTX:
        raise ValueError(f"{stem}: {EVAL_7B_SLOT_CTX} tokens per slot is under the "
                         f"{MIN_SLOT_CTX}-token floor a claims request needs")
    return max(int(EVAL_7B_SLOTS), 1), int(EVAL_7B_SLOT_CTX)


def choose_slots(meta: dict, weights: int, budget: int, slots: int, slot_ctx: int,
                 cache_type: str = "f16", kv_cap: int | None = None) -> "Choice":
    """A pool of slots * slot_ctx shared by the slots, when it fits beside the weights and under kv_cap."""
    full = trained(meta)
    per = kv_bytes_per_token(meta, cache_type)
    pool = slots * slot_ctx
    room = max(budget - weights, 0)
    room = min(room, kv_cap) if kv_cap is not None else room
    if not per or not full or slot_ctx > full:
        return Choice(0, full, per, f"refused: {slots} slots of {slot_ctx} tokens, "
                      f"trained for {full}, {per} B/token")
    if pool * per > room:
        return Choice(0, full, per, f"refused: {slots} slots of {slot_ctx} tokens need "
                      f"{pool * per / 1024 ** 3:.1f} GiB of KV, {room / 1024 ** 3:.1f} GiB fits")
    return Choice(pool, full, per, f"slots: {slots} x {slot_ctx} tokens, "
                  f"{pool * per / 1024 ** 3:.1f} GiB of KV (#665)")


class Choice(NamedTuple):
    ctx: int
    trained: int
    kv_per_token: int
    why: str


def _arch(meta: dict, key: str, default=None):
    return meta.get(f"{meta.get('general.architecture', '')}.{key}", default)


def trained(meta: dict) -> int:
    """The context the model was trained for, 0 when the header does not say."""
    return int(_arch(meta, "context_length") or 0)


def _kv_heads(meta: dict) -> list[int]:
    """KV heads per KV-bearing layer; recurrent layers are left out."""
    blocks = int(_arch(meta, "block_count") or 0)
    heads = _arch(meta, "attention.head_count_kv", _arch(meta, "attention.head_count"))
    if isinstance(heads, list):
        return [int(h) for h in heads if h]
    if not heads or not blocks:
        return []
    layers = blocks - int(_arch(meta, "nextn_predict_layers") or 0)
    interval = int(_arch(meta, "full_attention_interval") or 0)
    if interval > 1:
        layers //= interval
    return [int(heads)] * layers


def kv_bytes_per_token(meta: dict, cache_type: str = "f16") -> int:
    """f16 KV bytes one token costs; sliding-window layers are costed as full. 0 if unknown."""
    heads = _kv_heads(meta)
    q_heads = _arch(meta, "attention.head_count")
    q_heads = max(q_heads) if isinstance(q_heads, list) else q_heads
    embd = _arch(meta, "embedding_length")
    per_head = embd // q_heads if embd and q_heads else 0
    k = int(_arch(meta, "attention.key_length") or per_head)
    v = int(_arch(meta, "attention.value_length") or per_head)
    num, den = CACHE_BYTES[cache_type]
    return sum(h * (k + v) for h in heads) * num // den


def choose(meta: dict, weights: int, budget: int, slots: int = 1,
           cache_type: str = "f16", floor: int = FLOOR,
           kv_cap: int | None = None) -> Choice:
    """min(trained, what fits beside the weights and under kv_cap), per slot, in steps of 1024; 0 below `floor`."""
    full = trained(meta)
    per = kv_bytes_per_token(meta, cache_type)
    if not full or not per:
        return Choice(0, full, per, "refused: the GGUF header gives no trained context "
                      "or KV shape to size a cache from")
    room = max(budget - weights, 0)
    room = min(room, kv_cap) if kv_cap is not None else room
    fits = room // per // max(slots, 1) // STEP * STEP
    ctx = min(full // STEP * STEP or full, fits)
    if ctx < floor:
        limit = (f"trained for {full}" if full < floor else
                 f"{room / 1024 ** 3:.1f} GiB of KV room "
                 f"holds {fits} tokens at {per} B/token")
        return Choice(0, full, per, f"refused: {limit}, under the {floor}-token floor")
    if ctx < full:
        return Choice(ctx, full, per, f"memory: {fits} tokens fit in "
                      f"{room / 1024 ** 3:.1f} GiB of KV at {per} B/token "
                      f"(trained {full})")
    return Choice(ctx, full, per, f"trained: {full} tokens, {per * ctx / 1024 ** 3:.1f} "
                  "GiB of KV fits")


def read_meta(path) -> dict:
    """The key/value metadata of a GGUF file; long arrays come back as None."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{Path(path).name}: not a GGUF file")
        version = _read(fh, "<I")
        if version < 2:
            raise ValueError(f"{Path(path).name}: GGUF v{version} is not read")
        _read(fh, "<Q")
        count = _read(fh, "<Q")
        meta = {}
        for _ in range(count):
            key = _string(fh)
            meta[key] = _value(fh, _read(fh, "<I"))
        return meta


def _weights(path) -> int:
    return Path(path).stat().st_size


def budget_bytes(conn=None) -> int:
    """The machine's ceiling less its measured reserve (#415): what weights plus KV may take."""
    from harness import memory
    return int((memory.ceiling_gb() - memory.measured_reserve_gb(conn)) * 1024 ** 3)


def kv_cap_bytes() -> int:
    """The most KV cache one model may allocate: LLAMACPP_KV_MAX_GIB, default 8."""
    import os
    return int(float(os.environ.get("LLAMACPP_KV_MAX_GIB") or KV_MAX_GIB) * 1024 ** 3)


def coresident_bytes(conn, defaults=None, config=None, size_of=None) -> int:
    """The largest text model this machine serves on mlx_lm.server, which stays resident beside the router."""
    from harness import downloads, gateway, lanes, memory, serving
    if defaults is None:
        from harness import adopt
        defaults = adopt.lane_defaults(conn)
    if size_of is None:
        def size_of(repo):
            path = downloads.path_of(repo, conn, kind=downloads.HUB)
            got = memory.size_gb(str(path)) if path else None
            return int(got * 1024 ** 3) if got else 0
    text = set(lanes.TEXT_SERVED) | set(gateway.TEXT_LANES)
    entries = {str(e.get("model_name", "")).lower(): e
               for e in gateway.load(config).get("model_list") or []}
    best = 0
    for lane, spec in defaults.items():
        name = (spec or "").partition(",")[0].strip()
        if lane not in text or not name or ":" in name:
            continue
        if serving.engine_for(name, config=config) == serving.LLAMACPP:
            continue
        entry = entries.get(name.lower())
        repo = (gateway.strip_provider(str((entry.get("litellm_params") or {}).get("model", "")))
                if entry else name)
        if "/" in repo:
            best = max(best, int(size_of(repo) or 0))
    return best


def plan(conn, budget: int | None = None, slots: int = 1,
         cache_type: str = "f16", kv_cap: int | None = None) -> list[dict]:
    """Choose and record the context of every live GGUF row on this machine."""
    import time

    from harness import downloads
    budget = budget_bytes(conn) if budget is None else budget
    kv_cap = kv_cap_bytes() if kv_cap is None else kv_cap
    try:
        beside = coresident_bytes(conn)
    except Exception:  # noqa: BLE001
        beside = 0
    out = []
    for row in downloads.live(conn, kind=downloads.GGUF):
        path = Path(row["path"])
        if not path.exists():
            continue
        stem = path.name[:-len(".gguf")]
        fixed = slot_plan(stem)
        n = fixed[0] if fixed else slots
        try:
            meta, weights = read_meta(path), _weights(path)
            got = (choose_slots(meta, weights, budget - beside, *fixed, cache_type, kv_cap)
                   if fixed else choose(meta, weights, budget - beside, slots,
                                        cache_type, kv_cap=kv_cap))
        except (OSError, ValueError, KeyError, struct.error) as exc:
            got = Choice(0, 0, 0, f"refused: unreadable GGUF header ({exc})")
        conn.execute("UPDATE downloads SET ctx = ?, ctx_trained = ?, kv_bytes_token = ?, "
                     "ctx_slots = ?, ctx_why = ?, ctx_at = ?, kv_cap_bytes = ?, "
                     "coresident_bytes = ? WHERE id = ?",
                     (got.ctx, got.trained, got.kv_per_token, n, got.why,
                      time.time(), kv_cap, beside, row["id"]))
        out.append({"stem": stem, "path": str(path), "ctx": got.ctx, "slots": n,
                    "why": got.why})
    conn.commit()
    return out


def served(conn, stem: str) -> dict | None:
    """The recorded context row for a router stem on this machine, or None."""
    from harness import downloads
    for row in downloads.live(conn, kind=downloads.GGUF):
        if Path(row["path"]).name == f"{stem}.gguf" and row.get("ctx_at"):
            return row
    return None


def _get_json(url: str) -> dict:
    import json
    import urllib.request
    with urllib.request.urlopen(url, timeout=5.0) as r:
        return json.loads(r.read() or b"{}")


def _upstream(spec: str, config=None) -> tuple[str, str] | None:
    """(router base URL, stem) a text spec reaches llama-server through, else None."""
    from harness import gateway, router
    from harness.serving import LLAMACPP_PREFIX
    name = (spec or "").partition(",")[0].strip()
    if name.startswith(LLAMACPP_PREFIX):
        return router.url(), name[len(LLAMACPP_PREFIX):].strip()
    if not name or ":" in name:
        return None
    served = gateway.served_path(config)
    for entry in gateway.load(served if served.exists() else config).get("model_list") or []:
        if str(entry.get("model_name", "")).lower() == name.lower():
            params = entry.get("litellm_params") or {}
            base = str(params.get("api_base", "")).rstrip("/").removesuffix("/v1")
            return (base, gateway.strip_provider(str(params.get("model", "")))) if base else None
    return None


def served_ctx(spec: str, config=None, get=None, conn=None) -> int | None:
    """The per-slot context llama-server serves this spec at: the router's own launch args,
    else the stored choice; None when neither says (another engine, or unknown)."""
    from harness import ds4
    if (spec or "").startswith(ds4.PREFIX):
        return ds4.served_ctx(spec)
    where = _upstream(spec, config)
    if where is None:
        return None
    base, stem = where
    try:
        for m in (get or _get_json)(base + "/models").get("data") or []:
            args = (m.get("status") or {}).get("args") or [] if isinstance(m, dict) else []
            if m.get("id") == stem and "--ctx-size" in args:
                return int(args[args.index("--ctx-size") + 1])
    except Exception:  # noqa: BLE001
        pass
    from harness import downloads
    try:
        with downloads.store(conn) as c:
            row = served(c, stem)
    except Exception:  # noqa: BLE001
        return None
    return int(row["ctx"]) if row and row["ctx"] else None


def refusal(stem: str, conn=None) -> str:
    """Why this stem is not served, or "" when it is or nothing is recorded."""
    from harness import downloads
    try:
        with downloads.store(conn) as c:
            row = served(c, stem)
    except Exception:  # noqa: BLE001
        return ""
    return row["ctx_why"] if row and not row["ctx"] else ""


def preset_text(plans: list[dict], default_ctx: int, slots: int, cache_type: str = "f16") -> str:
    """llama-server's --models-preset INI: [*] for unrecorded files, one section per servable stem."""
    def args(ctx: int, n: int) -> list[str]:
        # The KV type the plan priced, so the server allocates what was budgeted. #521.
        out = [f"c = {ctx}", f"parallel = {n}", f"cache-type-k = {cache_type}",
               f"cache-type-v = {cache_type}"]
        return out + ["kv-unified = true"] if n > 1 else out
    lines = ["version = 1", "", "[*]", *args(default_ctx, slots), ""]
    for p in sorted(plans, key=lambda p: p["stem"]):
        if p["ctx"]:
            lines += [f"; {p['why']}", f"[{p['stem']}]", *args(p["ctx"], p.get("slots", slots)), ""]
        else:
            lines += [f"; {p['stem']} {p['why']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    """`python -m harness.context OUT.ini`: plan, record and write the router preset."""
    import os
    import sys

    from harness import memory_store as ms
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m harness.context PRESET.ini", file=sys.stderr)
        return 2
    slots = max(int(os.environ.get("LLAMACPP_PARALLEL") or 1), 1)
    cache = os.environ.get("LLAMACPP_CACHE_TYPE") or "f16"
    default = int(os.environ.get("LLAMACPP_CTX") or 16384)
    conn = ms.connect()
    try:
        plans = plan(conn, slots=slots, cache_type=cache)
    finally:
        conn.close()
    for p in plans:
        print(f"context: {p['stem']}: {p['ctx'] or 'refused'} ({p['why']})", file=sys.stderr)
    Path(argv[0]).write_text(preset_text(plans, default, slots, cache), encoding="utf-8")
    print(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
