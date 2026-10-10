"""Each benchmark row says what its model is, from what the store already holds. #670."""
import json
import re

import pytest

from harness import gateway, models, publish, runs
from harness import memory_store as ms

MAC = gateway.REPO / "gateway" / "config.yaml"
ORNITH_REPO = "ornith-ai/Ornith-1.5-35B-A3B-GGUF"
ORNITH = "Ornith-1.5-35B-Q4_K_M"


@pytest.fixture
def conn(monkeypatch):
    monkeypatch.setenv("GATEWAY_CONFIG", str(MAC))
    c = ms.connect()
    ms.record(c, ms.Seen("bartowski/Qwen2.5-7B-Instruct-GGUF", "test", registry="huggingface"))
    c.execute("UPDATE proposals SET card_tags = ? WHERE name = ?",
              (json.dumps(["gguf", "base_model:quantized:Qwen/Qwen2.5-7B-Instruct"]),
               "bartowski/Qwen2.5-7B-Instruct-GGUF"))
    ms.record(c, ms.Seen(ORNITH_REPO, "test", registry="huggingface"))
    c.execute("INSERT INTO downloads (repo, kind, path, file) VALUES (?, 'gguf', 'p', ?)",
              (ORNITH_REPO, f"{ORNITH}.gguf"))
    c.commit()
    yield c
    c.close()


def test_a_gguf_model_names_its_family_size_quant_publisher_and_source(conn):
    got = models.about(conn, "Qwen2.5-7B-Instruct-Q4_K_M")
    assert got == {"family": "Qwen/Qwen2.5-7B-Instruct", "params": "7B", "quant": "Q4_K_M",
                   "publisher": "bartowski",
                   "source": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF"}


def test_a_router_stem_is_traced_to_its_repo_through_the_downloads_row(conn):
    got = models.about(conn, f"llamacpp:{ORNITH}")
    assert got["publisher"] == "ornith-ai" and got["quant"] == "Q4_K_M"
    assert got["params"] == "35B (3B active)"
    assert got["source"] == f"https://huggingface.co/{ORNITH_REPO}"
    assert got["family"] == models.UNKNOWN_FACT


def test_an_mlx_repo_id_is_its_own_source(conn):
    got = models.about(conn, "mlx-community/Qwen3-4B-Instruct-2507-4bit")
    assert got["publisher"] == "mlx-community" and got["quant"] == "4bit"
    assert got["params"] == "4B"
    assert got["source"] == "https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit"


def test_a_model_the_store_knows_nothing_about_is_unknown(conn):
    assert models.about_text(models.about(conn, "")) == "unknown"
    assert models.about_text(models.about(conn, "osocr:auto")) == "unknown"


def test_the_line_lists_what_is_known_and_says_unknown_for_the_rest(conn):
    line = models.about_text(models.about(conn, f"llamacpp:{ORNITH}"))
    assert line.startswith("family unknown, 35B (3B active) params, Q4_K_M, by ornith-ai")


def _record(conn, path, lane, rows, when, receipt=None):
    runs.record(conn, path, {"generated": when,
                             "receipt": {"modality": lane, "tier": "measure", **(receipt or {})},
                             "environment": {"hw_model": "Mac17,15", "os": "macOS-27.0.1-arm64",
                                             "arch": "arm64", "memory_gb": 96,
                                             "accelerator": {"kind": "unified", "name": "arm64"}},
                             "specs": {}, "rows": rows})


def _rows(key, n=2):
    return [{"case_id": f"c{i}", "candidate": key, "passed": 1, "seconds": 1.0}
            for i in range(n)]


def test_no_benchmark_row_names_only_an_alias_and_every_row_says_what_it_is(conn):
    _record(conn, "runs/a", "code",
            _rows("eval-7b") + _rows("sohot-code") + _rows("sohot-web") + _rows(f"llamacpp:{ORNITH}"),
            "2026-10-05T10:00:00", receipt={"resolved": {"sohot-code": f"llamacpp:{ORNITH}"}})
    conn.commit()
    mid = conn.execute("SELECT id FROM machines WHERE hw_model = 'Mac17,15'").fetchone()["id"]
    doc = publish.export(conn, machine_id=mid, now=1.79e9)
    lane = next(l for l in doc["lanes"] if l["lane"] == "code")
    rows = [r for e in lane["exams"] for r in e["rows"]]
    assert {r["candidate"] for r in rows} == {"eval-7b", "sohot-code", "sohot-web", f"llamacpp:{ORNITH}"}
    for r in rows:
        assert r["about"], r
        if r["model"]:
            assert not models.is_alias(r["model"]), r
        else:
            assert r["candidate"] == "sohot-web", r
    by = {r["candidate"]: r for r in rows}
    assert by["eval-7b"]["model"] == "Qwen2.5-7B-Instruct-Q4_K_M"
    assert by["sohot-code"]["model"] == f"llamacpp:{ORNITH}"
    from harness import site
    page = site.benchmarks([doc], now=1.79e9)
    cells = re.findall(r"<tr[^>]*><td>(.*?)</td>", page, re.S)
    assert len(cells) == len(rows)
    for cell in cells:
        assert ("Qwen2.5-7B-Instruct-Q4_K_M" in cell or ORNITH in cell
                or "model unknown" in cell), cell
        assert 'class="about"' in cell, cell
    assert "served as eval-7b" in page and "sohot-web (model unknown)" in page
    assert f"https://huggingface.co/{ORNITH_REPO}" in page
