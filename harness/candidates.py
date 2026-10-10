"""One stored mapping: proposal name <-> engine spec <-> receipt key. #407.

Written by the code that resolves a spec (the screen, measure, adopt and a
receipt's own `specs` map) and read by every consumer, so nothing re-derives a
key from a name by string rules.
"""
from __future__ import annotations

import time


def key_of(spec: str) -> str:
    """The summary key evals.run writes for `spec`, from the runner itself."""
    from evals.run import receipt_key
    return receipt_key(spec)


def _proposal_id(conn, name: str):
    if not name:
        return None
    row = conn.execute("SELECT id FROM proposals WHERE name = ?",
                       (name,)).fetchone()
    return row["id"] if row else None


def ensure(conn, spec: str, *, proposal: str = "", lane: str = "",
           key: str = "") -> int | None:
    """The candidate row for `spec`, created or completed. None if it cannot run.

    `key` is what a receipt recorded and wins over the computed one; a
    proposal is attached only if it exists, and never created here.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    pid = _proposal_id(conn, proposal)
    base = spec.split(",", 1)[0]
    if pid is None and base != spec:
        # A run-option variant is the same proposal as its bare spec. #429.
        got = conn.execute("SELECT proposal_id FROM candidates WHERE spec = ?",
                           (base,)).fetchone()
        pid = got["proposal_id"] if got else None
    row = conn.execute("SELECT receipt_key FROM candidates WHERE spec = ?",
                       (spec,)).fetchone()
    key = key or (row["receipt_key"] if row else key_of(spec))
    if not key:
        return None
    # Insert-or-ignore first: a select-then-insert races two writers. RULE #386.
    conn.execute(
        "INSERT OR IGNORE INTO candidates (proposal_id, spec, receipt_key, "
        "lane, created_at) VALUES (?,?,?,?,?)",
        (pid, spec, key, lane or "", time.time()))
    conn.execute(
        "UPDATE candidates SET proposal_id = COALESCE(?, proposal_id), "
        "receipt_key = ?, lane = CASE WHEN lane = '' THEN ? ELSE lane END "
        "WHERE spec = ?", (pid, key, lane or "", spec))
    if "," not in spec:
        if pid is not None:
            link_variants(conn, spec)
        # Rows a run stored before this candidate had a row. #410.
        conn.execute(
            "UPDATE results SET candidate_id = (SELECT id FROM candidates "
            "WHERE spec = ?) WHERE candidate_id IS NULL AND candidate = ?",
            (spec, key))
    conn.commit()
    return conn.execute("SELECT id FROM candidates WHERE spec = ?",
                        (spec,)).fetchone()["id"]


def link_variants(conn, base: str | None = None) -> int:
    """Give each `spec,options` row its bare spec's proposal. #429."""
    owner = {r["spec"]: r["proposal_id"] for r in conn.execute(
        "SELECT spec, proposal_id FROM candidates "
        "WHERE proposal_id IS NOT NULL").fetchall()}
    n = 0
    for r in conn.execute("SELECT id, spec FROM candidates "
                          "WHERE proposal_id IS NULL").fetchall():
        head = r["spec"].split(",", 1)[0]
        if head == r["spec"] or head not in owner or base not in (None, head):
            continue
        conn.execute("UPDATE candidates SET proposal_id = ? WHERE id = ?",
                     (owner[head], r["id"]))
        n += 1
    return n


def for_proposal(conn, lane: str, name: str, attaches_to: str = "") -> str:
    """Build the spec for a proposal once and store it; "" when none exists."""
    from harness import screen
    spec = screen.candidate_for(lane, name, attaches_to, conn=conn)
    if spec:
        ensure(conn, spec, proposal=name, lane=lane)
    return spec


def from_receipt(conn, specs: dict, *, proposals: dict | None = None,
                 lane: str = "") -> int:
    """Import a receipt's `specs` map (key -> spec). Returns rows touched."""
    n = 0
    for key, spec in (specs or {}).items():
        if ensure(conn, spec, key=key, lane=lane,
                  proposal=(proposals or {}).get(spec, "")):
            n += 1
    return n


def get(conn, label: str) -> dict | None:
    """The candidate a spec, receipt key or proposal name stands for."""
    if not label:
        return None
    for sql in ("SELECT c.*, p.name AS proposal FROM candidates c "
                "LEFT JOIN proposals p ON p.id = c.proposal_id "
                "WHERE c.spec = ?",
                "SELECT c.*, p.name AS proposal FROM candidates c "
                "LEFT JOIN proposals p ON p.id = c.proposal_id "
                "WHERE c.receipt_key = ? ORDER BY c.id DESC",
                "SELECT c.*, p.name AS proposal FROM candidates c "
                "JOIN proposals p ON p.id = c.proposal_id "
                "WHERE p.name = ? ORDER BY c.id DESC"):
        row = conn.execute(sql, (label,)).fetchone()
        if row:
            return dict(row)
    return None


def keys(conn, label: str) -> set[str]:
    """Every receipt key stored for a proposal name, spec or key."""
    rows = conn.execute(
        "SELECT c.receipt_key FROM candidates c "
        "LEFT JOIN proposals p ON p.id = c.proposal_id "
        "WHERE c.spec = ? OR c.receipt_key = ? OR p.name = ?",
        (label, label, label)).fetchall()
    return {r["receipt_key"] for r in rows if r["receipt_key"]}


def key_for(conn, spec: str) -> str:
    """The receipt key for a spec: stored if known, else stored now."""
    row = get(conn, spec) if conn is not None else None
    if row and row["spec"] == spec:
        return row["receipt_key"]
    if conn is not None and ensure(conn, spec):
        return get(conn, spec)["receipt_key"]
    return key_of(spec)


def served(conn, spec: str, *, lane: str = "", config=None) -> int | None:
    """The candidate a lane's spec is measured under: its own row, or a deprecated old
    name's row while the renamed spec has no results of its own. #670."""
    from harness import models
    cid = ensure(conn, spec, lane=lane)
    if cid is None or _has_results(conn, cid):
        return cid
    for old in models.old_names(spec, config):
        if conn.execute("SELECT 1 FROM results WHERE candidate = ? LIMIT 1", (old,)).fetchone():
            return ensure(conn, old, lane=lane)
    return cid


def _has_results(conn, cid: int) -> bool:
    return conn.execute("SELECT 1 FROM results WHERE candidate_id = ? LIMIT 1",
                        (cid,)).fetchone() is not None
