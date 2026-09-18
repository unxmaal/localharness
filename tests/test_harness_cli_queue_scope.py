"""The lane scope must not be a sample of the backlog. Issue #209.

`--lane` was applied AFTER `ms.judgeable(limit=top * 4)`, so a narrow budget
narrowed the population the filter could see. At --top 25 the sample is 100 of
114 and nothing shows; at --top 2 it is 8, and the image lane reported empty
while holding five candidates. A narrow budget is when the scope matters most.
"""
import argparse

from harness import cli


class _Store:
    """Records the limit it was asked for, so the order can be asserted."""

    def __init__(self, rows):
        self.rows = rows

    def close(self):
        pass


def _rows():
    """Twenty rows, with the only image row LAST, which is the case that bit."""
    out = [{"name": f"org/code{i}", "lane": "code", "description": "",
            "times": 1} for i in range(19)]
    out.append({"name": "org/theimage", "lane": "image", "description": "",
                "times": 1})
    return out


def test_the_lane_scope_sees_the_whole_backlog(monkeypatch):
    asked = []

    def judgeable(conn, limit=50):
        asked.append(limit)
        return _rows()

    monkeypatch.setattr("harness.memory_store.judgeable", judgeable)
    rows, waiting = cli._queueable(_Store(_rows()), "image")
    assert [r["name"] for r in rows] == ["org/theimage"]
    assert waiting == 1
    assert asked and min(asked) > len(_rows()), (
        "the fetch must not be sized from --top, or the filter samples")


def test_an_unscoped_queue_keeps_every_row(monkeypatch):
    monkeypatch.setattr("harness.memory_store.judgeable",
                        lambda conn, limit=50: _rows())
    rows, waiting = cli._queueable(_Store(_rows()), "")
    assert waiting == 20


def test_a_text_candidate_is_in_scope_for_the_web_lane(monkeypatch):
    """The scope asks lanes.serves, not equality, so #208's fix survives."""
    monkeypatch.setattr("harness.memory_store.judgeable",
                        lambda conn, limit=50: _rows())
    rows, waiting = cli._queueable(_Store(_rows()), "web")
    assert waiting == 19


def test_the_denominator_counts_only_what_a_screen_could_answer(monkeypatch, capsys):
    """rank() drops adapters and laneless rows, so counting before it says
    "2 of 11 waiting" for a lane holding 6 LoRAs and 5 models."""
    rows = [{"name": "org/real", "lane": "image", "description": "", "times": 1},
            {"name": "org/a-style-lora", "lane": "image", "description": "",
             "times": 1},
            {"name": "org/pack-ComfyUI", "lane": "image", "description": "",
             "times": 1}]
    monkeypatch.setattr("harness.memory_store.judgeable",
                        lambda conn, limit=50: rows)
    monkeypatch.setattr("harness.memory_store.connect", lambda *a, **k: _Store(rows))
    monkeypatch.setattr("harness.rank.serving", lambda *a, **k: set())
    monkeypatch.setattr("harness.rank.lanes_with_receipts", lambda *a, **k: set())
    cli._report_queue(argparse.Namespace(lane="image", top=25, json=False))
    out = capsys.readouterr().out
    assert "1 of 1 waiting" in out, out
