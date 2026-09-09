"""Sweeping a constant and saying whether anything moved.

The machinery is tested with toy functions rather than the real probes: the
probes need a populated cache and 24 clones on disk, and a test that needs
those asserts this Mac rather than the code.
"""
import pytest

from harness import sensitivity as sv


def test_a_constant_nothing_depends_on_is_inert():
    f = sv.measure("k", 3, [1, 2, 3, 4], lambda v: ["a", "b", "c"])
    assert f.verdict == "inert"


def test_a_default_with_room_on_both_sides_is_a_band():
    f = sv.measure("k", 3, [1, 2, 3, 4, 9],
                   lambda v: ["a", "b"] if v < 5 else ["b", "a"])
    assert f.verdict == "band"
    assert f.band == (1, 2, 3, 4)


def test_a_default_whose_neighbour_changes_the_answer_is_an_edge():
    f = sv.measure("k", 3, [1, 2, 3, 4],
                   lambda v: ["a", "b"] if v == 3 else ["b", "a"])
    assert f.verdict == "edge"
    assert f.band == (3,)


def test_a_band_is_one_sided_when_the_default_sits_at_its_end():
    f = sv.measure("k", 2, [1, 2, 3, 4],
                   lambda v: ["a"] if v >= 2 else ["b"])
    assert f.band == (2, 3, 4)
    assert "LOW edge" in sv.report([f])


def test_a_sweep_that_omits_the_default_is_refused():
    """Without the value actually in use there is no way to ask whether the
    value NEXT DOOR behaves differently, which is the whole question."""
    with pytest.raises(ValueError) as exc:
        sv.measure("k", 3, [1, 2, 4],
                   lambda v: ["a"] if v < 4 else ["b"])
    assert "default" in str(exc.value)


def test_a_probe_that_cannot_start_is_unmeasurable_not_inert():
    def boom(v):
        raise RuntimeError("no cache")

    f = sv.measure("k", 3, [1, 2, 3], boom)
    assert f.verdict == "unmeasurable"
    assert "no cache" in f.readings[0].error


def test_one_value_failing_does_not_hide_the_others():
    def sometimes(v):
        if v == 9:
            raise RuntimeError("nope")
        return ["a", "b"] if v <= 3 else ["b", "a"]

    f = sv.measure("k", 3, [1, 3, 5, 9], sometimes)
    assert [r.taken for r in f.readings] == [True, True, True, False]
    assert f.verdict == "band"


# ---------------------------------------------------------------------------
# The distances. A statistic blind to the effect under test reads as a null
# result, which is how HALF_LIFE_DAYS was called inert while reordering ten of
# the top fifteen. See the module docstring.
# ---------------------------------------------------------------------------

def test_a_reordering_with_identical_membership_is_not_a_distance_of_zero():
    d, detail = sv.rank_distance(["a", "b", "c"], ["c", "b", "a"])
    assert d == 2
    assert "+0 -0" not in detail  # membership is unchanged and must not read as no change


def test_membership_change_is_reported_alongside_position_change():
    d, detail = sv.rank_distance(["a", "b", "c"], ["a", "b", "d"])
    assert d == 1
    assert "+1 -1" in detail
    assert "out: c" in detail


def test_an_identical_ranking_is_distance_zero():
    assert sv.rank_distance(["a", "b"], ["a", "b"])[0] == 0


def test_a_shorter_ranking_counts_the_missing_positions():
    assert sv.rank_distance(["a", "b", "c"], ["a", "b"])[0] == 1


def test_kendall_is_plus_one_for_the_same_order_and_minus_one_for_reversed():
    assert sv.kendall(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert sv.kendall(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_kendall_is_none_when_the_two_share_fewer_than_two_items():
    assert sv.kendall(["a", "b"], ["c", "d"]) is None


def test_a_mapping_reports_which_keys_changed():
    d, detail = sv.map_distance({"a": "fits", "b": "dead"},
                                {"a": "fits", "b": "too-big"})
    assert d == 1
    assert "b: dead->too-big" in detail


def test_distance_dispatches_on_the_shape_of_the_output():
    assert sv.distance({"a": 1}, {"a": 2})[0] == 1
    assert sv.distance(["a", "b"], ["b", "a"])[0] == 2
    assert sv.distance(7, 7)[0] == 0
    assert sv.distance(7, 8)[0] == 1


# ---------------------------------------------------------------------------
# bound(): the patch that silently does not take
# ---------------------------------------------------------------------------

def test_bound_pins_an_argument_a_caller_never_passes():
    """A module constant used as a DEFAULT is captured at definition time, so
    rebinding it changes nothing and the constant reports inert."""
    import types
    mod = types.ModuleType("m")
    mod.SCALE = 2
    mod.f = lambda x, scale=mod.SCALE: x * scale

    mod.SCALE = 10
    assert mod.f(3) == 6, "rebinding the constant must not be how this is done"

    with sv.bound(mod, "f", scale=10):
        assert mod.f(3) == 30
    assert mod.f(3) == 6


def test_bound_restores_the_original_even_when_the_body_raises():
    import types
    mod = types.ModuleType("m")
    original = mod.f = lambda x, scale=2: x * scale
    with pytest.raises(RuntimeError):
        with sv.bound(mod, "f", scale=10):
            raise RuntimeError("boom")
    assert mod.f is original


# ---------------------------------------------------------------------------
# The quality check, which answers a different question to the distance
# ---------------------------------------------------------------------------

def test_a_quality_check_is_reported_beside_the_distance():
    f = sv.measure("k", 3, [1, 3], lambda v: ["a"] if v == 3 else ["b"],
                   quality=lambda v: "separates" if v >= 3 else "LEAKS 1")
    assert [r.quality for r in f.readings] == ["LEAKS 1", "separates"]
    assert "[separates]" in sv.report([f])


def test_a_failing_quality_check_does_not_lose_the_measurement():
    def check(v):
        raise RuntimeError("control blew up")

    f = sv.measure("k", 3, [1, 3], lambda v: ["a"] if v == 3 else ["b"],
                   quality=check)
    assert f.verdict == "edge"
    assert "check failed" in f.readings[0].quality
