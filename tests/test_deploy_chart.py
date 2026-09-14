"""Assert the RENDERED chart, which is where chart bugs live.

`helm lint` says the YAML is a chart. It does not say a GPU job asks for a GPU,
that nothing cheap is parked on the scarce node, or that two profiles do not
both claim the driver. Those are the failures that cost a day on a cluster, and
every one is checkable on an ordinary runner with no cluster at all -- the same
move as the Linux port's phase 0, which found six defects before the hardware
existed.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from harness import feeds, github

REPO = Path(__file__).resolve().parent.parent
CHART = REPO / "deploy" / "localharness"
PROFILES = ("values-local.yaml", "values-gpu-host.yaml", "values-eks.yaml")

_HELM_MISSING = pytest.mark.skipif(
    shutil.which("helm") is None, reason="helm is not installed here")


def test_this_suite_is_not_silently_skipped_where_it_must_run():
    """RULE #201: a suite that has only ever run on one machine asserts that
    machine. These assertions were written for #148 phase 0 on the promise that
    they need no cluster and no card -- and then no CI job installed helm, so
    every one of them skipped on all three runners and had only ever run on the
    author's laptop.

    A skip is invisible. This is the only test here that must NOT be skipped,
    which is why it asserts the tool rather than skipping on it.
    """
    import os
    import sys
    if os.environ.get("CI") and sys.platform.startswith("linux"):
        assert shutil.which("helm"), (
            "the Linux job must install helm: without it the chart assertions "
            "skip and report green for something they never checked")


def render(*values: str) -> list[dict]:
    argv = ["helm", "template", "lh", str(CHART)]
    for v in values:
        argv += ["-f", str(CHART / v)]
    out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def pods(docs: list[dict]) -> list[tuple[dict, dict]]:
    """(workload, podSpec) for every schedulable thing the chart renders."""
    out = []
    for d in docs:
        spec = d.get("spec", {})
        if d.get("kind") == "CronJob":
            spec = spec["jobTemplate"]["spec"]
        template = spec.get("template", {})
        pod = template.get("spec")
        if pod:
            pod = dict(pod, metadata=template.get("metadata", {}))
            out.append((d, pod))
    return out


def tier(workload: dict, pod: dict | None = None) -> str:
    """The tier belongs to the POD, not to the workload wrapping it.

    Reading only the workload's own labels called the Postgres StatefulSet
    tier-less and then asserted it should know where the store is -- which is
    the store. A label on a template is not a label on its parent.
    """
    if pod is not None:
        got = pod.get("metadata", {}).get("labels", {}).get("localharness.io/tier")
        if got:
            return got
    return workload["metadata"]["labels"].get("localharness.io/tier", "")


# ---- it renders at all ---------------------------------------------------

@_HELM_MISSING
@pytest.mark.parametrize("profile", PROFILES)
@_HELM_MISSING
def test_every_profile_renders(profile):
    assert render(profile)


@_HELM_MISSING
def test_the_defaults_render_without_a_profile():
    """A chart whose defaults do not install is a chart nobody can try."""
    assert render()


# ---- the GPU, which is the whole scheduling problem ----------------------

@_HELM_MISSING
def test_a_gpu_workload_asks_for_a_gpu():
    """Without `limits.nvidia.com/gpu` the scheduler will place it anywhere,
    it runs on a node with no card, and the failure is a CUDA error inside a
    container rather than a scheduling refusal."""
    for w, pod in pods(render("values-gpu-host.yaml")):
        if tier(w, pod) not in ("screen", "measure"):
            continue
        limits = pod["containers"][0]["resources"]["limits"]
        assert int(limits["nvidia.com/gpu"]) >= 1, f"{w['metadata']['name']}"


@_HELM_MISSING
def test_a_gpu_workload_tolerates_the_taint_that_reserves_the_card():
    """A GPU pool is tainted so cheap work stays off it. A GPU job without the
    matching toleration is then unschedulable and sits Pending in silence."""
    docs = render("values-gpu-host.yaml")
    for w, pod in pods(docs):
        if tier(w, pod) not in ("screen", "measure"):
            continue
        keys = {t.get("key") for t in pod.get("tolerations", [])}
        selector = pod.get("nodeSelector", {})
        assert keys, f"{w['metadata']['name']} targets the GPU pool and tolerates nothing"
        assert any(k in str(selector) for k in keys), \
            "the toleration does not match the selector that puts it there"


@_HELM_MISSING
def test_nothing_cheap_is_parked_on_the_card():
    """THE EXPENSIVE MISTAKE. inspect imports no model and calls no gateway --
    it clones source and reads it. A pod holding the only card to read a README
    starves the one tier that needs one."""
    for w, pod in pods(render("values-gpu-host.yaml")):
        if tier(w, pod) in ("screen", "measure"):
            continue
        limits = pod["containers"][0].get("resources", {}).get("limits", {})
        assert "nvidia.com/gpu" not in limits, f"{tier(w, pod)} asks for a GPU"
        assert not pod.get("nodeSelector"), f"{tier(w, pod)} is pinned to the GPU pool"


@_HELM_MISSING
def test_the_default_profile_schedules_no_gpu_at_all():
    """On a machine with no card a GPU workload is not slow, it is Pending
    forever, which reads as a hung cluster rather than a wrong profile."""
    for _, pod in pods(render()):
        limits = pod["containers"][0].get("resources", {}).get("limits", {})
        assert "nvidia.com/gpu" not in limits


@_HELM_MISSING
def test_two_profiles_never_both_claim_the_driver():
    """The portability toggle with a real failure behind it: on a host-driver
    machine the driver must stay where it is, because another operating system
    on the same box has a prior claim on it. On a managed node pool the operator
    owns it. Same chart, opposite setting, and both at once breaks the host."""
    seen = {}
    for profile in PROFILES:
        values = yaml.safe_load((CHART / profile).read_text(encoding="utf-8")) or {}
        gpu = values.get("gpu") or {}
        if gpu.get("enabled"):
            seen[profile] = gpu.get("driver")
    assert seen, "no profile enables a GPU, so this test is asserting nothing"
    assert set(seen.values()) <= {"host", "operator"}
    for profile, driver in seen.items():
        assert driver is not None, f"{profile} enables a GPU and names no driver owner"


# ---- the cache TTL and the schedule, which live in two languages ---------

@_HELM_MISSING
def test_the_sweep_schedule_stays_above_the_cache_ttl():
    """#77: with everything cached for days, a sweep more often than the TTL
    re-reads its own answer and reports success without looking. The interval
    is a Python constant and the schedule is a cron string in YAML, so nothing
    holds them together but this."""
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))
    schedule = values["sweep"]["schedule"]
    minute, hour, dom, month, dow = schedule.split()
    assert dom == "*" and month == "*" and dow != "*", (
        f"{schedule!r} is not a weekly schedule; if the cadence changes, this "
        f"test must be taught the new arithmetic rather than deleted")
    interval_hours = 7 * 24
    ttl = github.TTL_BY_ENDPOINT_HOURS["/starred"]
    assert interval_hours > ttl, (
        f"a sweep every {interval_hours}h against a {ttl}h cache re-reads a "
        f"cached answer and reports success")


@_HELM_MISSING
def test_the_chart_interval_matches_the_one_the_cli_uses():
    """Two answers to one question: `lh discover` uses
    feeds.DEFAULT_INTERVAL_DAYS and the CronJob uses a cron string."""
    assert feeds.DEFAULT_INTERVAL_DAYS == 7, (
        "the CLI's sweep interval moved; the chart's weekly schedule and the "
        "test above encode 7 days and must move with it")


# ---- pinning -------------------------------------------------------------

@_HELM_MISSING
def test_no_workload_runs_a_moving_tag():
    """A CronJob pulling `latest` runs a different program every week and
    reports the difference as a discovery."""
    for profile in ("",) + PROFILES:
        docs = render(profile) if profile else render()
        for w, pod in pods(docs):
            for c in pod["containers"]:
                assert not c["image"].endswith(":latest"), \
                    f"{w['metadata']['name']} in {profile or 'defaults'}"


@_HELM_MISSING
def test_the_schema_rejects_a_moving_tag():
    """Belt and braces, and the schema is the half a user sees. Helm validates
    with Go's RE2, which has NO lookahead -- `(?!latest$)` is not a weak
    pattern there, it is a parse error that failed the entire chart."""
    schema = json.loads((CHART / "values.schema.json").read_text(encoding="utf-8"))
    tag = schema["properties"]["image"]["properties"]["tag"]
    assert "pattern" not in tag or "(?!" not in tag["pattern"]
    assert tag.get("not") == {"const": "latest"}


@_HELM_MISSING
def test_a_moving_tag_is_actually_refused():
    """Red-proofing the schema rather than trusting it: a check never observed
    to fail is not known to work."""
    r = subprocess.run(["helm", "template", "lh", str(CHART),
                        "--set", "image.tag=latest"],
                       capture_output=True, text=True)
    assert r.returncode != 0, "the schema let `latest` through"


# ---- the store -----------------------------------------------------------

@_HELM_MISSING
def test_every_workload_is_told_where_the_store_is():
    """Postgres, decided 2026-09-13. A pod that falls back to a local SQLite
    file writes results nobody will ever read."""
    for w, pod in pods(render("values-local.yaml")):
        if tier(w, pod) == "store":
            continue
        env = {e["name"] for e in pod["containers"][0].get("env", [])}
        assert {"LOCALHARNESS_STORE", "PGHOST", "PGDATABASE"} <= env, \
            f"{w['metadata']['name']} does not know where to write"


@_HELM_MISSING
def test_the_laptop_postgres_refuses_to_stand_beside_a_gpu_profile():
    """A single-replica in-chart database with no backup is a laptop
    convenience. Rendering it on the machine that holds the only card would
    make it look like infrastructure."""
    r = subprocess.run(["helm", "template", "lh", str(CHART),
                        "-f", str(CHART / "values-gpu-host.yaml"),
                        "--set", "postgres.dev.enabled=true"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "laptop convenience" in r.stderr


# ---- references between objects, not only properties of them -------------

@_HELM_MISSING
def test_every_secret_reference_resolves():
    """PHASE 0 MISSED THIS AND IT WOULD HAVE BLOCKED THE FIRST INSTALL. The
    assertions above check properties OF rendered objects; nothing checked that
    a reference BETWEEN them points at something. Three `secretKeyRef`s named a
    Secret no template created, so every pod would have sat in
    CreateContainerConfigError -- which reads as a broken cluster.

    This is `a reference must resolve` one level up from the prose version.
    """
    for profile in ("values-local.yaml",):
        docs = render(profile)
        created = {d["metadata"]["name"] for d in docs if d.get("kind") == "Secret"}
        declared = yaml.safe_load(
            (CHART / profile).read_text(encoding="utf-8")) or {}
        existing = (declared.get("postgres") or {}).get("existingSecret")
        if existing:
            created.add(existing)
        for w, pod in pods(docs):
            for c in pod["containers"]:
                for e in c.get("env", []):
                    ref = (e.get("valueFrom") or {}).get("secretKeyRef")
                    if not ref or ref.get("optional"):
                        continue
                    assert ref["name"] in created, (
                        f"{w['metadata']['name']} needs Secret {ref['name']!r}, "
                        f"which nothing creates. Rendered: {sorted(created)}")


@_HELM_MISSING
def test_a_profile_without_dev_ships_no_password():
    """The laptop password must not be reachable on a cluster that matters.

    An earlier cut enforced this by REFUSING to render, which also made the
    defaults unrenderable -- and a chart whose defaults do not install is a
    chart nobody can try. The contract instead: with dev off, no Secret is
    rendered at all and the pods reference a name the operator is expected to
    have created, which is the ordinary Helm convention.
    """
    docs = render()
    assert not [d for d in docs if d.get("kind") == "Secret"], \
        "a profile with dev off rendered a Secret, so it shipped a password"
    refs = {(e.get("valueFrom") or {}).get("secretKeyRef", {}).get("name")
            for _, pod in pods(docs) for c in pod["containers"]
            for e in c.get("env", [])}
    assert any(r and r.endswith("-postgres") for r in refs)


@_HELM_MISSING
def test_the_secret_name_follows_the_chart_name():
    """One question, one answer. The default was briefly a LITERAL
    `localharness-postgres` in values.yaml while the templates COMPUTED the
    same string, and a literal does not follow nameOverride: `--set
    nameOverride=other` left the gh-token reference tracking the new name while
    the postgres one stayed frozen, pointing at a Secret nobody would create.
    """
    out = subprocess.run(["helm", "template", "lh", str(CHART),
                          "--set", "nameOverride=other"],
                         capture_output=True, text=True, check=True).stdout
    docs = [d for d in yaml.safe_load_all(out) if d]
    refs = {(e.get("valueFrom") or {}).get("secretKeyRef", {}).get("name")
            for _, pod in pods(docs) for c in pod["containers"]
            for e in c.get("env", [])}
    refs = {r for r in refs if r}
    assert refs, "no secret references rendered, so this asserts nothing"
    assert all(r.startswith("other-") for r in refs), \
        f"a reference did not follow nameOverride: {sorted(refs)}"
