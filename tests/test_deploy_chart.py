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


@_HELM_MISSING
def test_the_fan_out_gives_each_worker_its_own_slice():
    """Every pod of a plain Job gets identical arguments, so `parallelism: 4`
    without an index is four workers doing the same work -- four times the API
    budget and four writers racing on the same rows, for one result.

    The chart claimed a fan-out the code could not honour until `--shard`
    existed. This asserts the two agree.
    """
    docs = render("values-local.yaml")
    jobs = [(w, pod) for w, pod in pods(docs) if tier(w, pod) == "inspect"]
    assert jobs, "no inspect job rendered, so this asserts nothing"
    for w, pod in jobs:
        spec = w["spec"]
        assert spec.get("completionMode") == "Indexed", w["metadata"]["name"]
        assert spec["completions"] == spec["parallelism"], (
            "an Indexed job with fewer completions than parallelism leaves "
            "indices nobody runs, so part of the candidate list is skipped")
        c = pod["containers"][0]
        assert "--shard" in " ".join(c.get("args", [])), "no shard passed"
        names = {e["name"] for e in c["env"]}
        assert "JOB_COMPLETION_INDEX" in names, "the shard has no index to use"


@_HELM_MISSING
def test_every_workload_that_reaches_github_carries_a_token():
    """The sweep had the token and inspect did not, so the first real fan-out
    failed all 125 candidates with "populate the GH_TOKEN environment
    variable" -- and it failed QUIETLY, because a per-candidate error still
    lets the Job exit 0 with `{"inspected": []}`.

    Nothing checked that a workload which talks to GitHub can authenticate.
    """
    for w, pod in pods(render("values-local.yaml")):
        c = pod["containers"][0]
        invocation = " ".join(c.get("args", []) + c.get("command", []))
        if "discover" not in invocation:
            continue
        names = {e["name"] for e in c.get("env", [])}
        assert "GH_TOKEN" in names, (
            f"{w['metadata']['name']} runs discover and cannot authenticate")


# ---- where the work executes, which is three answers ---------------------

def _env(pod: dict) -> dict:
    return {e["name"]: e.get("value", "")
            for e in pod["containers"][0].get("env", [])}


@_HELM_MISSING
def test_every_workload_says_where_its_work_executes():
    """A receipt that cannot name the place reads as "the cluster measured it"
    whether the cluster measured it or asked somebody else to."""
    for w, pod in pods(render("values-gpu-host.yaml")):
        if tier(w, pod) == "store":
            continue
        got = _env(pod).get("LOCALHARNESS_WHERE", "")
        assert got, f"{w['metadata']['name']} names no place"
        assert got in ("in-pod", "gpu-node", "host"), got


@_HELM_MISSING
def test_the_cheap_tiers_say_in_pod_and_the_gpu_tier_does_not():
    """The two are not the same exam: one has a card in view and one does
    not, and comparable() refuses to rank across them."""
    where = {tier(w, pod): _env(pod).get("LOCALHARNESS_WHERE")
             for w, pod in pods(render("values-gpu-host.yaml"))}
    assert where["sweep"] == "in-pod"
    assert where["inspect"] == "in-pod"
    assert where["measure"] == "gpu-node"


@_HELM_MISSING
def test_a_place_the_chart_cannot_deliver_is_refused_rather_than_rendered():
    """`host` means a pod holding a lane's identity while `lh` outside the
    cluster does the work. The receipt half exists and the dispatch half does
    not, so rendering it would produce a pod that measures the pod and labels
    it the host -- a part's cost reported as the whole's."""
    argv = ["helm", "template", "lh", str(CHART),
            "-f", str(CHART / "values-gpu-host.yaml"), "--set", "measure.where=host"]
    proc = subprocess.run(argv, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "dispatch half is not" in proc.stderr


@_HELM_MISSING
def test_a_gpu_job_that_claims_to_run_in_a_pod_is_refused():
    """The negative half: the guard must fire on the contradiction too, or it
    only catches the case its author was thinking about."""
    argv = ["helm", "template", "lh", str(CHART),
            "-f", str(CHART / "values-gpu-host.yaml"),
            "--set", "measure.where=in-pod"]
    proc = subprocess.run(argv, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "no card in view" in proc.stderr


@_HELM_MISSING
def test_a_place_nothing_recognises_is_refused_by_the_schema():
    argv = ["helm", "template", "lh", str(CHART), "--set", "measure.where=laptop"]
    proc = subprocess.run(argv, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "must be one of" in proc.stderr


def test_the_readme_accounts_for_every_profile_the_chart_ships():
    """A profile added without a row reads as one nobody thought about, and a
    row left behind after a profile goes reads as a cluster somebody tried.
    No helm needed: this is a question about two files."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    shipped = {p.name for p in CHART.glob("values-*.yaml")}
    assert shipped == set(PROFILES), (
        f"the chart ships {sorted(shipped)} and this suite tests "
        f"{sorted(PROFILES)}")
    for name in shipped:
        assert name in readme, f"{name} is shipped and the README never says so"
    # AND EACH ONE SAYS WHICH IT IS. "rendered" is not a synonym for "works",
    # and the whole point of the table is that a reader can tell them apart.
    for line in readme.splitlines():
        for name in shipped:
            if f"`{name}`" in line:
                assert "RUN" in line or "RENDERED" in line, (
                    f"{name} is listed with no status: say whether anything "
                    f"has ever run it")


# ---- the judge tier, which runs on a model outside the cluster -----------

JUDGE = ("--set", "judge.enabled=true",
         "--set", "judge.gateway=http://host.docker.internal:4000")


def _render(*flags):
    argv = ["helm", "template", "lh", str(CHART), *flags]
    return subprocess.run(argv, capture_output=True, text=True)


def _judge_pod(docs):
    for w, pod in pods(docs):
        if tier(w, pod) == "judge":
            return w, pod
    raise AssertionError("the chart rendered no judge workload")


@_HELM_MISSING
def test_the_judge_says_its_work_runs_on_a_host_outside_the_cluster():
    """The pod holds the tier's identity and its queue position; the model
    doing the scoring runs somewhere else. Calling this in-pod would be the
    cluster taking credit for somebody else's arithmetic."""
    out = _render(*JUDGE)
    assert out.returncode == 0, out.stderr
    _, pod = _judge_pod([d for d in yaml.safe_load_all(out.stdout) if d])
    assert _env(pod)["LOCALHARNESS_WHERE"] == "host"


@_HELM_MISSING
def test_the_judge_is_a_queue_of_one():
    """The text server hot-swaps models through a single queue, so a second
    judge pod does not double the throughput -- it interleaves, and each switch
    costs a full model load. The fan-out belongs to the tier with no model."""
    out = _render(*JUDGE)
    w, _ = _judge_pod([d for d in yaml.safe_load_all(out.stdout) if d])
    assert w["spec"]["parallelism"] == 1
    assert w["spec"]["completions"] == 1


@_HELM_MISSING
def test_the_judge_is_told_where_the_model_is_served():
    """A pod's localhost is the pod. Without this the tier reaches for a
    gateway inside its own container and finds nothing."""
    out = _render(*JUDGE)
    _, pod = _judge_pod([d for d in yaml.safe_load_all(out.stdout) if d])
    args = " ".join(pod["containers"][0]["args"])
    assert "--gateway" in args and "host.docker.internal:4000" in args


@_HELM_MISSING
def test_a_judge_with_nowhere_to_ask_is_refused_rather_than_rendered():
    """The negative half, and it is a seam: `judge.enabled` and `judge.gateway`
    are two settings that each look fine alone, and the agreement between them
    is owned by neither."""
    out = _render("--set", "judge.enabled=true")
    assert out.returncode != 0
    assert "pointing at its own localhost" in out.stderr


@_HELM_MISSING
def test_the_judge_does_not_score_without_running_its_control():
    """A tier that ranks unattended on a schedule has nobody present to doubt
    it, so --no-control must never reach a Job."""
    out = _render(*JUDGE)
    _, pod = _judge_pod([d for d in yaml.safe_load_all(out.stdout) if d])
    args = " ".join(pod["containers"][0]["args"])
    assert "--no-control" not in args
    assert "--runs" in args


@_HELM_MISSING
def test_the_judge_never_asks_for_a_card():
    """It calls a gateway and touches no weights. A pod holding a card to read
    prose starves the only tier that needs one."""
    out = _render(*JUDGE, "--set", "gpu.enabled=true",
                  "--set", "postgres.dev.enabled=false")
    _, pod = _judge_pod([d for d in yaml.safe_load_all(out.stdout) if d])
    limits = pod["containers"][0].get("resources", {}).get("limits", {})
    assert "nvidia.com/gpu" not in limits


# ---- a tier Job is a run, not a deployment -------------------------------

def test_every_tier_job_is_named_per_release_revision():
    """A Job's pod template is IMMUTABLE. With a fixed name the second `helm
    upgrade` that touches image, args, env or resources is refused by the API
    server, and the chart installs once and then accepts no change. Worse, the
    refusal is partial: the CronJob in the same release took its new schedule
    while helm exited non-zero. Issue #174.

    `helm template` always renders revision 1, so this reads the templates: a
    rendered pair cannot tell a constant name from a revisioned one.
    """
    paths = sorted((CHART / "templates").glob("job-*.yaml"))
    assert paths, "no job template found; a loop over nothing passes"
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert ".Release.Revision" in text, (
            f"{path.name} names a Job without the release revision, so an "
            f"upgrade will try to edit a frozen pod template")
        assert "ttlSecondsAfterFinished" in text, (
            f"{path.name} names a Job per revision and never cleans it up")


@_HELM_MISSING
def test_a_finished_run_cleans_itself_up():
    """Named per revision, Jobs accumulate one per upgrade forever. The RECORD
    of what ran lives in the store, so there is nothing in the cluster to
    keep."""
    jobs = [d for d in render("values-local.yaml") if d.get("kind") == "Job"]
    assert jobs, "no Job rendered; a loop over nothing passes"
    for d in jobs:
        assert d["spec"]["ttlSecondsAfterFinished"] > 0, d["metadata"]["name"]


# ---- the seam: what the chart invokes, and what the CLI accepts ----------

#: The image's ENTRYPOINT is `lh`, so a container either passes its argv
#: straight through, or wraps it in a shell, or escapes to another program with
#: `--`. Every container must be one of those three, and the third is named
#: rather than inferred: a tier that quietly matches none would be skipped by
#: this scanner, which is how a scanner comes to check nothing.
_ESCAPE = "--"


def _invocations(docs) -> tuple[list[list[str]], list[list[str]]]:
    """(lh argvs, escapes) over every container the chart renders.

    `${JOB_COMPLETION_INDEX}` is substituted with a plausible value, because
    the shell does that before the program ever sees it.
    """
    import shlex
    out, escaped = [], []
    for _, pod in pods(docs):
        for container in pod["containers"]:
            argv = [a.replace("${JOB_COMPLETION_INDEX}", "0")
                    for a in (container.get("args") or [])]
            if not argv:
                continue
            command = container.get("command") or []
            if command and command[0] in ("sh", "bash"):
                words = shlex.split(" ".join(argv))
                if "lh" not in words:
                    escaped.append(argv)
                    continue
                out.append(words[words.index("lh") + 1:])
            elif argv[0] == _ESCAPE:
                escaped.append(argv)
            elif command:
                escaped.append(argv)     # a container that is not this program
            else:
                out.append(argv)         # straight to the ENTRYPOINT
    return out, escaped


@_HELM_MISSING
def test_every_command_the_chart_runs_is_one_this_program_has():
    """THE SEAM, and this project has paid for it four times in one afternoon.
    The chart renders a shell string; the CLI defines flags; each half is
    checked against its own spec and the agreement between them is owned by
    neither, so a renamed flag is a green chart, a green test suite, and a pod
    that dies in a cluster.

    A scanner, not a per-flag assertion: it covers the tiers that exist and the
    ones nobody has written yet.
    """
    from harness import cli

    docs = render("values-local.yaml") + [
        d for d in yaml.safe_load_all(_render(
            *JUDGE, "--set", "gpu.enabled=true",
            "--set", "postgres.dev.enabled=false").stdout) if d]
    argvs, escaped = _invocations(docs)
    # NOT A BARE `assert argvs`. The first cut of this scanner matched only the
    # shell-wrapped form and silently skipped the sweep, which is the tier that
    # runs most often -- a scanner checking two of four workloads and reporting
    # green. Count what it saw against what the chart renders.
    tiers = {tier(w, pod) for w, pod in pods(docs)} - {"store"}
    assert len(argvs) + len(escaped) >= len(tiers), (
        f"{len(tiers)} tiers render, {len(argvs)} lh invocations and "
        f"{len(escaped)} escapes were found: something was skipped")
    assert argvs, "the chart rendered no lh invocation; this test checks nothing"
    parser = cli.build_parser()
    for argv in argvs:
        try:
            parser.parse_args(argv)
        except SystemExit as exc:  # argparse exits rather than raising
            raise AssertionError(
                f"the chart runs `lh {' '.join(argv)}`, which this CLI does "
                f"not accept ({exc})") from None


@_HELM_MISSING
def test_the_scanner_notices_a_flag_the_cli_does_not_have():
    """Red-proofed. A scanner never observed to fail is not known to work, and
    this one passes trivially if the extraction quietly finds nothing."""
    from harness import cli

    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["discover", "--judge", "--from-a-teapot"])


# ---- a lane pod reads its winner rather than being told it (#148 phase 5) --

GPU = ("--set", "gpu.enabled=true", "--set", "postgres.dev.enabled=false")


@_HELM_MISSING
def test_the_measure_job_reads_the_winner_rather_than_naming_one():
    """A candidate baked into the chart would be a THIRD copy of an answer that
    already lives in the receipts and in cli.py's constants -- and a pod built
    around a transcribed constant is built around whatever was true the last
    time somebody edited a source file."""
    out = _render(*GPU)
    assert out.returncode == 0, out.stderr
    docs = [d for d in yaml.safe_load_all(out.stdout) if d]
    measure = [a for a in _invocations(docs)[1] if "evals.run" in a]
    assert measure, "the measure job rendered no run"
    assert "--from-winners" in measure[0]
    assert "--candidates" not in measure[0]


@_HELM_MISSING
def test_a_pinned_candidate_is_possible_and_deliberate():
    """Reading the winner is the default, not the only option: pinning one is
    a decision somebody can make, and then the chart says so explicitly rather
    than both."""
    out = _render(*GPU, "--set", "measure.candidate=mflux:flux2-klein-4b")
    docs = [d for d in yaml.safe_load_all(out.stdout) if d]
    measure = [a for a in _invocations(docs)[1] if "evals.run" in a][0]
    assert "--candidates" in measure
    assert "mflux:flux2-klein-4b" in measure
    assert "--from-winners" not in measure


@_HELM_MISSING
def test_the_lane_the_measure_job_runs_is_a_value_not_a_literal():
    out = _render(*GPU, "--set", "measure.lane=video")
    docs = [d for d in yaml.safe_load_all(out.stdout) if d]
    measure = [a for a in _invocations(docs)[1] if "evals.run" in a][0]
    assert measure[measure.index("--modality") + 1] == "video"
