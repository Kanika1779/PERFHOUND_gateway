import json
from pathlib import Path

import pytest

from perfhound.gateway import Gateway
from perfhound.gateway.errors import GatewayError
from perfhound.gateway.fetcher import RepoFetcher
from perfhound.gateway.snapshot import find_tells
from perfhound.gateway.sources import open_source
from perfhound.gateway.sources import swefficiency as swe
from perfhound.gateway.sources.swefficiency import SweTask, save_task_file

from .fixture_repo import _git, git_env

SLOW = "def compute(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"
FAST = "def compute(n):\n    return n * (n - 1) // 2\n"
FAST2 = "def compute(n):\n    return (n * (n - 1)) >> 1\n"


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def commit(root: Path, msg: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


def build_upstream(root: Path, rewrite_after_pr: bool = False) -> Path:
    """Upstream with PR #123 (makes compute fast) followed by 4 mainline commits."""
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "uploadpack.allowFilter", "true")
    _git(root, "config", "uploadpack.allowAnySHA1InWant", "true")
    write(root, "lib/__init__.py", "")
    write(root, "lib/core.py", SLOW + "\n\ndef other():\n    return 1\n")
    write(root, "tests/test_core.py", "def test_compute():\n    pass\n")
    commit(root, "initial")
    _git(root, "checkout", "-q", "-b", "fast")
    write(root, "lib/core.py", FAST + "\n\ndef other():\n    return 1\n")
    write(root, "tests/test_core.py", "def test_compute():\n    assert True\n")
    commit(root, "Use closed form in compute")
    _git(root, "checkout", "-q", "main")
    _git(root, "merge", "-q", "--no-ff", "-m", "Merge pull request #123 from dev/fast\n\nFaster compute", "fast")
    write(root, "lib/util.py", "def helper():\n    return 2\n")
    commit(root, "Add util helper")
    _git(root, "checkout", "-q", "-b", "side")
    write(root, "docs.md", "docs\n")
    commit(root, "Docs")
    _git(root, "checkout", "-q", "main")
    _git(root, "merge", "-q", "--no-ff", "-m", "Merge pull request #124 from dev/docs", "side")
    if rewrite_after_pr:
        write(root, "lib/core.py", FAST2 + "\n\ndef other():\n    return 1\n")
        commit(root, "Micro-optimize compute")
    else:
        write(root, "lib/core.py", FAST + "\n\ndef other():\n    return 2\n")
        commit(root, "Change other()")
    write(root, "README.md", "readme\n")
    commit(root, "Readme")
    return root


TASK = SweTask(instance_id="org__proj-123", repo="org/proj", pr_number=123,
               workload="from lib.core import compute\ncompute(10**6)\n", expert_speedup=3.0)


@pytest.fixture
def setup(tmp_path):
    def make(rewrite=False, tasks=(TASK,)):
        upstream = build_upstream(tmp_path / ("up_rw" if rewrite else "up"), rewrite)
        tf = tmp_path / "tasks.json"
        save_task_file(list(tasks), tf)
        return upstream, tf, RepoFetcher(tmp_path / "repos")
    return make


def adapter(upstream, tf, fetcher, **kw):
    kw.setdefault("message_strategy", "neutral")   # APRCL-style extra commit unless a test says otherwise
    return open_source("swefficiency", tasks_file=tf, repos=None, fetcher=fetcher,
                       repo_url=lambda r: upstream.as_uri(), **kw)


def test_injected_case_has_culprit_at_requested_position(setup):
    upstream, tf, fetcher = setup()
    a = adapter(upstream, tf, fetcher, n_after=4, positions=[3])
    (case,) = list(a.cases())
    assert a.failures == []
    assert case.ground_truth == "injected" and case.language == "python"
    assert case.expected_magnitude == pytest.approx(2.0)          # 3x speedup reverted -> 200 % slower
    assert case.metadata["truth_position"] == 3

    gw = Gateway.for_case(case, fetcher=fetcher, cache=False)
    cands = gw.candidates_for(case.for_localizer())
    assert len(cands) == 5 and cands[2].sha == case.culprit
    culprit = cands[2]
    assert culprit.changed_functions == ("lib.core.compute",)
    assert [f.path for f in culprit.files] == ["lib/core.py"]       # test file is NOT reverted
    assert "+    for i in range(n):" in culprit.diff                  # the slow loop is back
    assert culprit.subject == "Refactor lib.core"


def test_replayed_commits_are_real_and_culprit_has_no_metadata_tell(setup):
    upstream, tf, fetcher = setup()
    (case,) = list(adapter(upstream, tf, fetcher, n_after=4, positions=[2]).cases())
    cands = Gateway.for_case(case, fetcher=fetcher, cache=False).candidates_for(case)
    upstream_msgs = _git(upstream, "log", "--first-parent", "--format=%s", "main~4..main").splitlines()[::-1]
    replayed = [c.subject for c in cands if c.sha != case.culprit]
    assert replayed == upstream_msgs
    assert find_tells(cands, [case.culprit]) == []
    mapping = case.metadata["truth_upstream_of"]
    assert mapping[case.culprit] == "INJECTED" and len(mapping) == 5


def test_built_cases_are_reused(setup, monkeypatch):
    upstream, tf, fetcher = setup()
    first = list(adapter(upstream, tf, fetcher, n_after=4, positions=[1, 5]).cases())
    monkeypatch.setattr(swe, "build_injected_case", lambda *a, **k: pytest.fail("should reuse stored case"))
    assert list(adapter(upstream, tf, fetcher, n_after=4, positions=[1, 5]).cases()) == first


def test_revert_message_strategy(setup):
    upstream, tf, fetcher = setup()
    (case,) = list(adapter(upstream, tf, fetcher, n_after=4, positions=[1], message_strategy="revert").cases())
    msg = _git(Path(case.repo), "log", "-1", "--format=%B", case.culprit)
    assert msg.startswith('Revert "Merge pull request #123')


def test_conflicts_are_reported_not_hidden(setup):
    upstream, tf, fetcher = setup(rewrite=True)
    a = adapter(upstream, tf, fetcher, n_after=4, positions=[1, 4])
    assert list(a.cases()) == []
    assert sorted(f.reason for f in a.failures) == ["replay_conflict", "revert_conflict"]


def test_other_failures(setup):
    upstream, tf, fetcher = setup(tasks=(TASK, SweTask("org__proj-999", "org/proj", 999, "pass")))
    a = adapter(upstream, tf, fetcher, n_after=50, positions=[1])
    assert list(a.cases()) == []
    assert sorted(f.reason for f in a.failures) == ["pr_not_found", "short_history"]


def test_piggyback_folds_revert_into_a_real_commit(setup):
    upstream, tf, fetcher = setup()
    a = adapter(upstream, tf, fetcher, n_after=4, positions=[2], message_strategy="piggyback")
    (case,) = list(a.cases())
    assert case.case_id.endswith("_k02_pb")
    cands = Gateway.for_case(case, fetcher=fetcher, cache=False).candidates_for(case)
    assert len(cands) == 4 and cands[1].sha == case.culprit           # no extra commit
    from perfhound.gateway.local_git import parse_git_date
    real = [line.split("\x1f") for line in
            _git(upstream, "log", "--first-parent", "--format=%s%x1f%an%x1f%aI%x1f%at", "main~4..main").splitlines()[::-1]]
    # compare moments, not text: newer git prints UTC as "Z", older as "+00:00"
    assert [(c.subject, c.author, c.timestamp) for c in cands] == [(s, a, parse_git_date(i, e)) for s, a, i, e in real]
    assert cands[1].subject == "Merge pull request #124 from dev/docs"
    assert "lib.core.compute" in cands[1].changed_functions and "+    for i in range(n):" in cands[1].diff
    assert case.metadata["truth_upstream_of"][case.culprit].startswith("PIGGYBACK:")
    assert find_tells(cands, [case.culprit]) == []


def build_merge_style_upstream(root: Path) -> Path:
    """Like sympy: after PR #123 every main-line commit is 'Merge pull request #...'."""
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "uploadpack.allowFilter", "true")
    _git(root, "config", "uploadpack.allowAnySHA1InWant", "true")
    write(root, "lib/core.py", SLOW)
    commit(root, "initial")
    for n, (path, text) in enumerate([("lib/core.py", FAST)] + [(f"f{i}.txt", f"{i}\n") for i in range(6)]):
        _git(root, "checkout", "-q", "-b", f"b{n}")
        write(root, path, text)
        commit(root, f"work {n}")
        _git(root, "checkout", "-q", "main")
        _git(root, "merge", "-q", "--no-ff", "-m", f"Merge pull request #{123 + n} from dev/b{n}", f"b{n}")
    return root


def test_neutral_commit_is_flagged_on_a_merge_style_main_line(tmp_path):
    """The leak found on sympy: one 'Refactor ...' among 'Merge pull request #...' commits."""
    upstream = build_merge_style_upstream(tmp_path / "up")
    tf = tmp_path / "tasks.json"
    save_task_file([TASK], tf)
    tells = {}
    for strategy in ("neutral", "piggyback"):
        fetcher = RepoFetcher(tmp_path / f"repos_{strategy}")
        (case,) = list(adapter(upstream, tf, fetcher, n_after=6, positions=[3], message_strategy=strategy).cases())
        cands = Gateway.for_case(case, fetcher=fetcher, cache=False).candidates_for(case)
        tells[strategy] = find_tells(cands, [case.culprit])
    assert any("does not look like the others" in t.reason for t in tells["neutral"])
    assert tells["piggyback"] == []


def test_never_injects_into_a_local_repo(setup):
    upstream, tf, fetcher = setup()
    a = open_source("swefficiency", tasks_file=tf, repos=None, fetcher=fetcher, repo_url=lambda r: str(upstream))
    with pytest.raises(GatewayError, match="only runs on clones"):
        list(a.cases())


def test_aprcl_instances_adapter(tmp_path):
    root = tmp_path / "APRCL"
    (root / "data" / "instances").mkdir(parents=True)
    (root / "data" / "workloads").mkdir(parents=True)
    (root / "data" / "workloads" / "t.py").write_text("print('wl')\n", encoding="utf-8")
    inst = {
        "instance_id": "sympy__sympy-1__n20_k05", "source_instance": "sympy__sympy-1", "repo": "sympy/sympy",
        "repo_path": "work/sympy", "branch": "aprcl/x", "workload_file": "data/workloads/t.py",
        "good_commit": "a" * 40, "bad_commit": "b" * 40, "culprit_commit": "c" * 40, "culprit_position": 5,
        "candidates": [], "upstream_of": {"c" * 40: "INJECTED"}, "upstream_opt_commit": "a" * 40,
        "regression_type": "revert_expert_optimization", "expected_magnitude": 1.84,
        "message_strategy": "neutral", "workload_origin": "synthetic",
    }
    (root / "data" / "instances" / "x.json").write_text(json.dumps(inst), encoding="utf-8")
    (case,) = list(open_source("aprcl-instances", instances_dir=root / "data" / "instances").cases())
    assert case.repo == str(root / "work" / "sympy")
    assert case.culprit == "c" * 40 and case.ground_truth == "injected"
    assert case.expected_magnitude == pytest.approx(0.84)
    assert case.benchmark.workload == "print('wl')\n"
    assert "c" * 40 not in case.for_localizer().to_json()


def test_injected_commit_keeps_the_neighbours_timezone(tmp_path, monkeypatch):
    """Found on Windows (IST laptop): the injected commit was always +0000 while all real ones were +0530."""
    monkeypatch.setenv("TZ", "IST-5:30")          # POSIX TZ string: commits below get +0530
    upstream = build_upstream(tmp_path / "up")
    tf = tmp_path / "tasks.json"
    save_task_file([TASK], tf)
    fetcher = RepoFetcher(tmp_path / "repos")
    (case,) = list(adapter(upstream, tf, fetcher, n_after=4, positions=[2], message_strategy="neutral").cases())
    cands = Gateway.for_case(case, fetcher=fetcher, cache=False).candidates_for(case)
    offsets = {c.timestamp.utcoffset() for c in cands}
    assert len(offsets) == 1 and next(iter(offsets)).total_seconds() == 5.5 * 3600
    assert find_tells(cands, [case.culprit]) == []
