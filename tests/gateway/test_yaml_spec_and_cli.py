import json
import textwrap

import pytest

try:
    from perfhound.__main__ import main
except ImportError:            # gateway installed on its own: the full Perfhound CLI is not there
    main = None
needs_cli = pytest.mark.skipif(main is None, reason="full Perfhound CLI not installed")
from perfhound.gateway import Gateway
from perfhound.gateway.fetcher import RepoFetcher
from perfhound.gateway.sources import open_source
from perfhound.gateway.sources.swefficiency import SweTask, build_real_change_case, save_task_file
from perfhound.gateway.sources.yaml_spec import SpecError, load_spec

from .fixture_repo import BAD_TAG, GOOD_TAG
from .test_swefficiency import TASK, _git, build_merge_style_upstream


def write(tmp_path, text, name="perfhound.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


# ------------------------------------------------------------------ perfhound.yaml

def test_minimal_spec_for_any_repo(tmp_path):
    (case,) = load_spec(write(tmp_path, """
        repo: https://github.com/jhy/jsoup
        language: java
        good: jsoup-1.17.1
        bad: master
        benchmark:
          name: parse
          setup: mvn -q -DskipTests package
          command: java -cp target/classes Bench
          metric: {type: regex, pattern: "took ([0-9.]+) ms"}
          unit: ms
    """))
    assert case.repo == "https://github.com/jhy/jsoup" and case.language == "java"
    assert (case.good, case.bad, case.direction, case.ground_truth) == ("jsoup-1.17.1", "master", "slower", "none")
    b = case.benchmark
    assert (b.setup, b.command, b.unit, b.framework) == ("mvn -q -DskipTests package", "java -cp target/classes Bench", "ms", "command")
    assert b.metric == {"type": "regex", "pattern": "took ([0-9.]+) ms"}
    assert case.case_id == "yaml:jsoup@jsoup-1.17.1..master"


def test_several_cases_with_defaults_and_relative_repo(tmp_path):
    cases = load_spec(write(tmp_path, """
        defaults:
          repo: ./myrepo
          benchmark: python bench.py
        cases:
          - {name: first, good: v1, bad: v2}
          - {name: second, good: v2, bad: v3, direction: faster, culprit: abc1234}
    """))
    assert [c.case_id for c in cases] == ["yaml:first", "yaml:second"]
    assert cases[0].repo == str((tmp_path / "myrepo").resolve())
    assert cases[0].benchmark.command == "python bench.py" and cases[0].benchmark.metric == {"type": "wall_time"}
    assert (cases[1].direction, cases[1].culprit, cases[1].ground_truth) == ("faster", "abc1234", "reported")
    assert cases[1].for_localizer().culprit is None


@pytest.mark.parametrize("text,match", [
    ("repo: x\ngood: a\nbad: b\nbenchmak: python b.py\n", "unknown key"),          # typo is an error
    ("repo: x\ngood: a\n", "'bad' is required"),
    ("repo: x\ngood: a\nbad: b\nbenchmark: {command: c, metric: {type: regex, pattern: 'no group'}}\n", "capture group"),
    ("repo: x\ngood: a\nbad: b\nbenchmark: {command: c, metric: {type: magic}}\n", "metric type"),
    ("repo: x\ngood: a\nbad: b\nlanguage: rust\n", "language"),
    ("repo: x\ngood: a\nbad: b\nbenchmark: {setup: make}\n", "needs a 'command'"),
    ("- just\n- a list\n", "mapping"),
    ("repo: [unclosed\n", "not valid YAML"),
])
def test_bad_specs_give_clear_errors(tmp_path, text, match):
    with pytest.raises(SpecError, match=match):
        load_spec(write(tmp_path, text))


def test_yaml_source_and_gateway_end_to_end(fixture_repo, tmp_path):
    spec = write(tmp_path, f"""
        repo: {fixture_repo.path.as_posix()}
        good: {GOOD_TAG}
        bad: {BAD_TAG}
        benchmark: python -c "import mathops"
    """)
    (case,) = open_source("yaml", path=spec).cases()
    cands = Gateway.for_case(case, cache=False).candidates_for(case)
    assert [c.sha for c in cands] == fixture_repo.first_parent_shas()


# ------------------------------------------------------------------ CLI

@needs_cli
def test_cli_sources_and_cases(tmp_path, capsys):
    assert main(["sources"]) == 0
    assert {"yaml", "swefficiency-real", "git"} <= set(capsys.readouterr().out.split())
    spec = write(tmp_path, "cases:\n  - {name: a, repo: r, good: g, bad: b, culprit: c0ffee}\n")
    assert main(["cases", "yaml", "-o", f"path={spec}"]) == 0
    out = capsys.readouterr().out
    assert "yaml:a" in out and "c0ffee" not in out                  # truth hidden by default
    assert main(["cases", "yaml", "-o", f"path={spec}", "--show-truth"]) == 0
    assert "c0ffee" in capsys.readouterr().out


@needs_cli
def test_cli_candidates_table_and_json(fresh_fixture_repo, tmp_path, capsys):
    repo = fresh_fixture_repo
    (repo.path / "mathops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8", newline="\n")
    repo.git("commit", "-qam", "Fix add ✓ café")                     # unicode must not crash the console
    spec = write(tmp_path, f"repo: {repo.path.as_posix()}\ngood: {GOOD_TAG}\nbad: HEAD\n")
    assert main(["candidates", str(spec)]) == 0
    out = capsys.readouterr().out
    assert "8 candidates" in out and "mathops.add" in out and "Fix add" in out
    assert main(["candidates", str(spec), "--json", "--no-analyze"]) == 0
    rows = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert len(rows) == 8 and rows[0]["position"] == 0


@needs_cli
def test_cli_reports_errors_in_one_line(tmp_path, capsys):
    assert main(["candidates", str(tmp_path / "missing.yaml")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("perfhound: error:") and "does not exist" in err


# ------------------------------------------------------------------ SWE-fficiency real changes (no injection)

def test_real_change_case_window_and_no_git_writes(tmp_path):
    upstream = build_merge_style_upstream(tmp_path / "up")   # initial, PR#123 (fast compute), 6 more merges
    refs_before = _git(upstream, "for-each-ref")
    case = build_real_change_case(upstream, TASK, n=4, position_k=1, ref="main")
    opt = _git(upstream, "log", "--first-parent", "--format=%H", "-E", "--grep=^Merge pull request #123 ", "main")
    assert case.culprit == opt and case.ground_truth == "reported" and case.direction == "faster"
    assert case.good == _git(upstream, "rev-parse", "main~7")                # initial commit
    assert case.expected_magnitude == pytest.approx(1 - 1 / 3.0)             # 3x faster -> time -66.7 %
    cands = Gateway(upstream, cache=False).candidates_for(case)
    assert len(cands) == 4 and cands[0].sha == opt
    assert "compute" in " ".join(cands[0].changed_functions)
    assert _git(upstream, "for-each-ref") == refs_before                     # read-only: no branches written


def test_real_change_adapter_positions_and_failures(tmp_path):
    upstream = build_merge_style_upstream(tmp_path / "up")   # only ONE commit before PR #123
    tf = tmp_path / "tasks.json"
    save_task_file([TASK, SweTask("org__proj-999", "org/proj", 999, "pass")], tf)
    a = open_source("swefficiency-real", tasks_file=tf, repos=None, n=4, positions=[1, 2],
                    fetcher=RepoFetcher(tmp_path / "repos"), repo_url=lambda r: str(upstream), ref="main")
    cases = list(a.cases())
    assert [c.metadata["truth_position"] for c in cases] == [1]     # k=2 needs 2 commits before the PR
    assert sorted(f.reason for f in a.failures) == ["pr_not_found", "pr_not_found", "short_history"]
    assert not (tmp_path / "repos").exists()                        # local repo used in place, never cloned
    assert cases[0].repo == str(upstream)                           # case records the repo it was given (URL or path)


def test_real_change_positions_are_reproducible(tmp_path):
    upstream = build_merge_style_upstream(tmp_path / "up")
    tf = tmp_path / "tasks.json"
    save_task_file([TASK], tf)

    def ids(seed):
        a = open_source("swefficiency-real", tasks_file=tf, repos=None, n=2, positions_per_task=1, seed=seed,
                        repo_url=lambda r: str(upstream), ref="main")
        return [c.case_id for c in a.cases()] + [f.instance_id for f in a.failures]

    assert ids(7) == ids(7)


def test_task_without_workload_is_skipped(tmp_path):
    """Found in the real data: one SWE-fficiency task (xarray-7374) has no workload."""
    upstream = build_merge_style_upstream(tmp_path / "up")
    empty = SweTask(TASK.instance_id, TASK.repo, TASK.pr_number, workload="", expert_speedup=2.0)
    failure = build_real_change_case(upstream, empty, n=4, position_k=1, ref="main")
    assert failure.reason == "no_workload"


def test_compare_link_is_a_case(tmp_path):
    from perfhound.gateway.sources.yaml_spec import SpecError, load_spec

    spec = tmp_path / "p.yaml"
    spec.write_text("compare: https://github.com/jhy/jsoup/compare/jsoup-1.17.1...master\nbenchmark: java Bench\n")
    (case,) = load_spec(spec)
    assert (case.repo, case.good, case.bad) == ("https://github.com/jhy/jsoup", "jsoup-1.17.1", "master")
    spec.write_text("compare: https://github.com/jhy/jsoup/compare/a...b\nrepo: x\ngood: a\nbad: b\n")
    with pytest.raises(SpecError, match="either 'compare'"):
        load_spec(spec)
    spec.write_text("compare: https://github.com/jhy/jsoup/pull/5\n")
    with pytest.raises(SpecError, match="compare link"):
        load_spec(spec)
