"""asv-runner source: parsing real issue formats, benchmark source extraction, windows, no answer leak."""

import json

from perfhound.gateway.gitcmd import run_git
from perfhound.gateway.sources import open_source
from perfhound.gateway.sources.asv_runner import (AsvRegression, benchmark_source, parse_issue, save_regressions)

SHA = "477bee05b1b677eca97209efe8e7d6881f9e64aa"
BODY = """[PR #69131](https://github.com/pandas-dev/pandas/pull/69131)

cc @jbrockmendel

 - [ ] [sparse.Arithmetic.time_divide](https://pandas-dev.github.io/asv-runner/#sparse.Arithmetic.time_divide)
   - [ ] [dense_proportion=0.1, fill_value=nan](https://x/#p) - 65.025% (1.275ms)

 - [ ] [series_methods.NanOps.time_func](https://x)
   - [ ] [func='max', N=1000000, dtype='float64'](https://x) - 57.622% (403.072us)
   - [ ] [func='min', N=1000000, dtype='float64'](https://x) - 74.950% (523.085us)
 - [ ] [libs.ScalarListLike.time_is_list_like](https://x) - 12.5% (1.1us)
"""


def test_parse_real_issue_formats():
    r = parse_issue({"number": 169, "title": f"Commit {SHA}", "body": BODY, "created_at": "2026-09-23T04:00:04Z",
                     "html_url": "https://github.com/pandas-dev/asv-runner/issues/169"})
    assert r.commit == SHA and r.pr == 69131 and r.issue == 169
    assert [(b["name"], b["pct"]) for b in r.benchmarks] == [
        ("sparse.Arithmetic.time_divide", 65.025), ("series_methods.NanOps.time_func", 57.622),
        ("series_methods.NanOps.time_func", 74.95), ("libs.ScalarListLike.time_is_list_like", 12.5)]
    assert r.benchmarks[2]["params"] == "func='min', N=1000000, dtype='float64'" and r.max_pct == 74.95
    assert parse_issue({"number": 168, "title": "Benchmark run failures", "body": "- Run for x produced no results"}) is None
    assert parse_issue({"number": 5, "title": f"Commit {SHA}", "body": BODY, "pull_request": {}}) is None
    assert parse_issue({"number": 6, "title": f"Commit {SHA}", "body": "no benchmarks listed"}) is None


def _repo(tmp_path, n=8, culprit_at=5):
    repo = tmp_path / "pandas"
    (repo / "asv_bench" / "benchmarks").mkdir(parents=True)
    run_git(repo, "init", "-q", "-b", "main")
    shas = []
    for i in range(n):
        (repo / "asv_bench" / "benchmarks" / "sparse.py").write_text(
            "import numpy as np\n\n\nclass Other:\n    def time_x(self):\n        pass\n\n\n"
            "class Arithmetic:\n    params = ([0.1], [0, 1])\n\n    def setup(self, p, f):\n        self.a = 1\n\n"
            f"    def time_divide(self, p, f):\n        self.a / 2  # v{i}\n")
        (repo / f"m{i}.py").write_text(f"x = {i}\n")
        run_git(repo, "add", "-A")
        run_git(repo, "-c", "user.name=d", "-c", "user.email=d@x", "commit", "-q", "-m", f"change {i}")
        shas.append(run_git(repo, "rev-parse", "HEAD").stdout.strip())
    return repo, shas


def test_benchmark_source_is_the_class(tmp_path):
    repo, shas = _repo(tmp_path)
    src = benchmark_source(repo, shas[-1], "sparse.Arithmetic.time_divide")
    assert src.startswith("class Arithmetic") and "time_divide" in src and "class Other" not in src
    assert benchmark_source(repo, shas[-1], "nosuch.Bench.time_x") is None


def test_adapter_builds_windows_without_leaking(tmp_path):
    repo, shas = _repo(tmp_path, n=12)
    regs = [AsvRegression(issue=169, commit=shas[6], pr=69131, created_at="2026-09-23", url="u",
                          benchmarks=[{"name": "sparse.Arithmetic.time_divide", "params": "p=0.1", "pct": 65.0,
                                       "abs": "1ms"}]),
            AsvRegression(issue=170, commit=shas[0], pr=None, created_at="2026-09-24", url="u",
                          benchmarks=[{"name": "sparse.Arithmetic.time_divide", "params": "", "pct": 10.0, "abs": "1"}])]
    f = tmp_path / "regs.json"
    save_regressions(regs, f, source="test")
    src = open_source("asv-runner", regressions_file=str(f), n=4, seed=1, repo_url=str(repo), ref="main")
    cases = list(src.cases())
    assert len(cases) + len(src.failures) == 2
    case = next(c for c in cases if c.metadata["truth_issue"] == 169)
    assert case.culprit == shas[6] and case.direction == "slower" and case.ground_truth == "detected"
    window = run_git(repo, "rev-list", "--first-parent", "--reverse", f"{case.good}..{case.bad}").stdout.split()
    assert len(window) == 4 and window.index(shas[6]) + 1 == case.metadata["truth_position"]
    assert "class Arithmetic" in case.benchmark.workload and case.benchmark.name == "sparse.Arithmetic.time_divide"
    hidden = case.for_localizer().to_json().replace(case.bad, "").replace(case.good, "")
    assert shas[6] not in hidden and "69131" not in hidden and "truth_" not in hidden
    assert any(f["reason"] == "short_history" for f in src.failures)          # culprit too close to the start
