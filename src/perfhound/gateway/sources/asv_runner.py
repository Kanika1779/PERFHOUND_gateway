"""pandas asv-runner: REAL performance regressions caught by CI, as Perfhound cases.

pandas-dev/asv-runner runs pandas' ASV benchmark suite on EVERY commit to main. When a
benchmark gets slower it opens an issue titled with that commit:

    Commit 477bee05b1b677eca97209efe8e7d6881f9e64aa
    [PR #69131](https://github.com/pandas-dev/pandas/pull/69131)
     - [ ] [sparse.Arithmetic.time_divide](...)
       - [ ] [dense_proportion=0.1, fill_value=nan](...) - 65.025% (1.275ms)

What makes these cases valuable (unlike SWE-fficiency):
  * a REGRESSION, not a deliberate optimization;
  * the benchmark was written independently of the culprit (it is pandas' own suite);
  * the culprit was found by CI, not by the culprit's author.
Ground truth = the flagged commit, labelled "detected": an automatic CI detector flagged it,
nobody confirmed it (see GROUND_TRUTH_KINDS in cases.py). Caveats (documented, not hidden):
a flag can be benchmark noise, and if a run was skipped the true culprit could be an earlier
commit since the previous run. Re-measure good / culprit's parent / culprit before trusting one.

Case = a window of n real main-line commits with the flagged commit at a random position
(same method as swefficiency-real). The symptom shown to the localizer is the benchmark name,
the regressed parameters and the benchmark's SOURCE CODE read from the repo at `bad`.

Issues are snapshotted to a JSON file first (scripts/fetch_asv_regressions.py) so a cases file
is reproducible without GitHub.
"""

from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from ..cases import BenchmarkSpec, RegressionCase
from ..errors import GatewayError
from ..fetcher import RepoFetcher
from ..gitcmd import run_git
from . import register_source

PANDAS = "https://github.com/pandas-dev/pandas"
_TITLE = re.compile(r"^Commit ([0-9a-f]{40})\s*$")
_PR = re.compile(r"\[PR #(\d+)\]")
_BENCH = re.compile(r"^\s*- \[[ x]\] \[([A-Za-z_][\w.]*)\]\([^)]*\)(?:\s*-\s*([\d.]+)%\s*\(([^)]*)\))?\s*$")
_PARAM = re.compile(r"^\s+- \[[ x]\] \[(.*)\]\([^)]*\)\s*-\s*([\d.]+)%\s*\(([^)]*)\)\s*$")


@dataclass
class AsvRegression:
    issue: int
    commit: str
    pr: int | None
    created_at: str
    url: str
    benchmarks: list[dict] = field(default_factory=list)   # {"name", "params", "pct", "abs"}

    @property
    def max_pct(self) -> float:
        return max((b["pct"] for b in self.benchmarks if b.get("pct") is not None), default=0.0)


def parse_issue(issue: dict) -> AsvRegression | None:
    """An asv-runner issue -> AsvRegression, or None for other issues (run failures, PRs ...)."""
    if "pull_request" in issue:                        # the issues API also lists PRs
        return None
    m = _TITLE.match(issue.get("title") or "")
    if not m:
        return None
    body = issue.get("body") or ""
    pr = _PR.search(body)
    benches: list[dict] = []
    current = None
    for line in body.splitlines():
        b = _BENCH.match(line)
        if b and not line.startswith("   "):
            current = b.group(1)
            if b.group(2):                                  # benchmark without parameters
                benches.append({"name": current, "params": "", "pct": float(b.group(2)), "abs": b.group(3)})
            continue
        p = _PARAM.match(line)
        if p and current:
            benches.append({"name": current, "params": p.group(1), "pct": float(p.group(2)), "abs": p.group(3)})
    if not benches:
        return None
    return AsvRegression(issue=issue["number"], commit=m.group(1), pr=int(pr.group(1)) if pr else None,
                         created_at=issue.get("created_at", ""), url=issue.get("html_url", ""), benchmarks=benches)


def save_regressions(regs: list[AsvRegression], path: str | Path, *, source: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"source": source, "regressions": [asdict(r) for r in regs]}, indent=1),
                          encoding="utf-8")


def load_regressions(path: str | Path) -> list[AsvRegression]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [AsvRegression(**r) for r in doc["regressions"]]


def benchmark_source(repo: Path, rev: str, name: str, root: str = "asv_bench/benchmarks") -> str | None:
    """Source of the benchmark CLASS (or function) `module.Class.method` at `rev`, e.g.
    sparse.Arithmetic.time_divide -> class Arithmetic in asv_bench/benchmarks/sparse.py."""
    parts = name.split(".")
    for cut in range(len(parts) - 1, 0, -1):               # longest module path that exists
        path = f"{root}/{'/'.join(parts[:cut])}.py"
        proc = run_git(repo, "show", f"{rev}:{path}", check=False)
        if proc.returncode != 0:
            continue
        text, rest = proc.stdout, parts[cut:]
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text[:6000]
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == rest[0]:
                return "\n".join(text.splitlines()[node.lineno - 1 - len(node.decorator_list):node.end_lineno])
        return text[:6000]
    return None


def build_asv_case(repo: Path, reg: AsvRegression, n: int, position_k: int, mainline: list[str],
                   repo_url: str = PANDAS) -> RegressionCase | dict:
    """Window of n main-line commits with the flagged commit at position k; dict = failure reason."""
    case_name = f"pandas-asv-{reg.issue}__n{n}_k{position_k:02d}"
    if reg.commit not in mainline:
        return {"case": case_name, "reason": "not_on_mainline", "detail": reg.commit}
    idx = mainline.index(reg.commit)
    start = idx - position_k
    if start < 0 or start + n >= len(mainline):
        return {"case": case_name, "reason": "short_history",
                "detail": f"need {position_k} commits before and {n - position_k} after"}
    good, cands = mainline[start], mainline[start + 1:start + n + 1]
    main = max(reg.benchmarks, key=lambda b: b.get("pct") or 0)
    src = benchmark_source(repo, cands[-1], main["name"])
    regressed = "; ".join(sorted({f"{b['name']}({b['params']})" if b["params"] else b["name"] for b in reg.benchmarks}))
    return RegressionCase(
        case_id=f"asv-runner:{case_name}", source="asv-runner", repo=repo_url, language="python",
        good=good, bad=cands[-1], culprit=reg.commit, ground_truth="detected", regression_type="ci_detected",
        direction="slower", expected_magnitude=main["pct"] / 100 if main.get("pct") else None,
        benchmark=BenchmarkSpec(name=main["name"], framework="asv", workload=src,
                                params={"asv_params": main["params"], "regressed": regressed}),
        metadata={"symptom": f"ASV benchmark regressions: {regressed}", "n": n, "truth_position": position_k,
                  "truth_issue": reg.issue, "truth_pr_number": reg.pr, "truth_issue_url": reg.url,
                  "upstream_repo": "pandas-dev/pandas"},
    )


class AsvRunnerAdapter:
    """open_source("asv-runner", regressions_file="data/asv_pandas_regressions.json", n=20, seed=0)"""

    name = "asv-runner"

    def __init__(self, *, regressions_file: str | Path, n: int = 20, positions_per_case: int = 1, seed: int = 0,
                 min_pct: float = 0.0, fetcher: RepoFetcher | None = None, repo_url: str = PANDAS,
                 ref: str | None = None) -> None:
        self.file, self.n, self.per, self.seed, self.min_pct = regressions_file, n, positions_per_case, seed, min_pct
        self.fetcher = fetcher or RepoFetcher()
        self.repo_url, self.ref = repo_url, ref
        self.failures: list[dict] = []

    def cases(self, limit: int | None = None) -> Iterator[RegressionCase]:
        from .swefficiency import default_branch

        regs = sorted((r for r in load_regressions(self.file) if r.max_pct >= self.min_pct), key=lambda r: r.issue)
        repo = self.fetcher.fetch(self.repo_url)
        missing = self.fetcher.ensure_commits(repo, [r.commit for r in regs])
        ref = self.ref or default_branch(repo)
        line = run_git(repo, "rev-list", "--first-parent", "--reverse", ref).stdout.split()
        rng = random.Random(self.seed)
        produced = 0
        for reg in regs:
            for k in rng.sample(range(1, self.n + 1), self.per):
                if limit is not None and produced >= limit:
                    return
                if reg.commit in missing:
                    self.failures.append({"case": f"pandas-asv-{reg.issue}", "reason": "commit_missing",
                                          "detail": reg.commit})
                    continue
                built = build_asv_case(repo, reg, self.n, k, line, self.repo_url)
                if isinstance(built, dict):
                    self.failures.append(built)
                    continue
                produced += 1
                yield built


register_source("asv-runner", AsvRunnerAdapter)
