"""SWE-fficiency source adapter (ported from APRCL's dataset/ + injector).

SWE-fficiency (HF `swefficiency/swefficiency`, 498 tasks, 9 Python libs)
contains performance OPTIMIZATIONS, not regressions. A regression case is
made from a task by injection:

  1. find the mainline commit O where the expert optimization (PR #n) landed;
  2. take the next N real mainline commits c1..cN;
  3. replay them on a new branch starting at O, inserting a commit R that
     reverse-applies the optimization (non-test .py files) before c_k;
  4. ground truth: good = O, culprit = R, bad = branch tip  (kind "injected").

Default "piggyback": instead of adding a commit R, the revert is folded
into the real upstream commit c_k, which becomes the culprit. Every
candidate then has real metadata. (APRCL's "neutral" extra commit is kept
for comparison; it was found to leak on sympy, where every other main-line
commit is a "Merge pull request #..." - see find_tells.) Replays that
conflict are reported as failures, never hand-fixed.

Injection writes a branch `perfhound/<instance>` - ONLY into clones that
Perfhound manages (RepoFetcher), never into a user's own repository.
Built cases are stored next to the clone and reused.

Scope: pure-Python repos by default (sympy, dask, xarray) - reverting a
patch in numpy / pandas / scipy C code would need a rebuild per commit.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

from ..cases import BenchmarkSpec, RegressionCase
from ..errors import GatewayError
from ..fetcher import RepoFetcher, is_remote
from ..gitcmd import ensure_objects, run_git, run_git_bytes
from ..local_git import prefetch_blobs
from ..worktree import WorktreeManager
from . import register_source

HF_DATASET = "swefficiency/swefficiency"
PURE_PYTHON_REPOS = ("sympy/sympy", "dask/dask", "pydata/xarray")
# piggyback - fold the revert INTO real upstream commit c_k (default). The culprit keeps
#             a real message, author, date and timezone: nothing to stand out.
# neutral   - extra commit "Refactor <module>" (APRCL's method). LEAKS on repos whose
#             main line is all "Merge pull request #..." (sympy) or "... (#123)" (xarray).
# revert    - extra commit 'Revert "<PR title>"': deliberately easy, for ablations.
MESSAGE_STRATEGIES = ("piggyback", "neutral", "revert")
TEST_PATH = re.compile(r"(^|/)(tests?|testing|benchmarks?|asv_bench)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")
# The SWE-fficiency instance id ("dask__dask-10356") contains the culprit PR's number,
# and merge/squash messages contain "#10356" - so it must never be visible to the localizer.
NEUTRAL_BENCHMARK_NAME = "workload"
BRANCH_PREFIX = "perfhound/"


@dataclass
class SweTask:
    instance_id: str                 # "sympy__sympy-25591"
    repo: str                        # "sympy/sympy"
    pr_number: int
    workload: str                    # Python script: setup() / workload() + timeit
    workload_origin: str = "official"
    base_commit: str | None = None
    patch: str | None = None         # expert optimization diff, if the source has it
    expert_speedup: float | None = None


@dataclass
class InjectionFailure:
    """Why a case could not be built (used by both the injected and the real-change builders)."""

    instance_id: str
    reason: str                      # short_history | empty_patch | revert_conflict | replay_conflict | pr_not_found
    detail: str = ""


# ---------------------------------------------------------------- loading tasks

def _pr_from_id(instance_id: str) -> int:
    return int(instance_id.rsplit("-", 1)[1])


def load_task_file(path: str | Path) -> list[SweTask]:
    """APRCL's frozen task file (data/tasks/*.json) or a list of SweTask dicts."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SweTask(**{k: v for k, v in r.items() if k in SweTask.__dataclass_fields__}) for r in rows]


def save_task_file(tasks: list[SweTask], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(t) for t in tasks], indent=2, ensure_ascii=False), encoding="utf-8")


def load_hf(repos: tuple[str, ...] | None = PURE_PYTHON_REPOS) -> list[SweTask]:
    """Official tasks from Hugging Face (needs `pip install datasets` and network)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise GatewayError("loading SWE-fficiency from Hugging Face needs: pip install datasets") from None
    tasks = []
    for row in load_dataset(HF_DATASET, split="test"):
        if repos and row["repo"] not in repos:
            continue
        tasks.append(SweTask(
            instance_id=row["instance_id"], repo=row["repo"], pr_number=_pr_from_id(row["instance_id"]),
            workload=row["workload"], workload_origin="official",
            base_commit=row.get("base_commit"), patch=row.get("patch"),
        ))
    return tasks


# ---------------------------------------------------------------- git helpers

def _git(repo, *args, env=None, input=None) -> str:
    return run_git(repo, *args, env=env, input=input).stdout.strip()


def default_branch(repo: Path) -> str:
    proc = run_git(repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD", check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    for cand in ("origin/main", "origin/master", "main", "master"):
        if run_git(repo, "rev-parse", "--verify", "--quiet", cand, check=False).returncode == 0:
            return cand
    raise GatewayError(f"cannot find the default branch of {repo}")


def find_pr_commit(repo: Path, pr_number: int, ref: str) -> str:
    """Mainline commit that landed PR #n: merge style (sympy) or squash style '(#n)' (xarray, dask)."""
    for pat in (f"^Merge pull request #{pr_number} ", rf"\(#{pr_number}\)$"):
        shas = _git(repo, "log", "--first-parent", "--format=%H", "-E", f"--grep={pat}", ref).split()
        if shas:
            return shas[-1]   # oldest match on the main line
    raise GatewayError(f"PR #{pr_number} not found on the first-parent history of {ref}")


def first_parent_after(repo: Path, commit: str, n: int, ref: str) -> list[str]:
    return _git(repo, "rev-list", "--first-parent", "--reverse", f"{commit}..{ref}").split()[:n]


def source_only_diff(repo: Path, base: str, head: str) -> bytes:
    """Expert patch as BYTES, so files in any encoding / with CRLF round-trip exactly."""
    files = [f for f in _git(repo, "diff", "--name-only", base, head).splitlines()
             if f.endswith(".py") and not TEST_PATH.search(f) and not f.startswith("doc")]
    if not files:
        return b""
    return run_git_bytes(repo, "diff", "--binary", base, head, "--", *files)


def _injected_message(strategy: str, repo: Path, opt: str, patch: bytes) -> str:
    if strategy == "revert":   # deliberately easy variant (an ablation): the message gives it away
        subject = _git(repo, "show", "-s", "--format=%s", opt)
        return f'Revert "{subject}"\n\nThis reverts commit {opt}.'
    if strategy == "neutral":
        changed: dict[str, int] = {}
        current = None
        for ln in patch.decode("utf-8", "replace").split("\n"):
            if ln.startswith("+++ b/"):
                current = ln[6:]
                changed[current] = 0
            elif current and ln[:1] in "+-" and not ln.startswith(("+++", "---")):
                changed[current] += 1
        ranked = sorted((f for f in changed if not f.startswith("bin/")), key=changed.get, reverse=True) or ["code"]
        return f"Refactor {ranked[0].removesuffix('.py').replace('/', '.')}"
    raise ValueError(f"unknown message strategy {strategy!r}; choose from {MESSAGE_STRATEGIES}")


# ---------------------------------------------------------------- injection

def _between(repo: Path, prev_ts: int, next_sha: str | None) -> int:
    """Author timestamp for the injected commit: inside (prev, next), else prev + 60 s."""
    if next_sha is None:
        return prev_ts + 60
    next_ts = int(_git(repo, "show", "-s", "--format=%at", next_sha))
    if next_ts - prev_ts >= 2:
        return prev_ts + min(60, (next_ts - prev_ts) // 2)
    return prev_ts          # neighbours share a second: same timestamp is the least visible choice


def instance_id_for(task_id: str, n_after: int, k: int, strategy: str) -> str:
    return f"{task_id}__n{n_after}_k{k:02d}" + ("_pb" if strategy == "piggyback" else "")


def build_injected_case(
    repo: Path, task: SweTask, n_after: int, position_k: int, *,
    message_strategy: str = "piggyback", ref: str | None = None,
) -> RegressionCase | InjectionFailure:
    """Build one injected range in `repo` (a Perfhound-managed clone).

    position_k is the culprit's 1-based index among the candidates:
    [1, n_after] for piggyback, [1, n_after + 1] when an extra commit is added.
    """
    if message_strategy not in MESSAGE_STRATEGIES:
        raise ValueError(f"unknown message strategy {message_strategy!r}; choose from {MESSAGE_STRATEGIES}")
    piggyback = message_strategy == "piggyback"
    if not 1 <= position_k <= n_after + (0 if piggyback else 1):
        raise ValueError("position_k out of range")
    instance_id = instance_id_for(task.instance_id, n_after, position_k, message_strategy)
    ref = ref or default_branch(repo)
    try:
        opt = find_pr_commit(repo, task.pr_number, ref)
    except GatewayError as exc:
        return InjectionFailure(instance_id, "pr_not_found", str(exc))
    mainline = first_parent_after(repo, opt, n_after, ref)
    if len(mainline) < n_after:
        return InjectionFailure(instance_id, "short_history", f"only {len(mainline)} commits after {opt[:10]}")

    # partial clone: download what checkout + cherry-picks need in a few batches
    tree_blobs = [l.split()[2] for l in _git(repo, "ls-tree", "-r", opt).splitlines() if l.split()[1] == "blob"]
    ensure_objects(repo, tree_blobs)
    prefetch_blobs(repo, [opt, *mainline])

    patch: bytes = (task.patch.encode("utf-8") if task.patch
                    else source_only_diff(repo, _git(repo, "rev-parse", f"{opt}^1"), opt))
    if not patch.strip():
        return InjectionFailure(instance_id, "empty_patch", "optimization touches no non-test .py files")
    if not patch.endswith(b"\n"):
        patch += b"\n"

    upstream_of: dict[str, str] = {}
    culprit = ""
    with WorktreeManager(repo) as wtm:
        wt = wtm.checkout(opt)
        def revert_optimization() -> str | None:
            """Reverse-apply the expert patch to the index; returns an error text on conflict."""
            patch_file = wt / ".perfhound_opt.patch"
            patch_file.write_bytes(patch)
            try:
                applied = run_git(wt, "apply", "-R", "--index", str(patch_file), check=False)
                if applied.returncode != 0:
                    applied = run_git(wt, "apply", "-R", "--3way", "--index", str(patch_file), check=False)
            finally:
                patch_file.unlink()
            return None if applied.returncode == 0 else (applied.stderr.strip()[:500] or "apply failed")

        sequence: list[str] = list(mainline)
        if not piggyback:
            sequence.insert(position_k - 1, "INJECT")
        prev = opt
        inject_at = position_k - 1
        next_real = sequence[inject_at + 1] if not piggyback and inject_at + 1 < len(sequence) else None
        for index, item in enumerate(sequence):
            if item == "INJECT":
                error = revert_optimization()
                if error:
                    return InjectionFailure(instance_id, "revert_conflict", error)
                # borrow identity from the previous real commit, and a date BETWEEN the previous
                # and next real commits (a fixed "+60 s" can land after the next commit: a tell)
                name, email, ts, iso = _git(repo, "show", "-s", "--format=%an%x1f%ae%x1f%at%x1f%ai", prev).split("\x1f")
                offset = iso.split()[-1]   # e.g. "-0500": keep the neighbour's timezone too ("+0000" was a tell)
                date = f"@{_between(repo, int(ts), next_real)} {offset}"
                env = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_AUTHOR_DATE": date,
                       "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email, "GIT_COMMITTER_DATE": date}
                run_git(wt, "commit", "-q", "--no-verify", "-m",
                        _injected_message(message_strategy, repo, opt, patch), env=env)
                culprit = _git(wt, "rev-parse", "HEAD")
                upstream_of[culprit] = "INJECTED"
                continue
            cname, cemail, cdate = _git(repo, "show", "-s", "--format=%cn%x1f%ce%x1f%cI", item).split("\x1f")
            env = {"GIT_COMMITTER_NAME": cname, "GIT_COMMITTER_EMAIL": cemail, "GIT_COMMITTER_DATE": cdate}
            args = ["cherry-pick", "--allow-empty", "--keep-redundant-commits"]
            if len(_git(repo, "rev-list", "--parents", "-n", "1", item).split()) > 2:
                args += ["-m", "1"]   # merges are flattened onto the first parent
            picked = run_git(wt, *args, item, env=env, check=False)
            if picked.returncode != 0:
                run_git(wt, "cherry-pick", "--abort", check=False)
                return InjectionFailure(instance_id, "replay_conflict", f"{item[:10]}: {picked.stderr.strip()[:300]}")
            if piggyback and index == inject_at:
                error = revert_optimization()
                if error:
                    return InjectionFailure(instance_id, "revert_conflict", error)
                # amend keeps the real author, author date and message; committer as upstream
                run_git(wt, "commit", "-q", "--amend", "--no-edit", "--no-verify", "--allow-empty", env=env)
                culprit = _git(wt, "rev-parse", "HEAD")
                upstream_of[culprit] = f"PIGGYBACK:{item}"
            else:
                upstream_of[_git(wt, "rev-parse", "HEAD")] = item
            prev = item
        candidates = _git(repo, "rev-list", "--reverse", f"{opt}..{_git(wt, 'rev-parse', 'HEAD')}").split()
        run_git(repo, "branch", "-f", BRANCH_PREFIX + instance_id, candidates[-1])

    magnitude = (task.expert_speedup - 1.0) if task.expert_speedup else None   # 1.84x speedup -> 0.84 slower
    return RegressionCase(
        case_id=f"swefficiency:{instance_id}", source="swefficiency", repo=str(repo), language="python",
        good=opt, bad=candidates[-1], culprit=culprit, ground_truth="injected",
        regression_type="revert_expert_optimization", expected_magnitude=magnitude,
        benchmark=BenchmarkSpec(name=NEUTRAL_BENCHMARK_NAME, framework="script", workload=task.workload),
        metadata={
            "truth_task": task.instance_id, "upstream_repo": task.repo, "truth_pr_number": task.pr_number,
            "workload_origin": task.workload_origin, "message_strategy": message_strategy,
            "truth_n_after": n_after, "truth_branch": BRANCH_PREFIX + instance_id,
            "truth_expert_speedup": task.expert_speedup,
            "truth_position": candidates.index(culprit) + 1, "truth_upstream_of": upstream_of,
        },
    )


# ---------------------------------------------------------------- adapter

def _store_path(repo: Path, instance_id: str) -> Path:
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir"))
    return git_dir / "perfhound" / "cases" / f"{instance_id}.json"


class SwefficiencyAdapter:
    """Injected regression cases from SWE-fficiency tasks.

        open_source("swefficiency", tasks_file="data/tasks/sympy_demo_tasks.json",
                    n_after=20, positions_per_task=2, seed=7)
    """

    name = "swefficiency"

    def __init__(
        self, *, tasks_file: str | Path | None = None, hf: bool = False,
        repos: tuple[str, ...] | None = PURE_PYTHON_REPOS, ids: list[str] | None = None,
        n_after: int = 20, positions: list[int] | None = None, positions_per_task: int = 1, seed: int = 0,
        message_strategy: str = "piggyback", fetcher: RepoFetcher | None = None,
        repo_url: Callable[[str], str] = lambda r: f"https://github.com/{r}",
        ref: str | None = None,
    ) -> None:
        if tasks_file is None and not hf:
            raise ValueError("give tasks_file=... or hf=True")
        if message_strategy not in MESSAGE_STRATEGIES:
            raise ValueError(f"message_strategy must be one of {MESSAGE_STRATEGIES}")
        self.tasks_file, self.hf, self.repos, self.ids = tasks_file, hf, repos, ids
        self.n_after, self.positions, self.per_task, self.seed = n_after, positions, positions_per_task, seed
        self.message_strategy = message_strategy
        self.fetcher = fetcher or RepoFetcher()
        self.repo_url = repo_url
        self.ref = ref
        self.failures: list[InjectionFailure] = []

    def tasks(self) -> list[SweTask]:
        tasks = load_hf(self.repos) if self.hf else load_task_file(self.tasks_file)
        if self.repos:
            tasks = [t for t in tasks if t.repo in self.repos]
        if self.ids:
            tasks = [t for t in tasks if t.instance_id in set(self.ids)]
        return sorted(tasks, key=lambda t: t.instance_id)

    def cases(self, limit: int | None = None) -> Iterator[RegressionCase]:
        rng = random.Random(self.seed)
        produced = 0
        for task in self.tasks():
            url = self.repo_url(task.repo)
            if not is_remote(url):
                raise GatewayError("injection writes branches; it only runs on clones made by RepoFetcher (give a URL)")
            repo = self.fetcher.fetch(url)
            top = self.n_after + (0 if self.message_strategy == "piggyback" else 1)
            ks = self.positions or rng.sample(range(1, top + 1), self.per_task)
            for k in ks:
                if limit is not None and produced >= limit:
                    return
                instance_id = instance_id_for(task.instance_id, self.n_after, k, self.message_strategy)
                stored = _store_path(repo, instance_id)
                case = None
                if stored.exists():
                    case = RegressionCase.from_json(stored.read_text(encoding="utf-8"))
                    tip = run_git(repo, "rev-parse", "--verify", "--quiet", case.metadata.get("truth_branch", ""), check=False)
                    if tip.stdout.strip() != case.bad or case.metadata.get("message_strategy") != self.message_strategy:
                        case = None   # branch moved / different settings: rebuild
                if case is None:
                    built = build_injected_case(repo, task, self.n_after, k,
                                                message_strategy=self.message_strategy, ref=self.ref)
                    if isinstance(built, InjectionFailure):
                        self.failures.append(built)
                        continue
                    case = built
                    stored.parent.mkdir(parents=True, exist_ok=True)
                    stored.write_text(case.to_json(), encoding="utf-8")
                produced += 1
                yield case


class AprclInstancesAdapter:
    """Read instances already built by APRCL (data/instances/*.json) - no git writes."""

    name = "aprcl-instances"

    def __init__(self, instances_dir: str | Path, aprcl_root: str | Path | None = None,
                 workloads_root: str | Path | None = None) -> None:
        self.dir = Path(instances_dir)
        self.root = Path(aprcl_root) if aprcl_root else self.dir.parent.parent
        self.workloads_root = Path(workloads_root) if workloads_root else self.root

    def cases(self, limit: int | None = None) -> Iterator[RegressionCase]:
        for i, path in enumerate(sorted(self.dir.glob("*.json"))):
            if limit is not None and i >= limit:
                return
            d = json.loads(path.read_text(encoding="utf-8"))
            repo = Path(d["repo_path"])
            repo = repo if repo.is_absolute() else self.root / repo
            wl = self.workloads_root / d["workload_file"] if d.get("workload_file") else None
            speed = d.get("expected_magnitude")
            yield RegressionCase(
                case_id=f"swefficiency:{d['instance_id']}", source="aprcl-instances", repo=str(repo),
                language="python", good=d["good_commit"], bad=d["bad_commit"], culprit=d["culprit_commit"],
                ground_truth="injected", regression_type=d.get("regression_type", "revert_expert_optimization"),
                expected_magnitude=(speed - 1.0) if speed else None,
                benchmark=BenchmarkSpec(name=NEUTRAL_BENCHMARK_NAME, framework="script",
                                        workload=wl.read_text(encoding="utf-8") if wl and wl.exists() else None,
                                        command=str(wl) if wl else None),
                metadata={
                    "truth_task": d["source_instance"], "upstream_repo": d["repo"], "truth_branch": d.get("branch"),
                    "message_strategy": d.get("message_strategy"), "workload_origin": d.get("workload_origin"),
                    "truth_expert_speedup": speed, "truth_position": d.get("culprit_position"),
                    "truth_upstream_of": d.get("upstream_of", {}),
                },
            )


# ---------------------------------------------------------------- real changes (no injection)

def build_real_change_case(
    repo: Path, task: SweTask, n: int, position_k: int, *, ref: str | None = None, mainline: list[str] | None = None,
    repo_ref: str | None = None,
) -> RegressionCase | InjectionFailure:
    """A window of n real main-line commits that contains the expert optimization at position k.

    Nothing is written to the repository. good = the commit k positions before the
    optimization O, candidates = the next n main-line commits (O is the k-th), and the
    change to find is a SPEEDUP (direction "faster"). Ground truth kind is "reported":
    O is the dataset's verified optimization, but other commits in the window may also
    move this workload - Step 4 (benchmark validation) checks every case.

    repo_ref is what the case records as its repo (normally the GitHub URL, so a cases
    file works on any machine); defaults to the local path.
    """
    if not 1 <= position_k <= n:
        raise ValueError("position_k must be in [1, n]")
    instance_id = f"{task.instance_id}__real_n{n}_k{position_k:02d}"
    if not (task.workload or "").strip():
        return InjectionFailure(instance_id, "no_workload", "task has no workload script - nothing to measure")
    ref = ref or default_branch(repo)
    try:
        opt = find_pr_commit(repo, task.pr_number, ref)
    except GatewayError as exc:
        return InjectionFailure(instance_id, "pr_not_found", str(exc))
    line = mainline if mainline is not None else _git(repo, "rev-list", "--first-parent", "--reverse", ref).split()
    idx = line.index(opt)
    start = idx - position_k                      # index of good
    if start < 0 or start + n >= len(line):
        return InjectionFailure(instance_id, "short_history",
                                f"need {position_k} commits before and {n - position_k} after {opt[:10]}")
    good, candidates = line[start], line[start + 1:start + n + 1]
    speedup = task.expert_speedup
    return RegressionCase(
        case_id=f"swefficiency-real:{instance_id}", source="swefficiency-real", repo=repo_ref or str(repo),
        language="python",
        good=good, bad=candidates[-1], culprit=opt, ground_truth="reported",
        regression_type="expert_optimization", direction="faster",
        expected_magnitude=(1.0 - 1.0 / speedup) if speedup else None,   # 1.84x faster -> time -45.7 %
        benchmark=BenchmarkSpec(name=NEUTRAL_BENCHMARK_NAME, framework="script", workload=task.workload),
        metadata={
            "truth_task": task.instance_id, "upstream_repo": task.repo, "truth_pr_number": task.pr_number, "n": n,
            "workload_origin": task.workload_origin, "truth_expert_speedup": speedup, "truth_position": position_k,
        },
    )


class SwefficiencyRealAdapter:
    """Real performance-change cases from SWE-fficiency - no injection, no git writes.

        open_source("swefficiency-real", tasks_file="data/tasks/swefficiency_pure_python.json",
                    n=20, positions_per_task=1, seed=0)
    """

    name = "swefficiency-real"

    def __init__(
        self, *, tasks_file: str | Path | None = None, hf: bool = False,
        repos: tuple[str, ...] | None = PURE_PYTHON_REPOS, ids: list[str] | None = None,
        n: int = 20, positions: list[int] | None = None, positions_per_task: int = 1, seed: int = 0,
        fetcher: RepoFetcher | None = None,
        repo_url: Callable[[str], str] = lambda r: f"https://github.com/{r}", ref: str | None = None,
    ) -> None:
        if tasks_file is None and not hf:
            raise ValueError("give tasks_file=... or hf=True")
        self.tasks_file, self.hf, self.repos, self.ids = tasks_file, hf, repos, ids
        self.n, self.positions, self.per_task, self.seed = n, positions, positions_per_task, seed
        self.fetcher = fetcher or RepoFetcher()
        self.repo_url, self.ref = repo_url, ref
        self.failures: list[InjectionFailure] = []

    def tasks(self) -> list[SweTask]:
        tasks = load_hf(self.repos) if self.hf else load_task_file(self.tasks_file)
        if self.repos:
            tasks = [t for t in tasks if t.repo in self.repos]
        if self.ids:
            tasks = [t for t in tasks if t.instance_id in set(self.ids)]
        return sorted(tasks, key=lambda t: t.instance_id)

    def cases(self, limit: int | None = None) -> Iterator[RegressionCase]:
        rng = random.Random(self.seed)          # same seed -> same positions -> same cases
        produced = 0
        mainlines: dict[Path, tuple[str, list[str]]] = {}
        for task in self.tasks():
            url = self.repo_url(task.repo)
            repo = self.fetcher.fetch(url)
            if repo not in mainlines:
                ref = self.ref or default_branch(repo)
                mainlines[repo] = (ref, _git(repo, "rev-list", "--first-parent", "--reverse", ref).split())
            ref, line = mainlines[repo]
            ks = self.positions or rng.sample(range(1, self.n + 1), self.per_task)
            for k in ks:
                if limit is not None and produced >= limit:
                    return
                built = build_real_change_case(repo, task, self.n, k, ref=ref, mainline=line, repo_ref=url)
                if isinstance(built, InjectionFailure):
                    self.failures.append(built)
                    continue
                produced += 1
                yield built


register_source("swefficiency", SwefficiencyAdapter)
register_source("swefficiency-real", SwefficiencyRealAdapter)
register_source("aprcl-instances", AprclInstancesAdapter)
