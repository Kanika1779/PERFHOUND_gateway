"""Gateway Facade (gateway part 1): the ONLY door other modules use.

    from perfhound.gateway import Gateway
    with Gateway("path/to/repo") as gw:
        candidates = gw.get_candidates("v1.2", "HEAD")
        with gw.worktree() as wt:
            folder = wt.checkout(candidates[0].sha)

Parts behind this door (range resolver, local git, analyzer, cache, later
GitHub and snapshots) can change freely as long as this API stays the same.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from dataclasses import dataclass, replace
from pathlib import Path

from .analyzer import ANALYZER_VERSION, CodeAnalyzer, FunctionChanges
from .cache import Cache
from .cases import RegressionCase
from .errors import GatewayError
from .fetcher import RepoFetcher
from .gitcmd import run_git
from .local_git import DEFAULT_MAX_DIFF_CHARS, DEFAULT_MAX_DIFF_LINES, LocalGitProvider, parse_git_date
from .models import SCHEMA_VERSION, CandidateCommit
from .range import CommitRange, find_repo_root, resolve_range
from .snapshot import Snapshot, build_snapshot
from .worktree import WorktreeManager


@dataclass(frozen=True)
class RequestStats:
    """What the last get_candidates() call cost (for the evaluation section)."""

    commits: int
    commit_cache_hits: int
    function_cache_hits: int
    seconds: float
    github_requests: int = 0          # 0 when GitHub is off
    github_with_pr: int = 0
    github_warnings: tuple[str, ...] = ()


def repo_identity(repo: Path) -> str:
    """Stable id for a repository: its root commit(s), not its folder.

    Survives moving / re-cloning the repo. Falls back to the folder path
    for a repo without commits.
    """
    proc = run_git(repo, "rev-list", "--max-parents=0", "HEAD", check=False)
    roots = sorted(proc.stdout.split()) if proc.returncode == 0 else []
    basis = ",".join(roots) if roots else "path:" + str(repo.resolve())
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


class Gateway:
    def __init__(
        self,
        repo: str | Path,
        *,
        cache: Cache | str | Path | bool = True,
        max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
        max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
        github=None,
        github_repo=None,
        time_rule: bool = True,
        batch_size: int = 500,
    ) -> None:
        """github: a GitHubPRProvider / GitHubGraphQLProvider (optional) - attaches PR title,
            description and linked issues to each commit.
        github_repo: RepoRef; default = parsed from the `origin` remote.
        time_rule: True (evaluation) = no GitHub data that appeared after `bad` landed;
            False (real use) = everything, e.g. a later "slow since #123" issue.
        batch_size: commits read / analyzed per step, so big ranges never load all at once."""
        self.repo = find_repo_root(repo)
        self.github = github
        self._github_repo = github_repo
        self.time_rule = time_rule
        self.batch_size = max(1, batch_size)
        self.max_diff_lines = max_diff_lines
        self.max_diff_chars = max_diff_chars
        self._local = LocalGitProvider(self.repo, max_diff_lines=max_diff_lines, max_diff_chars=max_diff_chars)
        self._analyzer = CodeAnalyzer(self.repo)
        if cache is True:
            self.cache: Cache | None = Cache()
        elif cache is False or cache is None:
            self.cache = None
        elif isinstance(cache, Cache):
            self.cache = cache
        else:
            self.cache = Cache(cache)
        self._repo_id: str | None = None
        self.last_stats: RequestStats | None = None

    @property
    def repo_id(self) -> str:
        if self._repo_id is None:
            self._repo_id = repo_identity(self.repo)
        return self._repo_id

    @classmethod
    def for_case(cls, case: RegressionCase, *, fetcher: RepoFetcher | None = None, **options) -> "Gateway":
        """Gateway on the case's repository (cloned on first use), with good/bad present locally."""
        fetcher = fetcher or RepoFetcher()
        path = fetcher.fetch(case.repo)
        missing = fetcher.ensure_commits(path, [case.good, case.bad])
        if missing:
            raise GatewayError(f"case {case.case_id}: commits not found in {case.repo}: {missing}")
        if options.get("github") is not None and options.get("github_repo") is None:
            from .github.links import is_github, parse_repo

            upstream = case.metadata.get("upstream_repo") or case.repo
            if is_github(upstream):
                options["github_repo"] = parse_repo(upstream)
        return cls(path, **options)

    def candidates_for(self, case: RegressionCase, **options) -> list[CandidateCommit]:
        """Candidates for a case. Never uses the case's ground truth."""
        return self.get_candidates(case.good, case.bad, **options)

    # -- main API --------------------------------------------------------------

    def resolve(self, good: str, bad: str, **range_options) -> CommitRange:
        """good..bad -> ordered SHAs (see range.resolve_range for options)."""
        return resolve_range(self.repo, good, bad, **range_options)

    def get_candidates(self, good: str, bad: str, *, analyze: bool = True, **range_options) -> list[CandidateCommit]:
        """All commits after `good` up to `bad` (first-parent), oldest first.

        Commits already in the cache are not read from git again; only the
        missing ones are (still in one batched call).
        """
        start = time.perf_counter()
        commit_range = self.resolve(good, bad, **range_options)
        commits: list[CandidateCommit] = []
        commit_hits = function_hits = 0
        for start in range(0, len(commit_range.shas), self.batch_size):     # big ranges: piece by piece
            part = replace(commit_range, shas=commit_range.shas[start:start + self.batch_size])
            batch, hits = self._load_commits(part)
            commit_hits += hits
            if analyze:
                batch, hits = self._add_functions(batch)
                function_hits += hits
            commits.extend(replace(c, position=c.position + start) for c in batch)
        gh_requests, gh_with_pr, gh_warnings = 0, 0, ()
        if self.github is not None and commits:
            commits, gh = self._add_github(commits)
            gh_requests, gh_with_pr, gh_warnings = gh.requests, gh.with_pr, tuple(gh.warnings)
        self.last_stats = RequestStats(len(commits), commit_hits, function_hits, time.perf_counter() - start,
                                       gh_requests, gh_with_pr, gh_warnings)
        return commits

    def github_repo(self):
        """RepoRef of this repository on GitHub (given, or from the `origin` remote), else None."""
        if self._github_repo is None:
            from .github.links import is_github, parse_repo

            proc = run_git(self.repo, "remote", "get-url", "origin", check=False)
            url = proc.stdout.strip() if proc.returncode == 0 else ""
            self._github_repo = parse_repo(url) if url and is_github(url) else False
        return self._github_repo or None

    def _add_github(self, commits: list[CandidateCommit]):
        from .github.provider import GitHubStats

        repo = self.github_repo()
        if repo is None:
            return commits, GitHubStats(warnings=["not a GitHub repository (origin remote) - PR data skipped"])
        cutoff = self.landed_at(commits[-1].sha) if self.time_rule else None
        return self.github.enrich(repo, self.repo_id, commits, cutoff=cutoff)

    def landed_at(self, sha: str) -> datetime:
        """When `sha` landed on its branch: its COMMITTER date (TIME RULE cut-off for GitHub data).

        Not the author date (CandidateCommit.timestamp): rebase merges and cherry-picks keep the
        original author date, so code can land days after it was authored.
        """
        iso, epoch = run_git(self.repo, "show", "-s", "--format=%cI%x00%ct", sha).stdout.strip().split("\x00")
        return parse_git_date(iso, epoch)

    def snapshot(self, path: str | Path, good: str, bad: str, *, sanitizer=None, **range_options) -> Snapshot:
        """Freeze good..bad into a verified JSON file for reproducible experiments."""
        from perfhound import __version__

        commit_range = self.resolve(good, bad, **range_options)
        candidates = self.get_candidates(commit_range.good, commit_range.bad, **range_options)
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "perfhound_version": __version__,
            "repo_id": self.repo_id,
            "good_ref": good, "bad_ref": bad,
            "good": commit_range.good, "bad": commit_range.bad,
            "settings": {"max_diff_lines": self.max_diff_lines, "max_diff_chars": self.max_diff_chars,
                         "analyzer_version": ANALYZER_VERSION},
        }
        snap = build_snapshot(candidates, meta, sanitizer)
        snap.save(path)
        return snap

    def worktree(self, **options) -> WorktreeManager:
        """A private checkout area for benchmarking; never touches the user's folder."""
        return WorktreeManager(self.repo, **options)

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()

    def __enter__(self) -> "Gateway":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals -----------------------------------------------------------

    @property
    def _commit_kind(self) -> str:
        return f"commit:s{SCHEMA_VERSION}:l{self.max_diff_lines}:c{self.max_diff_chars}"

    _FUNCTION_KIND = f"functions:a{ANALYZER_VERSION}"

    def _load_commits(self, commit_range: CommitRange) -> tuple[list[CandidateCommit], int]:
        by_sha: dict[str, CandidateCommit] = {}
        if self.cache is not None:
            for sha, payload in self.cache.get_many(self.repo_id, commit_range.shas, self._commit_kind).items():
                try:
                    by_sha[sha] = CandidateCommit.from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    pass   # damaged entry -> treat as a miss, it will be overwritten
        hits = len(by_sha)

        missing = [s for s in commit_range.shas if s not in by_sha]
        if missing:
            fresh = self._local.get_commits(commit_range, missing)
            by_sha.update((c.sha, c) for c in fresh)
            if self.cache is not None:
                self.cache.put_many(
                    self.repo_id,
                    {c.sha: replace(c, position=0).to_dict() for c in fresh},   # position is range-specific
                    self._commit_kind,
                )
        commits = [replace(by_sha[sha], position=i) for i, sha in enumerate(commit_range.shas)]
        return commits, hits

    def _add_functions(self, commits: list[CandidateCommit]) -> tuple[list[CandidateCommit], int]:
        changes: dict[str, FunctionChanges] = {}
        if self.cache is not None:
            for sha, p in self.cache.get_many(self.repo_id, [c.sha for c in commits], self._FUNCTION_KIND).items():
                try:
                    changes[sha] = FunctionChanges(
                        tuple(p["changed"]), tuple(p["added"]), tuple(p["deleted"]), tuple(p["skipped"])
                    )
                except (KeyError, TypeError):
                    pass
        hits = len(changes)

        need = [c for c in commits if c.sha not in changes]
        if need:
            fresh = self._analyzer.analyze(need)
            changes.update(fresh)
            if self.cache is not None:
                self.cache.put_many(
                    self.repo_id,
                    {
                        sha: {"changed": list(fc.changed), "added": list(fc.added),
                              "deleted": list(fc.deleted), "skipped": list(fc.skipped_files)}
                        for sha, fc in fresh.items()
                    },
                    self._FUNCTION_KIND,
                )
        out = [
            replace(
                c,
                changed_functions=changes[c.sha].changed,
                added_functions=changes[c.sha].added,
                deleted_functions=changes[c.sha].deleted,
                unanalyzed_files=changes[c.sha].skipped_files,
            )
            for c in commits
        ]
        return out, hits
