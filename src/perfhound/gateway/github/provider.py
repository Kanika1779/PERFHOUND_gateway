"""GitHub PR provider: the pull request behind each candidate commit -> CandidateCommit.pr.

Git stores commit messages; GitHub stores the PR title and description, which often explain
WHAT a change is for ("Speed up Matrix.atoms", "Validate inputs in ..."). RAG and the LLM read
pr.title + pr.body when present (rag/documents.py), so this adds signal for terse commits.

    GET /repos/{o}/{r}/commits/{sha}/pulls      one request per commit (cached, see below)
    choose: the PR whose merge commit IS this commit; else the earliest merged PR containing it
    GET /repos/{o}/{r}/issues/{n}               the issues that PR closes ("Fixes #45"), at most
                                                `max_issues_per_pr`, only with fields="title+body"

Linked issues are added to pr.body as text ("Issue #45: <title>\\n<body>"), so RAG and the LLM
read them without any change on their side.

TIME RULE (leak guard): nothing that appeared after the `bad` commit LANDED is used. The cut-off
is the committer date of `bad` (the gateway passes it), not an author date: with rebase merges
code lands long after it was authored.
  * PRs: a PR merged after the cut-off is never used - a later PR ("fix the slowdown from #123")
    would hand the localizer the answer. Exception: the PR whose merge commit IS the candidate
    commit. It landed that very commit, so it cannot be a later PR, and a maintainer who merged
    locally and pushed later would otherwise lose it.
  * Issues: an issue opened after the cut-off is never used ("X got slow since #123").
Remaining risk, documented, not hidden: GitHub returns the CURRENT title/body; an author may
have edited them after the merge. fields="title" (titles are rarely edited; no descriptions,
no issues) is the cautious setting for evaluations; the default "title+body" is for real use.

Cost & caching: answers are cached per commit / issue with their ETag (cache "volatile" store,
24 h). A stale entry is re-validated with If-None-Match: a 304 answer costs nothing against the
limit. Never blocks localization: no network / rate limit / bad token -> stop asking, keep what
is known, report a warning. Everything else works exactly as without GitHub.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Sequence

from ..models import CandidateCommit, PRInfo
from .client import GitHubClient
from .errors import GitHubError, GitHubNotFound
from .links import RepoRef, linked_issues

KIND = "github_pulls_v1"
ISSUE_KIND = "github_issue_v1"


@dataclass
class GitHubStats:
    requests: int = 0
    not_modified: int = 0
    cache_hits: int = 0
    with_pr: int = 0
    skipped_after_cutoff: int = 0
    issues_used: int = 0
    issues_skipped_after_cutoff: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Run:
    """State of one enrich() call: request budget and whether GitHub stopped answering."""

    repo: RepoRef
    repo_id: str
    start_requests: int
    stats: GitHubStats
    stopped: bool = False


def _parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _slim(pr: dict) -> dict:
    return {"number": pr.get("number"), "title": pr.get("title") or "", "body": pr.get("body") or "",
            "labels": [l.get("name", "") for l in pr.get("labels") or [] if isinstance(l, dict)],
            "url": pr.get("html_url") or "", "merged_at": pr.get("merged_at"),
            "merge_commit_sha": pr.get("merge_commit_sha")}


def _slim_pulls(data: Any) -> list[dict]:
    return [_slim(p) for p in data or [] if isinstance(p, dict)]


def _slim_issue(data: Any) -> dict:
    d = data if isinstance(data, dict) else {}
    return {"number": d.get("number"), "title": d.get("title") or "", "body": d.get("body") or "",
            "created_at": d.get("created_at"), "url": d.get("html_url") or "",
            "is_pr": "pull_request" in d}


class GitHubPRProvider:
    def __init__(self, client: GitHubClient, cache=None, *, fields: str = "title+body",
                 max_requests: int = 300, issues: bool = True, max_issues_per_pr: int = 3,
                 issue_chars: int = 2000) -> None:
        if fields not in ("title", "title+body"):
            raise ValueError("fields must be 'title' or 'title+body'")
        self.client = client
        self.cache = cache
        self.fields = fields
        self.max_requests = max_requests
        self.issues = issues and fields == "title+body"
        self.max_issues_per_pr = max_issues_per_pr
        self.issue_chars = issue_chars

    def enrich(self, repo: RepoRef, repo_id: str, commits: Sequence[CandidateCommit], *,
               cutoff: datetime | None = None) -> tuple[list[CandidateCommit], GitHubStats]:
        run = _Run(repo, repo_id, self.client.requests, GitHubStats())
        out: list[CandidateCommit] = []
        for c in commits:
            pulls = self._get(run, c.sha, KIND, f"/repos/{repo.slug}/commits/{c.sha}/pulls", _slim_pulls,
                              not_found=f"GitHub does not know {repo.slug} or commit {c.short_sha} (private "
                                        "repo without access, or commits not pushed) - PR data skipped")
            p, late = self._choose(c, pulls or [], cutoff)
            run.stats.skipped_after_cutoff += late
            if p is None:
                out.append(c)
                continue
            run.stats.with_pr += 1
            out.append(replace(c, pr=self._pr_info(run, p, cutoff)))
        run.stats.requests = self.client.requests - run.start_requests
        return out, run.stats

    # -- internals ---------------------------------------------------------------------------

    def _get(self, run: _Run, key: str, kind: str, path: str, slim: Callable[[Any], Any], *,
             not_found: str | None) -> Any:
        """Cached GET. not_found=<warning>: a 404 stops all further requests (commits);
        not_found=None: a 404 just means "no such item" (issues)."""
        cached, fresh = (self.cache.get_volatile(run.repo_id, key, kind) if self.cache else (None, False))
        if cached is not None and fresh:
            run.stats.cache_hits += 1
            return cached.payload
        budget_left = self.client.requests - run.start_requests < self.max_requests
        if not run.stopped and budget_left:
            try:
                resp = self.client.get(path, etag=cached.etag if cached else None)
                if resp.not_modified and cached is not None:
                    payload, etag = cached.payload, cached.etag
                    run.stats.not_modified += 1
                else:
                    payload, etag = slim(resp.data), resp.etag
                if self.cache:
                    self.cache.put_volatile(run.repo_id, key, kind, payload, etag)
                return payload
            except GitHubNotFound:
                if not_found is None:
                    return None
                run.stopped = True
                run.stats.warnings.append(not_found)
            except GitHubError as e:
                run.stopped = True
                run.stats.warnings.append(f"GitHub PR data incomplete: {e}")
        if cached is not None:
            return cached.payload                            # stale beats nothing
        if not run.stopped and not budget_left:
            run.stopped = True
            run.stats.warnings.append(f"stopped after {self.max_requests} GitHub requests (max_requests)")
        return None

    @staticmethod
    def _choose(c: CandidateCommit, pulls: list[dict], cutoff: datetime | None) -> tuple[dict | None, int]:
        late = 0
        usable = []
        for p in pulls:
            merged = _parse_time(p.get("merged_at"))
            if merged is None:
                continue                                     # open/closed-unmerged PRs did not land this code
            exact = (p.get("merge_commit_sha") or "") == c.sha
            if cutoff is not None and merged > cutoff and not exact:
                late += 1
                continue
            usable.append((p, merged, exact))
        if not usable:
            return None, late
        exact = [u for u in usable if u[2]]
        return (exact or sorted(usable, key=lambda u: u[1]))[0][0], late

    def _pr_info(self, run: _Run, p: dict, cutoff: datetime | None) -> PRInfo:
        body = p.get("body", "") if self.fields == "title+body" else ""
        numbers = linked_issues(p.get("body", ""), run.repo)
        if numbers and self.fields == "title+body":
            body += "\n\nLinked issues: " + ", ".join(f"#{n}" for n in numbers)
            if self.issues:
                body += "".join(self._issue_texts(run, numbers[:self.max_issues_per_pr], cutoff))
        return PRInfo(number=int(p["number"]), title=p.get("title", ""), body=body,
                      labels=tuple(p.get("labels") or ()), url=p.get("url", ""))

    def _issue_texts(self, run: _Run, numbers: list[int], cutoff: datetime | None) -> list[str]:
        texts = []
        for n in numbers:
            issue = self._get(run, f"issue#{n}", ISSUE_KIND, f"/repos/{run.repo.slug}/issues/{n}", _slim_issue,
                              not_found=None)
            if not issue or issue.get("is_pr"):
                continue                                     # deleted / transferred, or a PR number
            created = _parse_time(issue.get("created_at"))
            if cutoff is not None and (created is None or created > cutoff):
                run.stats.issues_skipped_after_cutoff += 1
                continue
            run.stats.issues_used += 1
            text = (issue.get("body") or "").strip()
            if len(text) > self.issue_chars:
                text = text[:self.issue_chars] + " ..."
            texts.append(f"\nIssue #{n}: {issue.get('title', '')}" + (f"\n{text}" if text else ""))
        return texts
