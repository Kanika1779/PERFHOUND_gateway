"""GitHub GraphQL provider: the PR behind each commit AND the issues it closes, for many
commits per request.

Same result as GitHubPRProvider (CandidateCommit.pr: title, body, linked issues), cheaper:

    REST provider:     1 request per commit (+1 per linked issue)
    GraphQL provider:  1 request per `per_request` commits (default 50), issues included

Why it matters: inside GitHub Actions the built-in GITHUB_TOKEN allows only 1,000 REST requests
per hour per repository - a big range would run out. GraphQL needs a token (GitHub has no
anonymous GraphQL); without one the gateway falls back to the REST provider.

Linked issues come from GitHub's own link (closingIssuesReferences: "Fixes #45" in the PR
text, or an issue linked in the PR's sidebar).

Time rule, as in the REST provider: cutoff=None (real use) keeps everything; a datetime
(evaluation) drops PRs merged and issues opened after it - except a commit's own PR.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Sequence

from ..models import CandidateCommit, PRInfo
from .client import GitHubClient
from .errors import GitHubError
from .links import RepoRef
from .provider import GitHubPRProvider, GitHubStats, _parse_time

KIND = "github_graphql_v1"

_COMMIT = ('c{i}: object(oid: "{sha}") {{ ... on Commit {{ associatedPullRequests(first: 5) {{ nodes {{ '
           'number title body url mergedAt mergeCommit {{ oid }} labels(first: 20) {{ nodes {{ name }} }} '
           'closingIssuesReferences(first: {issues}) {{ nodes {{ number title body createdAt url }} }} '
           '}} }} }} }}')


def build_query(shas: Sequence[str], issues_per_pr: int) -> str:
    parts = " ".join(_COMMIT.format(i=i, sha=sha, issues=max(1, issues_per_pr)) for i, sha in enumerate(shas))
    return f"query($owner: String!, $name: String!) {{ repository(owner: $owner, name: $name) {{ {parts} }} }}"


def _slim(node: dict) -> dict:
    return {"number": node.get("number"), "title": node.get("title") or "", "body": node.get("body") or "",
            "url": node.get("url") or "", "merged_at": node.get("mergedAt"),
            "merge_commit_sha": (node.get("mergeCommit") or {}).get("oid"),
            "labels": [l.get("name", "") for l in (node.get("labels") or {}).get("nodes") or [] if l],
            "issues": [{"number": i.get("number"), "title": i.get("title") or "", "body": i.get("body") or "",
                        "created_at": i.get("createdAt"), "url": i.get("url") or ""}
                       for i in (node.get("closingIssuesReferences") or {}).get("nodes") or [] if i]}


class GitHubGraphQLProvider:
    def __init__(self, client: GitHubClient, cache=None, *, fields: str = "title+body", per_request: int = 50,
                 max_requests: int = 200, max_issues_per_pr: int = 3, issue_chars: int = 2000) -> None:
        if fields not in ("title", "title+body"):
            raise ValueError("fields must be 'title' or 'title+body'")
        self.client = client
        self.cache = cache
        self.fields = fields
        self.per_request = max(1, per_request)
        self.max_requests = max_requests
        self.max_issues_per_pr = max_issues_per_pr
        self.issue_chars = issue_chars

    def enrich(self, repo: RepoRef, repo_id: str, commits: Sequence[CandidateCommit], *,
               cutoff: datetime | None = None) -> tuple[list[CandidateCommit], GitHubStats]:
        stats = GitHubStats()
        start = self.client.requests
        pulls: dict[str, list[dict]] = {}
        todo: list[str] = []
        for c in commits:
            cached, fresh = self.cache.get_volatile(repo_id, c.sha, KIND) if self.cache else (None, False)
            if cached is not None and fresh:
                pulls[c.sha] = cached.payload
                stats.cache_hits += 1
            else:
                todo.append(c.sha)

        for i in range(0, len(todo), self.per_request):
            if self.client.requests - start >= self.max_requests:
                stats.warnings.append(f"stopped after {self.max_requests} GitHub requests (max_requests)")
                break
            chunk = todo[i:i + self.per_request]
            error = self._fetch(repo, repo_id, chunk, pulls)
            if error:
                stats.warnings.append(error)
                break

        out = []
        for c in commits:
            p, late = GitHubPRProvider._choose(c, pulls.get(c.sha) or [], cutoff)
            stats.skipped_after_cutoff += late
            if p is None:
                out.append(c)
                continue
            stats.with_pr += 1
            out.append(replace(c, pr=self._pr_info(p, cutoff, stats)))
        stats.requests = self.client.requests - start
        return out, stats

    def _fetch(self, repo: RepoRef, repo_id: str, shas: list[str], pulls: dict[str, list[dict]]) -> str | None:
        """One GraphQL request for `shas`; fills `pulls`. Returns a warning text on failure."""
        try:
            resp = self.client.graphql(build_query(shas, self.max_issues_per_pr),
                                       {"owner": repo.owner, "name": repo.name})
        except GitHubError as exc:
            return f"GitHub PR data incomplete: {exc}"
        body = resp.data if isinstance(resp.data, dict) else {}
        errors = body.get("errors") or []
        data = (body.get("data") or {}).get("repository")
        if data is None:
            kinds = {e.get("type") for e in errors if isinstance(e, dict)}
            if "RATE_LIMITED" in kinds:
                return "GitHub request limit reached (GraphQL) - run again later"
            if "NOT_FOUND" in kinds:
                return f"GitHub does not know {repo.slug} (private repo without access?) - PR data skipped"
            msg = "; ".join(str(e.get("message", e)) for e in errors[:2] if isinstance(e, dict)) or "no data"
            return f"GitHub PR data incomplete: {msg}"
        for i, sha in enumerate(shas):
            node = data.get(f"c{i}")                     # None: GitHub does not know this commit (not pushed)
            nodes = ((node or {}).get("associatedPullRequests") or {}).get("nodes") or []
            pulls[sha] = [_slim(n) for n in nodes if n]
            if self.cache and node is not None:
                self.cache.put_volatile(repo_id, sha, KIND, pulls[sha], None)
        return None

    def _pr_info(self, p: dict, cutoff: datetime | None, stats: GitHubStats) -> PRInfo:
        body = p.get("body", "") if self.fields == "title+body" else ""
        issues = p.get("issues") or []
        if issues and self.fields == "title+body":
            body += "\n\nLinked issues: " + ", ".join(f"#{i['number']}" for i in issues)
            for issue in issues[:self.max_issues_per_pr]:
                created = _parse_time(issue.get("created_at"))
                if cutoff is not None and (created is None or created > cutoff):
                    stats.issues_skipped_after_cutoff += 1
                    continue
                stats.issues_used += 1
                text = (issue.get("body") or "").strip()
                if len(text) > self.issue_chars:
                    text = text[:self.issue_chars] + " ..."
                body += f"\nIssue #{issue['number']}: {issue.get('title', '')}" + (f"\n{text}" if text else "")
        return PRInfo(number=int(p["number"]), title=p.get("title", ""), body=body,
                      labels=tuple(p.get("labels") or ()), url=p.get("url", ""))
