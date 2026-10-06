"""Range fetch: only the commits between `good` and `bad`, live, with their PRs and linked issues.

This is what a real user (and the GitHub Action) needs: "it got slow between v1.2 and main".
Nothing else of the project's history is processed.

    python -m perfhound.gateway range https://github.com/psf/requests --good v2.31.0 --bad v2.32.0
    python -m perfhound.gateway range https://github.com/psf/requests/compare/v2.31.0...v2.32.0
    python -m perfhound.gateway range .  --good HEAD~20 --bad HEAD           # a repo already on disk

Input:  a GitHub link, a compare link (good...bad inside), or a folder with a git repo
        (inside GitHub Actions: the checked-out workspace - no extra download).
Output: ONE JSON file (default perfhound-range.json) + an optional Markdown summary:
        repo, good / bad (ref, SHA, when bad landed), mode, every commit oldest first
        (message, author, dates, files, diff, changed functions, its PR + linked issues).

Modes:  real (default)  - every clue GitHub has, e.g. a later "slow since #123" issue.
        evaluation      - nothing that appeared after `bad` landed (for testing the system).

GitHub data: GraphQL (50 commits per request) when a token is available, else REST
(1 request per commit, 60/hour without a token). Without GitHub (or not a GitHub repo):
git data only.
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings as pywarnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .api import Gateway
from .cache import Cache
from .errors import GatewayError
from .fetcher import RepoFetcher
from .gitcmd import run_git
from .github.client import GitHubClient
from .github.errors import InvalidGitHubLink
from .github.links import RepoRef, is_github, parse_compare, parse_repo
from .local_git import parse_git_date

RANGE_FORMAT_VERSION = 1
ZERO_SHA = "0" * 40
DEFAULT_OUT = "perfhound-range.json"


def resolve_input(repo: str, good: str | None, bad: str | None) -> tuple[str, str | None, str | None, RepoRef | None]:
    """(repo link or folder, good, bad, GitHub repo) - a compare link fills good and bad."""
    repo = repo.strip()
    if "/compare/" in repo:
        try:
            ref, g, b = parse_compare(repo)
        except InvalidGitHubLink as exc:
            raise GatewayError(str(exc)) from None
        return ref.url, good or g, bad or b, ref
    if Path(repo).exists():
        return repo, good, bad, None
    from .ingest import resolve_link
    url, ref = resolve_link(repo)
    return url, good, bad, ref


def fetch_range(repo: str, good: str | None = None, bad: str | None = None, *, out: str | Path | None = DEFAULT_OUT,
                github: bool = True, github_repo: RepoRef | str | None = None, client: GitHubClient | None = None,
                evaluation: bool = False, from_merge_base: bool = False, fetcher: RepoFetcher | None = None,
                cache: Cache | bool = True, batch_size: int = 500,
                log: Callable[[str], None] = print) -> dict[str, Any]:
    """Commits good..bad (+ PRs, linked issues) -> dict, also written to `out` (None: not written)."""
    t0 = time.perf_counter()
    source, good, bad, ref = resolve_input(repo, good, bad)
    warn: list[str] = []

    # 1. the repository: a folder on disk is used as-is; a link is cloned once and kept current
    if Path(source).exists():
        path = Path(source)
    else:
        fetcher = fetcher or RepoFetcher()
        log(f"[1/3] repository {source}")
        path = fetcher.fetch(source)
        names = [r for r in (good, bad) if r and r != ZERO_SHA]
        missing = fetcher.ensure_commits(path, names or ["HEAD"])
        if missing:
            raise GatewayError(f"not found in {source}: {', '.join(missing)}")

    if isinstance(github_repo, str) and github_repo:
        github_repo = parse_repo(github_repo)
    ref = github_repo or ref or _env_repo() or _origin_repo(path)

    # 2. which range
    bad = bad or "HEAD"
    if not good or good == ZERO_SHA:                 # first push of a branch / nothing given: just `bad`
        if run_git(path, "rev-parse", "--verify", "--quiet", f"{bad}^1^{{commit}}", check=False).returncode != 0:
            return _nothing_to_compare(repo, path, bad, ref, evaluation, out, t0, log)
        good = f"{bad}^"
        warn.append(f"no good commit given - checking only {bad} itself (good = its parent)")
    if from_merge_base:                              # pull request: only the PR's own commits
        mb = run_git(path, "merge-base", good, bad, check=False)
        if mb.returncode == 0 and mb.stdout.strip():
            good = mb.stdout.strip()

    # 3. GitHub provider
    provider = None
    if github and ref is not None:
        provider = _provider(client)

    log(f"[2/3] commits {good} .. {bad}" + (f"  (GitHub {ref.slug}, "
                                             f"{'GraphQL' if _is_graphql(provider) else 'REST'})" if provider else ""))
    with pywarnings.catch_warnings(record=True) as caught:
        pywarnings.simplefilter("always")
        with Gateway(path, cache=cache, github=provider, github_repo=ref, time_rule=evaluation,
                     batch_size=batch_size) as gw:
            commit_range = gw.resolve(good, bad)
            commits = gw.get_candidates(commit_range.good, commit_range.bad)
            stats = gw.last_stats
            gh_with_pr, gh_requests, gh_warnings = stats.github_with_pr, stats.github_requests, list(stats.github_warnings)
            if _is_graphql(provider) and gh_with_pr == 0 and gh_warnings:
                # GraphQL refused (e.g. a token that may not use GraphQL on this repository): try REST once
                from .github.provider import GitHubPRProvider
                gw.github = GitHubPRProvider(provider.client, provider.cache)
                commits, gh = gw._add_github([replace(c, pr=None) for c in commits])
                gh_warnings = [f"GitHub GraphQL failed ({'; '.join(gh_warnings)}) - used REST instead", *gh.warnings]
                gh_with_pr, gh_requests = gh.with_pr, gh_requests + gh.requests
            landed = gw.landed_at(commit_range.bad)
    warn += [str(w.message) for w in caught]
    warn += gh_warnings

    committed = _committed_at(path, [c.sha for c in commits])
    rows = []
    for c in commits:
        d = c.to_dict()
        d["committed_at"] = committed.get(c.sha)
        rows.append(d)

    result = {
        "format_version": RANGE_FORMAT_VERSION,
        "repo": repo, "github": ref.slug if ref else None,
        "mode": "evaluation" if evaluation else "real",
        "good": {"ref": good, "sha": commit_range.good},
        "bad": {"ref": bad, "sha": commit_range.bad, "landed_at": landed.isoformat()},
        "count": len(rows),
        "commits": rows,
        "stats": {"seconds": round(time.perf_counter() - t0, 2), "from_cache": stats.commit_cache_hits,
                  "github_requests": gh_requests, "with_pr": gh_with_pr},
        "warnings": warn,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(out) + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, out)
    log(f"[3/3] {len(rows)} commits, {gh_with_pr} with a PR ({gh_requests} GitHub requests, "
        f"{result['stats']['seconds']} s)" + (f" -> {out}" if out else ""))
    for w in warn:
        log(f"      warning: {w}")
    return result


def summary_markdown(result: dict[str, Any], max_rows: int = 50) -> str:
    """A short Markdown report (GitHub Actions job summary / PR comment)."""
    g, b = result["good"], result["bad"]
    short = lambda ref: ref[:7] if re.fullmatch(r"[0-9a-f]{40}", ref or "") else (ref or "the start")
    lines = [f"### Perfhound: {result['count']} commits between `{short(g['ref'])}` and `{short(b['ref'])}`", "",
             f"Repository: {result['github'] or result['repo']} · mode: {result['mode']} · "
             f"{result['stats']['with_pr']} with a PR · {result['stats']['seconds']} s", "",
             "| # | commit | date | author | PR | title | functions changed |",
             "|---|---|---|---|---|---|---|"]
    for c in result["commits"][:max_rows]:
        title = c["message"].split("\n", 1)[0].replace("|", "\\|")[:70]
        pr = f"#{c['pr']['number']}" if c.get("pr") else ""
        fns = list(c.get("changed_functions") or []) + list(c.get("added_functions") or [])
        more = f" +{len(fns) - 3}" if len(fns) > 3 else ""
        lines.append(f"| {c['position'] + 1} | `{c['sha'][:7]}` | {(c.get('committed_at') or c['timestamp'])[:10]} | "
                     f"{c['author'][:20]} | {pr} | {title} | {', '.join(f'`{f}`' for f in fns[:3])}{more} |")
    if result["count"] > max_rows:
        lines.append(f"\n... and {result['count'] - max_rows} more commits (see the JSON file).")
    if result["warnings"]:
        lines += ["", "**Warnings**", *[f"- {w}" for w in result["warnings"]]]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------- helpers

def _nothing_to_compare(repo, path, bad, ref, evaluation, out, t0, log) -> dict[str, Any]:
    """`bad` is the repository's very first commit: there is nothing before it to compare with."""
    if isinstance(ref, str):
        ref = parse_repo(ref) if ref else None
    sha = run_git(path, "rev-parse", "--verify", "--quiet", f"{bad}^{{commit}}", check=False).stdout.strip()
    if not sha:
        raise GatewayError(f"{bad!r} is not a commit, tag or branch in this repository")
    warning = f"{bad} is the first commit of the repository - there is nothing before it to compare with"
    result = {
        "format_version": RANGE_FORMAT_VERSION, "repo": repo, "github": ref.slug if ref else None,
        "mode": "evaluation" if evaluation else "real",
        "good": {"ref": None, "sha": None}, "bad": {"ref": bad, "sha": sha, "landed_at": None},
        "count": 0, "commits": [],
        "stats": {"seconds": round(time.perf_counter() - t0, 2), "from_cache": 0, "github_requests": 0, "with_pr": 0},
        "warnings": [warning], "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[3/3] 0 commits: {warning}")
    return result


def _provider(client: GitHubClient | None):
    from .github.graphql import GitHubGraphQLProvider
    from .github.provider import GitHubPRProvider

    if client is None:
        from .github.auth import find_token
        client = GitHubClient(find_token()[0])
    cache = Cache()
    # GraphQL needs a token; REST works without one (60 requests/hour)
    return (GitHubGraphQLProvider(client, cache) if client.token else
            GitHubPRProvider(client, cache, max_requests=55))


def _is_graphql(provider) -> bool:
    return type(provider).__name__ == "GitHubGraphQLProvider"


def _env_repo() -> RepoRef | None:
    """Inside GitHub Actions: GITHUB_REPOSITORY = owner/name."""
    value = os.environ.get("GITHUB_REPOSITORY", "")
    return parse_repo(value) if value and is_github(value) else None


def _origin_repo(path: Path) -> RepoRef | None:
    proc = run_git(path, "remote", "get-url", "origin", check=False)
    url = proc.stdout.strip() if proc.returncode == 0 else ""
    return parse_repo(url) if url and is_github(url) else None


def _committed_at(path: Path, shas: list[str]) -> dict[str, str]:
    if not shas:
        return {}
    out = run_git(path, "log", "--no-walk=unsorted", "--stdin", "--format=%H%x1f%cI%x1f%ct",
                  input="\n".join(shas) + "\n").stdout
    result = {}
    for line in out.splitlines():
        if line.count("\x1f") == 2:
            sha, iso, epoch = line.split("\x1f")
            result[sha] = parse_git_date(iso, epoch).isoformat()
    return result
