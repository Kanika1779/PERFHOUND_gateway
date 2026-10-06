"""Ingest: ONE repository link in -> ONE folder with everything the system needs, as JSON.

    python -m perfhound.gateway https://github.com/psf/requests
    python -m perfhound.gateway https://github.com/psf/requests --limit 200      # quick test: newest 200 commits
    python -m perfhound.gateway https://github.com/psf/requests --comments       # also every issue comment

    from perfhound.gateway.ingest import ingest, load
    result = ingest("https://github.com/psf/requests")
    data = load(result.folder, until=<when `bad` landed>)          # leak-safe view, see below

Output folder (default: ingest/<owner>__<repo> in the current folder):

    repo.json            what was ingested: link, default branch, branches, tags, root commits,
                         GitHub repo info, counts, settings, warnings, complete / incomplete flags
    commits.jsonl        EVERY commit reachable from the default branch, oldest first, one per line:
                         CandidateCommit fields (message, author, date, changed files, diff,
                         changed / added / deleted functions, its PR) + parents, committed_at,
                         mainline (true = on the first-parent line of the default branch)
    pull_requests.jsonl  every PR (open, closed, merged): title, body, labels, dates, merge commit
    issues.jsonl         every issue (PRs excluded): title, body, state, labels, dates, author,
                         comment count (+ the comments themselves with --comments)

Costs, honestly:
  * Commits: one partial clone, then batches of `batch_size` commits; file contents are
    downloaded batch by batch. Big repos (pandas: ~35k commits) take a while and give a
    commits.jsonl of a few hundred MB. Re-runs are fast: every processed commit is cached.
  * GitHub: 1 request per 100 PRs / 100 issues (+1 per issue with --comments). Without a token
    GitHub allows 60 requests/hour, so big repos need `python -m perfhound.gateway login` (5,000/hour).
    A run stopped by the rate limit keeps what it got and says so (repo.json "complete");
    run it again later: only PRs / issues updated since the last run are fetched again.
  * Root commits (no parent) have no diff; they are listed in repo.json, not in commits.jsonl.

LEAK WARNING: the folder holds the FUTURE of every commit - later PRs, reverts, issues like
"X got slow since #123". Never give it to the prioritizer / localizer as it is.
load(folder, until=<committer date of bad>) returns only what existed at that moment.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .analyzer import ANALYZER_VERSION
from .api import Gateway
from .cache import Cache
from .errors import GatewayError
from .fetcher import RepoFetcher, is_remote, validate_url
from .github.client import GitHubClient
from .github.errors import GitHubError, InvalidGitHubLink
from .github.links import RepoRef, parse_repo
from .gitcmd import run_git
from .local_git import parse_git_date
from .models import PRInfo
from .range import CommitRange

INGEST_FORMAT_VERSION = 1
DEFAULT_BATCH = 500
SINCE_MARGIN_SECONDS = 3600            # re-fetch a little overlap on incremental GitHub runs


@dataclass
class IngestResult:
    folder: Path
    repo_path: Path
    commits: int
    pull_requests: int
    issues: int
    complete: bool
    warnings: list[str] = field(default_factory=list)


# =================================================================== link -> clone URL

def resolve_link(link: str) -> tuple[str, RepoRef | None]:
    """Repository link -> (clone URL, GitHub repo or None).

    Accepts https://github.com/o/r (also .../tree/main/..., .../pulls, .git), git@github.com:o/r.git,
    "o/r", and any other git URL (GitLab, file://...: no GitHub data then).
    """
    link = link.strip()
    if not link:
        raise GatewayError("empty repository link")
    looks_short = re.fullmatch(r"[\w.-]+/[\w.-]+", link) is not None and not Path(link).exists()
    if "github.com" in link or looks_short:
        try:
            ref = parse_repo(link)
            return ref.url, ref
        except InvalidGitHubLink as exc:
            raise GatewayError(f"{link!r} is not a GitHub repository link: {exc}") from None
    if is_remote(link):
        validate_url(link)
        return link, None
    raise GatewayError(f"{link!r} is not a repository link (expected e.g. https://github.com/owner/name)")


def default_out_dir(clone_url: str, ref: RepoRef | None) -> Path:
    if ref is not None:
        name = f"{ref.owner}__{ref.name}"
    else:
        name = re.sub(r"[^\w.-]", "_", clone_url.rstrip("/").split("/")[-1].removesuffix(".git")) or "repo"
    return Path("ingest") / name


# =================================================================== main entry point

def ingest(link: str, out: str | Path | None = None, *, limit: int | None = None, github: bool = True,
           comments: bool = False, client: GitHubClient | None = None, github_repo: RepoRef | None = None,
           fetcher: RepoFetcher | None = None, cache: Cache | bool = True, batch_size: int = DEFAULT_BATCH,
           log: Callable[[str], None] = print) -> IngestResult:
    """Repository link -> folder of JSON files. See the module docstring."""
    t0 = time.perf_counter()
    clone_url, ref = resolve_link(link)
    ref = github_repo or ref
    folder = Path(out) if out else default_out_dir(clone_url, ref)
    folder.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    # 1. clone (first time) or bring the clone up to date
    fetcher = fetcher or RepoFetcher()
    log(f"[1/4] repository {clone_url}")
    path = fetcher.fetch(clone_url)
    if fetcher.is_managed(path):
        fetcher.update(path, max_age=0)
    branch, tip = _default_branch(path)
    old_repo = _read_json(folder / "repo.json")

    # 2. GitHub: repo info, every PR, every issue (first, so commits can carry their PR)
    gh: dict[str, Any] = {"complete": None, "repo": (old_repo or {}).get("github_repo"),
                          "pulls": list(_read_jsonl(folder / "pull_requests.jsonl").values()),
                          "issues": list(_read_jsonl(folder / "issues.jsonl").values()),
                          "sync": (old_repo or {}).get("github_sync"), "warnings": []}
    if github and ref is not None:
        if client is None:
            from .github.auth import find_token
            client = GitHubClient(find_token()[0])
        log(f"[2/4] GitHub {ref.slug}: pull requests and issues" + (" (+ comments)" if comments else ""))
        gh = _fetch_github(client, ref, folder, old_repo, comments=comments, log=log)
        warnings += gh["warnings"]
    else:
        log("[2/4] GitHub skipped" + ("" if github else " (--no-github)") + ("" if ref else " (not a GitHub repo)"))
    pr_by_commit = {p["merge_commit_sha"]: p for p in gh["pulls"] if p.get("merge_commit_sha") and p.get("merged_at")}

    # 3. every commit of the default branch, in batches
    rows = _commit_rows(path, tip, limit)
    roots = [r for r in rows if not r["parents"]]
    todo = [r for r in rows if r["parents"]]
    mainline = set(run_git(path, "rev-list", "--first-parent", tip).stdout.split())
    log(f"[3/4] {len(todo)} commits on {branch}" + (f" (newest {limit})" if limit else "") + f", batches of {batch_size}")
    gw = Gateway(path, cache=cache)
    tmp = folder / "commits.jsonl.tmp"
    n_commits = 0
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                shas = tuple(r["sha"] for r in batch)
                commits, _ = gw._load_commits(CommitRange(gw.repo, "", "", "", "", shas))
                commits, _ = gw._add_functions(commits)
                for row, c in zip(batch, commits):
                    c = replace(c, position=c.position + start)            # position in the whole history
                    p = pr_by_commit.get(c.sha)
                    if p is not None:
                        c = replace(c, pr=PRInfo(number=int(p["number"]), title=p["title"], body=p["body"],
                                                 labels=tuple(p["labels"]), url=p["url"]), source="local+github")
                    rec = c.to_dict()
                    rec.update(parents=row["parents"], committed_at=row["committed_at"], mainline=c.sha in mainline)
                    fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                    n_commits += 1
                log(f"      commits {min(start + batch_size, len(todo))}/{len(todo)}  ({time.perf_counter() - t0:.0f} s)")
        os.replace(tmp, folder / "commits.jsonl")
    finally:
        gw.close()
        if tmp.exists():
            tmp.unlink()

    # 4. repo.json
    complete = gh["complete"]
    repo_info = {
        "format_version": INGEST_FORMAT_VERSION,
        "link": link, "clone_url": clone_url, "github": ref.slug if ref else None,
        "local_clone": str(path),
        "ingested_at": _now(),
        "default_branch": branch, "head": tip,
        "branches": _branches(path), "tags": _tags(path),
        "root_commits": roots,
        "counts": {"commits": n_commits, "pull_requests": len(gh["pulls"]), "issues": len(gh["issues"])},
        "complete": {"commits": True, "github": complete},
        "settings": {"limit": limit, "batch_size": batch_size, "comments": comments,
                     "max_diff_lines": gw.max_diff_lines, "max_diff_chars": gw.max_diff_chars,
                     "analyzer_version": ANALYZER_VERSION},
        "github_repo": gh["repo"],
        "github_sync": gh["sync"],
        "warnings": warnings,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    _write_json(folder / "repo.json", repo_info)
    log(f"[4/4] {folder}: {n_commits} commits, {len(gh['pulls'])} PRs, {len(gh['issues'])} issues"
        + ("  (GitHub data INCOMPLETE - run the same command again later to continue)" if complete is False else ""))
    for w in warnings:
        log(f"      warning: {w}")
    return IngestResult(folder, path, n_commits, len(gh["pulls"]), len(gh["issues"]), complete is not False,
                        warnings)


# =================================================================== git side

def _default_branch(path: Path) -> tuple[str, str]:
    """(branch name, tip SHA) of the default branch."""
    proc = run_git(path, "symbolic-ref", "--short", "refs/remotes/origin/HEAD", check=False)
    name = proc.stdout.strip().removeprefix("origin/") if proc.returncode == 0 else ""
    if not name:
        proc = run_git(path, "symbolic-ref", "--short", "HEAD", check=False)
        name = proc.stdout.strip() if proc.returncode == 0 else "HEAD"
    for candidate in (f"refs/remotes/origin/{name}", f"refs/heads/{name}", "HEAD"):
        proc = run_git(path, "rev-parse", "--verify", "--quiet", candidate + "^{commit}", check=False)
        if proc.returncode == 0:
            return name, proc.stdout.strip()
    raise GatewayError(f"{path}: cannot find the default branch (empty repository?)")


def _commit_rows(path: Path, tip: str, limit: int | None) -> list[dict]:
    """Every commit reachable from `tip`, parents before children (oldest first)."""
    args = ["log", "--topo-order", "--format=%H%x1f%P%x1f%cI%x1f%ct"]
    if limit:
        args.append(f"--max-count={int(limit)}")
    out = run_git(path, *args, tip).stdout
    rows = []
    for line in reversed(out.splitlines()):
        if not line.strip():
            continue
        sha, parents, iso, epoch = line.split("\x1f")
        rows.append({"sha": sha, "parents": parents.split(), "committed_at": parse_git_date(iso, epoch).isoformat()})
    return rows


def _branches(path: Path) -> dict[str, str]:
    out = run_git(path, "for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/origin/").stdout
    branches = {}
    for line in out.splitlines():
        ref, sha = line.rsplit(" ", 1)
        name = ref.removeprefix("refs/remotes/origin/")
        if name != "HEAD":
            branches[name] = sha
    if not branches:     # no origin (should not happen for a clone): local branches instead
        out = run_git(path, "for-each-ref", "--format=%(refname:short) %(objectname)", "refs/heads/").stdout
        branches = dict(line.rsplit(" ", 1) for line in out.splitlines() if line.strip())
    return dict(sorted(branches.items()))


def _tags(path: Path) -> dict[str, str]:
    """tag -> commit SHA (annotated tags peeled)."""
    out = run_git(path, "for-each-ref", "--format=%(refname:short)%00%(objectname)%00%(*objectname)",
                  "refs/tags/").stdout
    tags = {}
    for line in out.splitlines():
        if line.strip():
            name, obj, peeled = line.split("\x00")
            tags[name] = peeled or obj
    return dict(sorted(tags.items()))


# =================================================================== GitHub side

def _slim_pull(p: dict) -> dict:
    return {"number": p.get("number"), "title": p.get("title") or "", "body": p.get("body") or "",
            "state": p.get("state"), "draft": bool(p.get("draft")),
            "labels": [l.get("name", "") for l in p.get("labels") or [] if isinstance(l, dict)],
            "author": (p.get("user") or {}).get("login"),
            "created_at": p.get("created_at"), "updated_at": p.get("updated_at"),
            "closed_at": p.get("closed_at"), "merged_at": p.get("merged_at"),
            "merge_commit_sha": p.get("merge_commit_sha"),
            "base": (p.get("base") or {}).get("ref"), "head": (p.get("head") or {}).get("ref"),
            "url": p.get("html_url") or ""}


def _slim_issue(i: dict) -> dict:
    return {"number": i.get("number"), "title": i.get("title") or "", "body": i.get("body") or "",
            "state": i.get("state"), "state_reason": i.get("state_reason"),
            "labels": [l.get("name", "") for l in i.get("labels") or [] if isinstance(l, dict)],
            "author": (i.get("user") or {}).get("login"),
            "created_at": i.get("created_at"), "updated_at": i.get("updated_at"), "closed_at": i.get("closed_at"),
            "comments": i.get("comments", 0), "url": i.get("html_url") or ""}


def _slim_repo(r: dict) -> dict:
    keys = ("full_name", "description", "default_branch", "language", "topics", "stargazers_count",
            "forks_count", "open_issues_count", "created_at", "pushed_at", "html_url", "fork", "archived")
    out = {k: r.get(k) for k in keys}
    out["license"] = (r.get("license") or {}).get("spdx_id")
    out["parent"] = (r.get("parent") or {}).get("full_name")
    return out


def _last_page(link_header: str) -> int | None:
    m = re.search(r'[?&]page=(\d+)[^>]*>;\s*rel="last"', link_header or "")
    return int(m.group(1)) if m else None


def _next(link_header: str) -> str | None:
    m = re.search(r'<([^>]+)>;\s*rel="next"', link_header or "")
    return m.group(1) if m else None


def _pages(client: GitHubClient, path: str, params: dict, *, start_page: int = 1,
           stop: Callable[[dict], bool] | None = None, log: Callable[[str], None],
           what: str) -> tuple[list[dict], int | None, str | None]:
    """Items of a paginated list from `start_page` on -> (items, next page or None if done, error).
    On an error the items so far are kept and the page to resume from is returned."""
    items: list[dict] = []
    page = start_page
    while True:
        try:
            resp = client.get(path, {**params, "per_page": 100, "page": page})
        except GitHubError as exc:
            return items, page, f"{what}: {exc}"
        if page == start_page:
            last = _last_page(resp.headers.get("link", ""))
            if last and last > start_page:
                left = f", requests left this hour: {client.rate_remaining}" if client.rate_remaining is not None else ""
                log(f"      {what}: pages {start_page}..{last} of 100{left}")
        batch = [x for x in resp.data or [] if isinstance(x, dict)]
        items.extend(batch)
        if not batch or not _next(resp.headers.get("link", "")) or (stop is not None and stop(batch[-1])):
            return items, None, None
        page += 1


def _fetch_github(client: GitHubClient, ref: RepoRef, folder: Path, old_repo: dict | None, *, comments: bool,
                  log: Callable[[str], None]) -> dict[str, Any]:
    """Every PR and issue of the repo, resumable.

    First time: a FULL sync, oldest first, page by page. If the rate limit stops it, the next
    run continues at the page where it stopped. After a full sync, later runs are INCREMENTAL:
    only items updated since the last complete sync (minus a margin) are fetched again.
    """
    now = _now()
    old_repo = old_repo or {}
    sync = dict(old_repo.get("github_sync") or {}) if old_repo.get("github") == ref.slug else {}
    pulls = _read_jsonl(folder / "pull_requests.jsonl") if sync else {}
    issues = _read_jsonl(folder / "issues.jsonl") if sync else {}
    warnings: list[str] = []
    errors: list[str] = []

    repo_info = old_repo.get("github_repo")
    try:
        repo_info = _slim_repo(client.get_json(f"/repos/{ref.slug}") or {})
    except GitHubError as exc:
        errors.append(f"repository info: {exc}")

    def merge(kind: str, raw: list[dict]) -> None:
        for x in raw:
            if kind == "pulls":
                pulls[x["number"]] = _slim_pull(x)
            elif "pull_request" not in x:                     # the issues API lists PRs too
                new = _slim_issue(x)
                old = issues.get(new["number"])
                if old and "comment_list" in old and old.get("updated_at") == new["updated_at"]:
                    new["comment_list"] = old["comment_list"]   # unchanged issue: keep its comments
                issues[new["number"]] = new

    paths = {"pulls": f"/repos/{ref.slug}/pulls", "issues": f"/repos/{ref.slug}/issues"}
    names = {"pulls": "pull requests", "issues": "issues"}
    if not errors and sync.get("last_complete_at"):          # ---- incremental
        since = datetime.fromtimestamp(_parse(sync["last_complete_at"]).timestamp() - SINCE_MARGIN_SECONDS,
                                       timezone.utc)
        older = lambda x: (t := _parse(x.get("updated_at"))) is not None and t < since
        for kind in ("pulls", "issues"):
            params = {"state": "all", "sort": "updated", "direction": "desc"}
            if kind == "issues":
                params["since"] = since.isoformat().replace("+00:00", "Z")
            raw, _, err = _pages(client, paths[kind], params, stop=older if kind == "pulls" else None,
                                 log=log, what=names[kind])
            merge(kind, raw)
            if err:
                errors.append(err)
                break
        if not errors:
            sync = {"last_complete_at": now}
    elif not errors:                                         # ---- full sync, resumable
        sync.setdefault("started_at", now)
        for kind in ("pulls", "issues"):
            page = sync.get(f"{kind}_next_page", 1)
            if page is None:
                continue                                     # this list is already complete
            raw, next_page, err = _pages(client, paths[kind], {"state": "all", "sort": "created", "direction": "asc"},
                                         start_page=page, log=log, what=names[kind])
            merge(kind, raw)
            sync[f"{kind}_next_page"] = next_page
            if err:
                errors.append(err)
                break
        if not errors:
            sync = {"last_complete_at": sync["started_at"]}  # updates during the sync: next run catches them

    if comments and not errors:
        need = [i for i in issues.values() if i.get("comments") and "comment_list" not in i]
        if need:
            log(f"      comments: {len(need)} issues (1+ request each)")
        for n, issue in enumerate(need, 1):
            got, _, err = _pages(client, f"/repos/{ref.slug}/issues/{issue['number']}/comments", {},
                                 log=lambda s: None, what=f"comments of #{issue['number']}")
            if err:
                errors.append(err)
                break
            issue["comment_list"] = [{"author": (c.get("user") or {}).get("login"), "created_at": c.get("created_at"),
                                      "body": c.get("body") or ""} for c in got]
            if n % 200 == 0:
                log(f"      comments {n}/{len(need)}")

    _write_jsonl(folder / "pull_requests.jsonl", [pulls[k] for k in sorted(pulls)])
    _write_jsonl(folder / "issues.jsonl", [issues[k] for k in sorted(issues)])
    warnings += errors
    if any("limit" in e for e in errors):
        warnings.append("GitHub request limit reached: run the same command again after the reset to continue "
                        "(`python -m perfhound.gateway login` raises the limit to 5,000/hour)")
    complete = not errors and "last_complete_at" in sync
    return {"complete": complete, "repo": repo_info, "pulls": list(pulls.values()), "issues": list(issues.values()),
            "sync": sync, "warnings": warnings}


# =================================================================== reading a folder back

def load(folder: str | Path, *, until: datetime | None = None) -> dict[str, Any]:
    """Read an ingest folder: {"repo", "commits", "pull_requests", "issues"} (lists of dicts).

    until=<datetime>: what the world looked like at that moment (use the committer date of `bad`,
    Gateway.landed_at(bad)). Commits committed later, PRs / issues opened later and comments
    written later are dropped; PRs merged later look unmerged, issues closed later look open.
    """
    folder = Path(folder)
    repo = _read_json(folder / "repo.json") or {}
    commits = list(_iter_jsonl(folder / "commits.jsonl"))
    pulls = list(_read_jsonl(folder / "pull_requests.jsonl").values())
    issues = list(_read_jsonl(folder / "issues.jsonl").values())
    if until is not None:
        if until.tzinfo is None:
            raise ValueError("until must be timezone-aware")
        before = lambda s: (t := _parse(s)) is not None and t <= until
        commits = [c for c in commits if before(c.get("committed_at"))]
        for c in commits:
            if c.get("pr") and not _pr_merged_by(pulls, c["pr"]["number"], until):
                c["pr"] = None                     # merged later (or unknown): not visible yet
        pulls = [_as_of_pull(p, until) for p in pulls if before(p.get("created_at"))]
        issues = [_as_of_issue(i, until) for i in issues if before(i.get("created_at"))]
    return {"repo": repo, "commits": commits, "pull_requests": pulls, "issues": issues}


def _pr_merged_by(pulls: list[dict], number: int, until: datetime) -> bool:
    for p in pulls:
        if p.get("number") == number:
            t = _parse(p.get("merged_at"))
            return t is not None and t <= until
    return True        # PR list not ingested (no GitHub): the commit's own PR landed with the commit


def _as_of_pull(p: dict, until: datetime) -> dict:
    p = dict(p)
    for key in ("merged_at", "closed_at"):
        t = _parse(p.get(key))
        if t is not None and t > until:
            p[key] = None
    if p.get("merged_at") is None:
        p["merge_commit_sha"] = None
    if p.get("closed_at") is None:
        p["state"] = "open"
    return p


def _as_of_issue(i: dict, until: datetime) -> dict:
    i = dict(i)
    t = _parse(i.get("closed_at"))
    if t is not None and t > until:
        i.update(closed_at=None, state="open", state_reason=None)
    if "comment_list" in i:
        i["comment_list"] = [c for c in i["comment_list"] if (ct := _parse(c.get("created_at"))) and ct <= until]
        i["comments"] = len(i["comment_list"])
    return i


# =================================================================== small file helpers

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
    except FileNotFoundError:
        return


def _read_jsonl(path: Path) -> dict[int, dict]:
    return {d["number"]: d for d in _iter_jsonl(path) if "number" in d}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)
