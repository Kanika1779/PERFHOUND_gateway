"""Repo Fetcher: turn a repository URL from a dataset into a local repo.

    fetcher = RepoFetcher()
    path = fetcher.fetch("https://github.com/pandas-dev/pandas")
    fetcher.ensure_commits(path, [good, bad])

* Clones are PARTIAL (--filter=blob:none) and without a checkout: commits
  and trees only (pandas: a fraction of a full clone). File contents are
  downloaded later in batches by the gateway (local_git.prefetch_blobs).
* One clone per repo URL under $PERFHOUND_REPOS_DIR (default
  ~/.perfhound/repos), reused across cases and runs.
* Clone folders ALWAYS stay inside that folder. A URL whose path has a "." or
  ".." segment is refused: "https://github.com/../../Documents" would
  otherwise point fetch() / update() / remove() at any folder on the disk.
* Clones made here are KEPT UP TO DATE. When a case names a branch or tag
  ("main", "v1.2", "HEAD~3") instead of a full SHA, the clone is refreshed
  from origin first (at most once per `refresh_seconds`), and its local
  branches follow origin: "main" means upstream's main now, not main as it
  was at the first clone. Offline -> a FetchWarning and the existing copy is
  used. Full SHAs need the network only when they are missing. Local-only
  branches (e.g. the injection branches perfhound/...) are never touched.
* A local path is used as-is: never cloned, never refreshed; its branches
  and files are never changed (missing commits may be fetched from its origin).
* Only https / http / ssh / file and scp-style (git@host:path) URLs are
  accepted: git's ext:: and fd:: transports can run arbitrary commands, so
  a URL from a dataset file must never reach `git clone` unchecked.
* Cloning goes to a temporary folder that is renamed into place, so a
  crash never leaves a half-cloned repo that looks complete.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .errors import GatewayError
from .gitcmd import run_git
from .worktree import _rmtree

_SCP_RE = re.compile(r"^[\w.-]+@[\w.-]+:[\w./~-]+$")
_ALLOWED_SCHEMES = {"https", "http", "ssh", "file"}
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
UPDATE_MARKER = "perfhound-updated"        # file in <clone>/.git: when the clone was last refreshed
DEFAULT_REFRESH_SECONDS = 600


class FetchError(GatewayError):
    """Repository could not be cloned / updated, or the URL is not allowed."""


class FetchWarning(UserWarning):
    """A clone could not be refreshed from origin; the existing (possibly old) copy is used."""


def default_repos_dir() -> Path:
    base = os.environ.get("PERFHOUND_REPOS_DIR")
    return Path(base) if base else Path.home() / ".perfhound" / "repos"


def is_remote(repo: str) -> bool:
    return "://" in repo or bool(_SCP_RE.match(repo))


def validate_url(url: str) -> None:
    if url.startswith("-"):
        raise FetchError(f"refusing repository URL {url!r}")
    if _SCP_RE.match(url):
        return
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise FetchError(f"repository URL scheme {scheme!r} is not allowed ({url!r})")


def _safe_part(part: str, url: str) -> str:
    """One folder name from a URL segment; "", "." and ".." (and "...") are refused."""
    clean = re.sub(r"[^\w.-]", "_", part)
    if not clean.strip("."):
        raise FetchError(f"refusing repository URL {url!r}: segment {part!r} would leave the repos folder")
    return clean


def _inside(child: Path, parent: Path) -> bool:
    """True if `child` is strictly inside `parent` (lexically, after normalizing)."""
    c, p = os.path.normcase(os.path.abspath(child)), os.path.normcase(os.path.abspath(parent))
    try:
        return c != p and os.path.commonpath([c, p]) == p
    except ValueError:                    # different drives on Windows
        return False


def local_dir_for(url: str, base: Path) -> Path:
    """github.com/pandas-dev/pandas -> <base>/github.com/pandas-dev/pandas (always inside `base`)."""
    if _SCP_RE.match(url):
        host, path = url.split("@", 1)[1].split(":", 1)
    else:
        p = urlparse(url)
        # file:// URLs are local, whatever sits in the host slot: on Windows "file://C:/x"
        # puts the drive there ("c"), and "file://server/share" a network host
        host = "local" if p.scheme.lower() == "file" else (p.hostname or "local")
        path = p.path
    segments = [x for x in path.strip("/").removesuffix(".git").split("/") if x]
    if host == "local" or not segments:   # file:// URLs: a hash, never the path itself
        parts = ["_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]]
    else:
        parts = [_safe_part(x, url) for x in segments]
    dest = base.joinpath(_safe_part(host, url), *parts)
    if not _inside(dest, base):           # belt and braces
        raise FetchError(f"refusing repository URL {url!r}: it maps outside {base}")
    return dest


class RepoFetcher:
    def __init__(self, base_dir: str | Path | None = None, *, filter_blobs: bool = True,
                 refresh_seconds: float = DEFAULT_REFRESH_SECONDS) -> None:
        # absolute: git runs with other working directories, a relative path would be resolved twice
        self.base_dir = (Path(base_dir) if base_dir else default_repos_dir()).resolve()
        self.filter_blobs = filter_blobs
        self.refresh_seconds = refresh_seconds

    def fetch(self, repo: str) -> Path:
        """Local path of `repo`, cloning it first if it is a URL seen for the first time."""
        if not is_remote(repo):
            path = Path(repo)
            if not path.exists():
                raise FetchError(f"local repository {repo!r} does not exist")
            return path
        validate_url(repo)
        dest = local_dir_for(repo, self.base_dir)
        if (dest / ".git").exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_tmp(dest)
        tmp = dest.parent / f".{dest.name}.tmp-{uuid.uuid4().hex[:8]}"
        args = ["clone", "--quiet", "--no-checkout"]
        if self.filter_blobs:
            args.append("--filter=blob:none")
        try:
            proc = run_git(self.base_dir, *args, "--", repo, str(tmp), check=False)
            if proc.returncode != 0:
                raise FetchError(f"could not clone {repo}: {proc.stderr.strip()}")
            _touch_marker(tmp)                  # fresh clone = up to date
            try:
                os.replace(tmp, dest)
            except OSError:
                if (dest / ".git").exists():     # another process won the race; use its clone
                    return dest
                raise
        finally:
            if tmp.exists():
                _rmtree(tmp)
        return dest

    STALE_TMP_SECONDS = 6 * 3600

    def _remove_stale_tmp(self, dest: Path) -> None:
        """Half-finished clones of killed processes (older than 6 h; a younger one may still be running)."""
        for p in dest.parent.glob(f".{dest.name}.tmp-*"):
            try:
                if time.time() - p.stat().st_mtime > self.STALE_TMP_SECONDS:
                    _rmtree(p)
            except OSError:
                pass

    # -- keeping clones current ---------------------------------------------------------------

    def is_managed(self, path: str | Path) -> bool:
        """True for a clone this fetcher made (inside base_dir) - the only repos it ever refreshes."""
        path = Path(path)
        return _inside(path.resolve(), self.base_dir) and (path / ".git").is_dir()

    def update(self, path: str | Path, *, max_age: float | None = None) -> bool:
        """Refresh a managed clone from origin; local branches then match origin's.

        Skipped when the last refresh is younger than `max_age` seconds
        (default: refresh_seconds; 0 = always). Returns True if it refreshed now.
        Never raises for network problems: warns and keeps the existing copy.
        """
        path = Path(path)
        if not self.is_managed(path):
            raise FetchError(f"{path} is not a clone made by RepoFetcher; refusing to change it")
        age = self.refresh_seconds if max_age is None else max_age
        marker = path / ".git" / UPDATE_MARKER
        try:
            if time.time() - marker.stat().st_mtime < age:
                return False
        except OSError:
            pass
        proc = run_git(path, "fetch", "--quiet", "--prune", "--tags", "--force", "origin", check=False)
        if proc.returncode != 0:
            warnings.warn(f"could not update {path} from origin ({proc.stderr.strip()[:200]}); "
                          f"using the existing copy{_marker_age(marker)}", FetchWarning, stacklevel=2)
            return False
        _sync_local_branches(path)
        _touch_marker(path)
        return True

    def ensure_commits(self, path: str | Path, refs: list[str]) -> list[str]:
        """Make sure `refs` exist locally (and, in a managed clone, that branch / tag names are
        current), fetching from origin if needed. Returns the refs still missing afterwards."""
        path = Path(path)

        def missing() -> list[str]:
            out = []
            for r in refs:
                if r.startswith("-") or run_git(path, "rev-parse", "--verify", "--quiet",
                                                f"{r}^{{commit}}", check=False).returncode != 0:
                    out.append(r)
            return out

        managed = self.is_managed(path)
        refreshed = False
        if managed and any(not _FULL_SHA.fullmatch(r) for r in refs):
            refreshed = self.update(path)          # names must mean what upstream has NOW
        todo = missing()
        if not todo:
            return []
        if run_git(path, "remote", "get-url", "origin", check=False).returncode != 0:
            return todo
        if managed:
            if not refreshed:
                self.update(path, max_age=0)
        else:
            run_git(path, "fetch", "--quiet", "--tags", "origin", check=False)
        todo = missing()
        for r in todo:   # e.g. commits only reachable from a PR ref - GitHub serves them by SHA
            if _FULL_SHA.fullmatch(r):
                run_git(path, "fetch", "--quiet", "origin", r, check=False)
        return missing()

    def remove(self, repo: str) -> None:
        """Delete the local clone of a URL (frees disk). Never deletes anything outside base_dir."""
        if is_remote(repo):
            dest = local_dir_for(repo, self.base_dir)
            if dest.exists():
                if not _inside(dest.resolve(), self.base_dir):     # e.g. a symlink pointing elsewhere
                    raise FetchError(f"refusing to delete {dest}: it is not inside {self.base_dir}")
                _rmtree(dest)


# ---------------------------------------------------------------- helpers

def _sync_local_branches(path: Path) -> None:
    """refs/heads/<b> = refs/remotes/origin/<b> for every branch origin has (one git process).
    Branches that exist only locally are left alone."""
    out = run_git(path, "for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/origin/").stdout
    prefix = "refs/remotes/origin/"
    lines = []
    for line in out.splitlines():
        ref, sha = line.rsplit(" ", 1)
        name = ref[len(prefix):]
        if name and name != "HEAD":
            lines.append(f"update refs/heads/{name} {sha}")
    if lines:
        run_git(path, "update-ref", "--stdin", input="\n".join(lines) + "\n")


def _touch_marker(path: Path) -> None:
    try:
        (path / ".git" / UPDATE_MARKER).write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    except OSError:
        pass


def _marker_age(marker: Path) -> str:
    try:
        when = datetime.fromtimestamp(marker.stat().st_mtime, timezone.utc)
    except OSError:
        return ""
    return f" from {when:%Y-%m-%d %H:%M} UTC"
