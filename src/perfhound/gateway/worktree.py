"""Worktree Manager (gateway part 9): a private checkout of any commit.

Rule 2 of the guide: NEVER touch the user's folder. Benchmarks run in a
separate `git worktree` that lives in the temp directory:

    checkout(sha) -> Path      first call:  git worktree add --detach <tmp> <sha>
                               next calls:  same folder, git checkout --detach --force <sha>
                                            + git clean -ffdx (no leftovers from the previous commit)
    close()                    git worktree remove --force + git worktree prune

What this does touch in the user's repo: git's own bookkeeping under
.git/worktrees/<name> while the worktree exists (removed by close()).
No branches are created; HEAD, index, stash and working files are untouched.

Crash safety: every worktree has a lock file held open (OS-level lock) for
as long as its manager lives. A later manager removes worktrees whose lock
can be taken - i.e. whose owner process has died - and leaves worktrees of
other running Perfhound processes alone. (Checking PIDs is not used: on
Windows os.kill(pid, 0) would terminate the process.)

Known limitation: git submodules and Git LFS files are not populated.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from typing import IO

from .errors import GatewayError
from .gitcmd import run_git
from .range import find_repo_root, resolve_ref

_PREFIX = "wt-"


# ---------------------------------------------------------------- locking

def _try_lock(fh: IO[bytes]) -> bool:
    """Non-blocking exclusive lock on an open file. True if acquired."""
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fh: IO[bytes]) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _rmtree(path: Path) -> None:
    """shutil.rmtree that also deletes read-only files (common on Windows)."""

    def on_error(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if path.exists():
        shutil.rmtree(path, onerror=on_error)


def default_base_dir(repo: Path) -> Path:
    repo_id = hashlib.sha1(str(repo.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / "perfhound" / repo_id


# ---------------------------------------------------------------- manager

class WorktreeManager:
    """One reusable private worktree per manager. Use as a context manager:

        with WorktreeManager(repo) as wt:
            path = wt.checkout(sha)   # run the benchmark inside `path`
    """

    def __init__(self, repo: str | Path, *, base_dir: str | Path | None = None) -> None:
        self.repo = find_repo_root(repo)
        self.base_dir = Path(base_dir) if base_dir else default_base_dir(self.repo)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.removed_stale = self.cleanup_stale()

        name = _PREFIX + uuid.uuid4().hex[:10]
        self.path = self.base_dir / name
        self._lock_path = self.base_dir / (name + ".lock")
        self._lock_fh: IO[bytes] | None = open(self._lock_path, "a+b")
        if not _try_lock(self._lock_fh):
            raise GatewayError(f"could not lock {self._lock_path}")
        self._created = False
        self.current: str | None = None
        self._closed = False
        atexit.register(self.close)

    # -- public API ---------------------------------------------------------

    def checkout(self, ref: str) -> Path:
        """Put commit `ref` into the private worktree and return its folder."""
        if self._closed:
            raise GatewayError("WorktreeManager is closed")
        sha = resolve_ref(self.repo, ref)
        if not self._created:
            run_git(self.repo, "worktree", "add", "--detach", "--force", str(self.path), sha)
            self._created = True
        else:
            run_git(self.path, "checkout", "--quiet", "--detach", "--force", sha)
            run_git(self.path, "clean", "-ffdxq")   # untracked + ignored files of the previous run
        self.current = sha
        return self.path

    def close(self) -> None:
        """Remove the worktree and its bookkeeping. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._created:
                run_git(self.repo, "worktree", "remove", "--force", "--force", str(self.path), check=False)
            _rmtree(self.path)
            run_git(self.repo, "worktree", "prune", check=False)
        finally:
            if self._lock_fh is not None:
                _unlock(self._lock_fh)
                self._lock_fh.close()
                self._lock_fh = None
            try:
                self._lock_path.unlink()
            except OSError:
                pass
            atexit.unregister(self.close)

    def cleanup_stale(self) -> list[Path]:
        """Remove worktrees left behind by crashed Perfhound processes."""
        removed: list[Path] = []
        for lock_path in sorted(self.base_dir.glob(_PREFIX + "*.lock")):
            wt_path = lock_path.with_suffix("")
            try:
                fh = open(lock_path, "a+b")
            except OSError:
                continue
            try:
                if not _try_lock(fh):
                    continue                     # owner is still running
                run_git(self.repo, "worktree", "remove", "--force", "--force", str(wt_path), check=False)
                _rmtree(wt_path)
                removed.append(wt_path)
                _unlock(fh)
            finally:
                fh.close()
            try:
                lock_path.unlink()
            except OSError:
                pass
        run_git(self.repo, "worktree", "prune", check=False)
        return removed

    def __enter__(self) -> "WorktreeManager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
