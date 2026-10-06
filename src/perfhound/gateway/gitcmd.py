"""The ONE place where Perfhound starts git processes.

Every gateway part calls git through `run_git`, so encoding, environment
and error handling are identical everywhere (and easy to mock in tests).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .errors import GitCommandError, GitNotFoundError

# Environment applied to every git call.
_GIT_ENV_OVERRIDES = {
    "GIT_TERMINAL_PROMPT": "0",   # never block waiting for a password
    "GIT_OPTIONAL_LOCKS": "0",    # read-only commands must not take index.lock in the user's repo
    "LC_ALL": "C",                # stable, English error messages
}


def git_executable() -> str:
    exe = shutil.which("git")
    if exe is None:
        raise GitNotFoundError("git is not installed or not on PATH; Perfhound needs git >= 2.25")
    return exe


def run_git(
    repo: str | Path,
    *args: str,
    check: bool = True,
    ok_codes: tuple[int, ...] = (0,),
    input: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `git <args>` inside `repo` and return the completed process.

    Arguments are passed as a list (no shell), so refs with spaces or
    special characters cannot inject commands.

    I/O is done in BYTES and decoded here (UTF-8, invalid bytes replaced).
    Python's text mode would translate newlines on Windows: "\n" written to
    stdin becomes "\r\n" (git fetch --stdin then rejects every SHA), and
    "\r\n" / "\r" in git's output become "\n" (diffs of CRLF files change).
    """
    cmd = [git_executable(), *args]
    full_env = dict(os.environ)
    full_env.update(_GIT_ENV_OVERRIDES)
    full_env.update(env or {})
    raw = subprocess.run(
        cmd,
        cwd=str(repo),
        input=None if input is None else input.encode("utf-8"),
        env=full_env,
        capture_output=True,
    )
    proc = subprocess.CompletedProcess(
        raw.args, raw.returncode,
        raw.stdout.decode("utf-8", "replace"), raw.stderr.decode("utf-8", "replace"),
    )
    if check and proc.returncode not in ok_codes:
        raise GitCommandError(list(args), proc.returncode, proc.stderr)
    return proc


def git_out(repo: str | Path, *args: str) -> str:
    """Run git and return stdout with surrounding whitespace stripped."""
    return run_git(repo, *args).stdout.strip()


def run_git_bytes(repo: str | Path, *args: str, input: bytes | None = None) -> bytes:
    """Like run_git but binary-safe (file contents may be any encoding)."""
    cmd = [git_executable(), *args]
    env = dict(os.environ)
    env.update(_GIT_ENV_OVERRIDES)
    proc = subprocess.run(cmd, cwd=str(repo), env=env, input=input, capture_output=True)
    if proc.returncode != 0:
        raise GitCommandError(list(args), proc.returncode, proc.stderr.decode("utf-8", "replace"))
    return proc.stdout


def read_blobs(repo: str | Path, specs: list[str]) -> dict[str, bytes | None]:
    """Fetch many file contents in ONE `git cat-file --batch` process.

    specs are "<commit>:<path>" strings; value is None when missing.
    """
    specs = [s for s in dict.fromkeys(specs) if "\n" not in s]
    if not specs:
        return {}
    out = run_git_bytes(repo, "cat-file", "--batch", input=("\n".join(specs) + "\n").encode("utf-8"))
    result: dict[str, bytes | None] = {}
    pos = 0
    for spec in specs:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].decode("utf-8", "replace").split()
        pos = nl + 1
        if len(header) == 3 and header[1] == "blob":
            size = int(header[2])
            result[spec] = out[pos:pos + size]
            pos += size + 1  # content is followed by a newline
        elif len(header) == 3:          # tree/commit: skip its content too
            pos += int(header[2]) + 1
            result[spec] = None
        else:                            # "<spec> missing" / "ambiguous"
            result[spec] = None
    return result


# ---------------------------------------------------------------- partial clones

def is_partial_clone(repo: str | Path) -> bool:
    """True for `git clone --filter=...` repos, where file contents are fetched lazily."""
    proc = run_git(repo, "config", "--get-regexp", r"^(remote\..*\.promisor|extensions\.partialclone)$", check=False)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def ensure_objects(repo: str | Path, oids: list[str]) -> int:
    """Download missing objects of a partial clone in ONE fetch. Returns how many were missing.

    Without this, every `git log -p` / `cat-file` touching a missing blob makes
    git fetch it on its own - one network round trip per file (jsoup, 150
    commits: 134 s instead of ~5 s).
    """
    wanted = {o for o in oids if o and set(o) != {"0"}}
    if not wanted or not is_partial_clone(repo):
        return 0
    # Lists only objects that are present locally; never triggers a fetch.
    have = set(run_git(repo, "cat-file", "--batch-check=%(objectname)", "--batch-all-objects").stdout.split())
    missing = sorted(wanted - have)
    if not missing:
        return 0
    remote = run_git(repo, "config", "--get", "extensions.partialclone", check=False).stdout.strip() or "origin"
    run_git(
        repo,
        "-c", "fetch.negotiationAlgorithm=noop",
        "fetch", remote, "--no-tags", "--no-write-fetch-head", "--recurse-submodules=no",
        "--filter=blob:none", "--stdin",
        input="\n".join(missing) + "\n",
    )
    return len(missing)
