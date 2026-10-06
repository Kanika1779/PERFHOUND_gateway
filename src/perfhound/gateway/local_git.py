"""Local Git Provider (gateway part 4): commit details straight from .git.

Works fully offline. Rule 4 of the guide: git calls are BATCHED - however
many commits are asked for, exactly two git processes are started:

  1. metadata + changed files:  git log --no-walk --stdin --raw --numstat -z
  2. diffs:                     git log --no-walk --stdin -p

SHAs are fed on stdin, so the same code works for a whole range or for the
handful of commits the cache (Step 7) has not seen yet.

Merge commits are diffed against their FIRST parent, i.e. "what did merging
this branch change on the main line". Requires git >= 2.31
(--diff-merges=first-parent).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .errors import GatewayError
from .gitcmd import ensure_objects, is_partial_clone, run_git
from .models import CandidateCommit, FileChange
from .range import CommitRange

DEFAULT_MAX_DIFF_LINES = 400
DEFAULT_MAX_DIFF_CHARS = 40_000   # guards against one-line minified files

_HEADER_RE = re.compile(r"^\x1e([0-9a-f]{40})$")
_DIFF_MARKER = "\x1e\x1ePERFHOUND-COMMIT "   # cannot start a real patch line

# Flags that neutralize user git config that would change the output format.
_COMMON_LOG_FLAGS = (
    "--no-walk=unsorted",   # exactly the SHAs we pass, in our order
    "--stdin",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--no-show-signature",
    "--encoding=UTF-8",
    "-M",                   # detect renames
    "--diff-merges=first-parent",
)


@dataclass(frozen=True)
class RawCommitInfo:
    """Everything from the metadata pass (before diff / analysis)."""

    sha: str
    parents: tuple[str, ...]
    author: str
    timestamp: datetime
    message: str
    files: tuple[FileChange, ...]


def parse_git_date(iso: str, epoch: str | int | None = None) -> datetime:
    """Git ISO-8601 date (%aI / %cI) -> timezone-aware datetime.

    Newer git prints UTC as "Z" (older: "+00:00"); Python 3.10 cannot read "Z" - handled.
    Old histories contain broken time zones (requests, 2011: "+051800", printed by git as
    "+518:00"), which Python cannot parse. Then the moment comes from the epoch seconds
    (%at / %ct, always correct), in UTC.
    """
    iso = iso.strip()
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.utcoffset() is not None and (epoch is None or int(dt.timestamp()) == int(epoch)):
            return dt
    except (ValueError, OverflowError):
        pass
    if epoch is None or not str(epoch).strip():
        raise GatewayError(f"cannot parse git date {iso!r}")
    return datetime.fromtimestamp(int(epoch), timezone.utc)


# --------------------------------------------------------------------------
# pass 1: metadata + files
# --------------------------------------------------------------------------

def _parse_raw_and_numstat(tokens: list[str]) -> tuple[FileChange, ...]:
    """Parse the NUL-separated --raw + --numstat tokens of ONE commit."""
    raw_entries: list[tuple[str, str, str | None]] = []   # (status, path, old_path)
    counts: dict[str, tuple[int | None, int | None]] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i].lstrip("\n")
        if not tok:
            i += 1
            continue
        if tok.startswith(":"):
            status = tok.split()[-1]
            letter = status[0]
            if letter in ("R", "C"):
                old, new = tokens[i + 1], tokens[i + 2]
                raw_entries.append((letter, new, old))
                i += 3
            else:
                raw_entries.append((letter, tokens[i + 1], None))
                i += 2
            continue
        # numstat: "add\tdel\tpath"  or  "add\tdel\t" + old + new  (rename)
        add_s, del_s, path = tok.split("\t", 2)
        if path == "":
            path = tokens[i + 2]
            i += 3
        else:
            i += 1
        counts[path] = (None if add_s == "-" else int(add_s), None if del_s == "-" else int(del_s))

    files = []
    for letter, path, old in raw_entries:
        if letter not in ("A", "M", "D", "R", "C", "T"):
            letter = "M"   # 'U'nmerged / 'X' unknown cannot occur in history; be lenient
        add, dele = counts.get(path, (0, 0))
        files.append(FileChange(path=path, status=letter, additions=add, deletions=dele, old_path=old))
    return tuple(files)


def read_commit_info(repo: str | Path, shas: Sequence[str]) -> list[RawCommitInfo]:
    """Metadata + changed files for `shas`, in the same order, ONE git process."""
    if not shas:
        return []
    proc = run_git(
        repo,
        "log",
        *_COMMON_LOG_FLAGS,
        "--raw",
        "--numstat",
        "-z",
        "--format=%x1e%H%x00%P%x00%an%x00%aI%x00%at%x00%B%x00",
        input="\n".join(shas) + "\n",
    )
    tokens = proc.stdout.split("\x00")
    infos: list[RawCommitInfo] = []
    i = 0
    n = len(tokens)
    while i < n:
        m = _HEADER_RE.match(tokens[i].lstrip("\n"))
        if not m:
            i += 1
            continue
        sha = m.group(1)
        parents = tuple(tokens[i + 1].split())
        author = tokens[i + 2]
        timestamp = parse_git_date(tokens[i + 3], tokens[i + 4])
        message = tokens[i + 5].rstrip("\n")
        i += 6
        j = i
        while j < n and not _HEADER_RE.match(tokens[j].lstrip("\n")):
            j += 1
        files = _parse_raw_and_numstat(tokens[i:j])
        infos.append(RawCommitInfo(sha, parents, author, timestamp, message, files))
        i = j

    got = [info.sha for info in infos]
    if got != list(shas):
        raise GatewayError(f"git log returned {len(got)} commits, expected {len(shas)} (order or SHAs differ)")
    return infos


# --------------------------------------------------------------------------
# pass 0 (partial clones only): prefetch every blob the other passes need
# --------------------------------------------------------------------------

def changed_blob_ids(repo: str | Path, shas: Sequence[str]) -> list[str]:
    """Old and new blob ids of every file changed by `shas` (vs first parent).

    Uses --raw WITHOUT rename detection: that needs only commits and trees,
    which a blob:none clone has, so this call itself never downloads anything.
    """
    if not shas:
        return []
    out = run_git(
        repo, "log", "--no-walk=unsorted", "--stdin", "--no-renames", "--raw", "--no-abbrev",
        "--diff-merges=first-parent", "--format=", input="\n".join(shas) + "\n",
    ).stdout
    oids: set[str] = set()
    for line in out.splitlines():
        if line.startswith(":"):
            parts = line.split()
            oids.update(parts[2:4])
    return sorted(oids)


def prefetch_blobs(repo: str | Path, shas: Sequence[str]) -> int:
    """Batch-download the blobs `shas` touch, if `repo` is a partial clone."""
    if not shas or not is_partial_clone(repo):
        return 0
    return ensure_objects(repo, changed_blob_ids(repo, shas))


# --------------------------------------------------------------------------
# pass 2: diffs
# --------------------------------------------------------------------------

def _split_lines(text: str) -> list[str]:
    """Split on "\n" ONLY, keeping the newline.

    str.splitlines() also splits on \x0b, \x0c, \x1c-\x1e, \x85, \u2028 ...
    which can appear inside source files and would corrupt diffs.
    """
    lines = text.split("\n")
    out = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def truncate_diff(diff: str, max_lines: int, max_chars: int) -> tuple[str, bool]:
    """Cut a diff to the size limits. Returns (diff, was_truncated)."""
    truncated = False
    lines = _split_lines(diff)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    out = "".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars]
        truncated = True
    return out, truncated


def read_diffs(
    repo: str | Path,
    shas: Sequence[str],
    *,
    max_lines: int = DEFAULT_MAX_DIFF_LINES,
    max_chars: int = DEFAULT_MAX_DIFF_CHARS,
) -> dict[str, tuple[str, bool]]:
    """sha -> (possibly truncated diff vs first parent, truncated?) in ONE git process."""
    if not shas:
        return {}
    proc = run_git(
        repo,
        "log",
        *_COMMON_LOG_FLAGS,
        "-p",
        "--format=" + _DIFF_MARKER.replace("\x1e", "%x1e") + "%H",
        input="\n".join(shas) + "\n",
    )
    result: dict[str, tuple[str, bool]] = {}
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is not None:
            text = "".join(buf).strip("\n")
            text = text + "\n" if text else ""
            result[current] = truncate_diff(text, max_lines, max_chars)

    for line in _split_lines(proc.stdout):
        if line.startswith(_DIFF_MARKER):
            flush()
            current = line[len(_DIFF_MARKER):].strip()
            buf = []
        else:
            buf.append(line)
    flush()

    for sha in shas:   # commits with no textual changes still get an entry
        result.setdefault(sha, ("", False))
    return result


# --------------------------------------------------------------------------
# provider
# --------------------------------------------------------------------------

class LocalGitProvider:
    """Builds CandidateCommit records from the local repository only."""

    def __init__(
        self,
        repo: str | Path,
        *,
        max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
        max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
    ) -> None:
        self.repo = Path(repo)
        self.max_diff_lines = max_diff_lines
        self.max_diff_chars = max_diff_chars

    def get_commits(self, commit_range: CommitRange, shas: Iterable[str] | None = None) -> list[CandidateCommit]:
        """CandidateCommits for `shas` (default: the whole range), oldest first.

        Positions always come from `commit_range`, so a subset keeps the
        same positions it would have in the full list.
        """
        wanted = list(commit_range.shas if shas is None else shas)
        self.prefetched = prefetch_blobs(self.repo, wanted)
        infos = read_commit_info(self.repo, wanted)
        diffs = read_diffs(self.repo, wanted, max_lines=self.max_diff_lines, max_chars=self.max_diff_chars)
        out = []
        for info in infos:
            diff, truncated = diffs[info.sha]
            out.append(
                CandidateCommit(
                    sha=info.sha,
                    parent=info.parents[0],
                    position=commit_range.position_of(info.sha),
                    message=info.message,
                    author=info.author,
                    timestamp=info.timestamp,
                    files=info.files,
                    diff=diff,
                    diff_truncated=truncated,
                    pr=None,
                    source="local",
                )
            )
        return out
