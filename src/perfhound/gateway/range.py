"""Range Resolver (gateway part 2): good..bad -> ordered list of commit SHAs.

Uses the FIRST-PARENT history of `bad`: the chain of commits on the main
line. Commits that came in through a merge are represented by their merge
commit. That gives a linear sequence, which is what bisection / SPRT search
needs (every position is a buildable state of the main branch).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from .errors import InvalidRangeError, NotAGitRepoError, RangeWarning, UnknownRefError
from .gitcmd import git_out, run_git

DEFAULT_WARN_THRESHOLD = 1000


@dataclass(frozen=True)
class CommitRange:
    """Result of resolving good..bad.

    shas: first-parent commits strictly after `good` up to and including
    `bad`, oldest first. shas[i] has position i in CandidateCommit.
    """

    repo: Path
    good_ref: str
    bad_ref: str
    good: str
    bad: str
    shas: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.shas)

    def __iter__(self):
        return iter(self.shas)

    def position_of(self, sha: str) -> int:
        return self.shas.index(sha)


def find_repo_root(path: str | Path) -> Path:
    """Return the top-level folder of the repo containing `path`."""
    path = Path(path)
    if not path.is_dir():
        raise NotAGitRepoError(f"{path} is not a folder")
    proc = run_git(path, "rev-parse", "--show-toplevel", check=False)
    if proc.returncode != 0:
        raise NotAGitRepoError(f"{path} is not inside a git repository")
    return Path(proc.stdout.strip())


def resolve_ref(repo: str | Path, ref: str) -> str:
    """Turn a tag / branch / short SHA / HEAD~3 into a full commit SHA."""
    if not ref or not ref.strip():
        raise UnknownRefError("empty ref")
    if ref.startswith("-"):
        # would be parsed by git as an option
        raise UnknownRefError(f"invalid ref {ref!r}")
    proc = run_git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise UnknownRefError(f"{ref!r} is not a commit, tag or branch in this repository")
    return proc.stdout.strip()


def is_ancestor(repo: str | Path, older: str, newer: str) -> bool:
    proc = run_git(repo, "merge-base", "--is-ancestor", older, newer, ok_codes=(0, 1))
    return proc.returncode == 0


def resolve_range(
    repo: str | Path,
    good: str,
    bad: str,
    *,
    warn_threshold: int = DEFAULT_WARN_THRESHOLD,
    max_commits: int | None = None,
) -> CommitRange:
    """Resolve good..bad into an ordered CommitRange.

    Raises InvalidRangeError when the range cannot be searched, and emits a
    RangeWarning when it can be searched but the result may be misleading.
    """
    root = find_repo_root(repo)
    good_sha = resolve_ref(root, good)
    bad_sha = resolve_ref(root, bad)

    if good_sha == bad_sha:
        raise InvalidRangeError(f"good ({good}) and bad ({bad}) are the same commit - nothing to search")

    if not is_ancestor(root, good_sha, bad_sha):
        if is_ancestor(root, bad_sha, good_sha):
            raise InvalidRangeError(
                f"good ({good}) is newer than bad ({bad}). Did you swap them? "
                f"Usage: good = last fast commit, bad = first slow commit."
            )
        raise InvalidRangeError(
            f"good ({good}) is not an ancestor of bad ({bad}); they are on unrelated branches"
        )

    if git_out(root, "rev-parse", "--is-shallow-repository") == "true":
        warnings.warn(
            "repository is a shallow clone; history may be incomplete (run `git fetch --unshallow`)",
            RangeWarning,
            stacklevel=2,
        )

    out = git_out(root, "rev-list", "--first-parent", "--reverse", f"{good_sha}..{bad_sha}")
    shas = tuple(out.split())

    if not shas:  # cannot really happen after the ancestor check, but be explicit
        raise InvalidRangeError(f"no commits between {good} and {bad}")

    first_parent = git_out(root, "rev-parse", f"{shas[0]}^")
    if first_parent != good_sha:
        warnings.warn(
            f"good ({good}) is not on the main (first-parent) line of bad ({bad}); "
            f"the search starts from {first_parent[:7]}, which does not contain good's changes. "
            f"Pick a good commit on the main branch for reliable results.",
            RangeWarning,
            stacklevel=2,
        )

    if max_commits is not None and len(shas) > max_commits:
        raise InvalidRangeError(
            f"range has {len(shas)} commits, more than max_commits={max_commits}; choose a narrower good/bad pair"
        )
    if len(shas) > warn_threshold:
        warnings.warn(
            f"range has {len(shas)} commits; this will take a long time. Consider a narrower good/bad pair.",
            RangeWarning,
            stacklevel=2,
        )

    return CommitRange(repo=root, good_ref=good, bad_ref=bad, good=good_sha, bad=bad_sha, shas=shas)
