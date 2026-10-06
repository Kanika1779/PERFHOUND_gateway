"""Gateway exceptions. Callers can catch GatewayError for everything."""

from __future__ import annotations


class GatewayError(Exception):
    """Base class for every error raised by the gateway."""


class GitNotFoundError(GatewayError):
    """The `git` executable is not installed / not on PATH."""


class GitCommandError(GatewayError):
    """A git command failed unexpectedly."""

    def __init__(self, args: list[str], returncode: int, stderr: str) -> None:
        self.git_args = args
        self.returncode = returncode
        self.stderr = stderr.strip()
        super().__init__(f"git {' '.join(args)} failed (exit {returncode}): {self.stderr}")


class NotAGitRepoError(GatewayError):
    """The given path is not inside a git repository."""


class UnknownRefError(GatewayError):
    """A ref (tag / branch / SHA) does not resolve to a commit."""


class InvalidRangeError(GatewayError):
    """good..bad is not a usable range (empty, swapped, unrelated, too large)."""


class RangeWarning(UserWarning):
    """Range is usable but suspicious (very large, shallow clone, side-branch good)."""
