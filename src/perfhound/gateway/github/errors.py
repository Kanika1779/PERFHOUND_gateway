"""GitHub errors: one name per failure, each message says what to do next."""

from __future__ import annotations

from datetime import datetime, timezone

from ..errors import GatewayError


class GitHubError(GatewayError):
    """Base class for everything that can go wrong talking to GitHub."""


class GitHubAuthError(GitHubError):
    """401: the token is missing, wrong, expired or revoked."""


class GitHubNotFound(GitHubError):
    """404. GitHub also answers 404 (not 403) for PRIVATE repos the token cannot see."""


class GitHubRateLimited(GitHubError):
    """The hourly request allowance is used up."""

    def __init__(self, message: str, reset_at: int | None = None) -> None:
        self.reset_at = reset_at
        if reset_at:
            when = datetime.fromtimestamp(reset_at, tz=timezone.utc).astimezone().strftime("%H:%M")
            message += f" - the allowance resets at {when}"
        super().__init__(message)


class GitHubRequestFailed(GitHubError):
    """Server errors / network problems that survived the retries, or other unexpected answers."""


class InvalidGitHubLink(GitHubError):
    """Text that is not a GitHub repo / PR / commit / compare link."""
