"""GitHub part of the gateway: token handling, a small REST client, link parsing, and the
provider that attaches the pull request behind each candidate commit. Optional: everything
in Perfhound works without it (offline, no token) - GitHub only ADDS PR titles/descriptions."""

from .auth import delete_token, find_token, mask, save_token
from .client import GitHubClient, Response
from .errors import (GitHubAuthError, GitHubError, GitHubNotFound, GitHubRateLimited, GitHubRequestFailed,
                     InvalidGitHubLink)
from .links import RepoRef, is_github, linked_issues, parse_commit, parse_compare, parse_pr, parse_repo
from .graphql import GitHubGraphQLProvider
from .provider import GitHubPRProvider, GitHubStats

__all__ = ["delete_token", "find_token", "mask", "save_token", "GitHubClient", "Response", "GitHubAuthError",
           "GitHubError", "GitHubNotFound", "GitHubRateLimited", "GitHubRequestFailed", "InvalidGitHubLink",
           "RepoRef", "is_github", "linked_issues", "parse_commit", "parse_compare", "parse_pr", "parse_repo",
           "GitHubPRProvider", "GitHubGraphQLProvider", "GitHubStats"]
