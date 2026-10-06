"""Understanding GitHub links - pure text processing, no network.

    parse_repo("https://github.com/sympy/sympy")          -> RepoRef("sympy", "sympy")
    parse_repo("git@github.com:sympy/sympy.git") / "sympy/sympy"
    parse_pr("https://github.com/o/r/pull/12/files")      -> (RepoRef, 12)        also "o/r#12"
    parse_commit("https://github.com/o/r/commit/9ccb32e") -> (RepoRef, "9ccb32e") also "o/r@9ccb32e"
    parse_compare("https://github.com/o/r/compare/v1.2...main") -> (RepoRef, "v1.2", "main")
        a compare link IS a Perfhound case range: good = left side, bad = right side
    linked_issues("Fixes #45, closes o/r#7", repo)        -> [45, 7]   (other repos ignored)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import InvalidGitHubLink

_NAME = r"[A-Za-z0-9_.-]+"
_HTTP = re.compile(rf"^https?://(?:www\.)?github\.com/({_NAME})/({_NAME}?)(?:\.git)?(?:/(.*))?$")
_SSH = re.compile(rf"^(?:ssh://)?git@github\.com[:/]({_NAME})/({_NAME}?)(?:\.git)?/?$")
_SHORT = re.compile(rf"^({_NAME})/({_NAME})$")
_ISSUE_KEYWORD = re.compile(
    rf"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b:?\s+"
    rf"(?:https?://github\.com/({_NAME})/({_NAME})/issues/(\d+)|({_NAME})/({_NAME})#(\d+)|#(\d+))",
    re.IGNORECASE)


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def url(self) -> str:
        return f"https://github.com/{self.slug}"


def _split(text: str) -> tuple[RepoRef, str]:
    text = text.strip().rstrip("/")
    m = _HTTP.match(text)
    if m and m.group(2):
        return RepoRef(m.group(1), m.group(2)), (m.group(3) or "")
    m = _SSH.match(text)
    if m and m.group(2):
        return RepoRef(m.group(1), m.group(2)), ""
    raise InvalidGitHubLink(f"not a GitHub link: {text!r}")


def parse_repo(text: str) -> RepoRef:
    text = text.strip()
    m = _SHORT.match(text)
    if m and not text.startswith(("http", "git@")):
        return RepoRef(m.group(1), m.group(2).removesuffix(".git"))
    return _split(text)[0]


def is_github(text: str) -> bool:
    try:
        parse_repo(text)
        return True
    except InvalidGitHubLink:
        return False


def parse_pr(text: str) -> tuple[RepoRef, int]:
    m = re.match(rf"^({_NAME})/({_NAME})#(\d+)$", text.strip())
    if m:
        return RepoRef(m.group(1), m.group(2)), int(m.group(3))
    repo, rest = _split(text)
    m = re.match(r"^pulls?/(\d+)", rest)
    if not m:
        raise InvalidGitHubLink(f"not a pull-request link: {text!r}")
    return repo, int(m.group(1))


def parse_commit(text: str) -> tuple[RepoRef, str]:
    m = re.match(rf"^({_NAME})/({_NAME})@([0-9a-fA-F]{{7,40}})$", text.strip())
    if m:
        return RepoRef(m.group(1), m.group(2)), m.group(3).lower()
    repo, rest = _split(text)
    m = re.match(r"^commits?/([0-9a-fA-F]{7,40})", rest)
    if not m:
        raise InvalidGitHubLink(f"not a commit link: {text!r}")
    return repo, m.group(1).lower()


def parse_compare(text: str) -> tuple[RepoRef, str, str]:
    repo, rest = _split(text)
    m = re.match(r"^compare/(.+?)\.\.\.?(.+)$", rest)
    if not m:
        raise InvalidGitHubLink(f"not a compare link (…/compare/GOOD...BAD): {text!r}")
    return repo, m.group(1), m.group(2)


def linked_issues(text: str, repo: RepoRef) -> list[int]:
    """Issues a PR says it closes, with GitHub's own keywords. Issues in OTHER repos are ignored."""
    found: list[int] = []
    for m in _ISSUE_KEYWORD.finditer(text or ""):
        if m.group(3):
            owner, name, num = m.group(1), m.group(2), m.group(3)
        elif m.group(6):
            owner, name, num = m.group(4), m.group(5), m.group(6)
        else:
            owner, name, num = repo.owner, repo.name, m.group(7)
        if owner.lower() == repo.owner.lower() and name.lower() == repo.name.lower() and int(num) not in found:
            found.append(int(num))
    return found
