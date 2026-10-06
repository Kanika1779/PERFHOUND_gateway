"""A small GitHub REST client (standard library only - same style as our Gemini client).

Why hand-written instead of PyGithub: Perfhound needs 3 endpoints. A client the team can read
and explain beats a library whose retries and caching happen out of sight, and it COUNTS every
request so each run reports how much of the hourly allowance it used.

    client = GitHubClient(token)
    resp = client.get("/repos/sympy/sympy/commits/<sha>/pulls", etag=previous_etag)
    resp.status == 304  -> unchanged since last time (free: does not count against the limit)

Rules:
  * headers: Authorization: Bearer <token>, X-GitHub-Api-Version, Accept, User-Agent
  * 5xx and dropped connections are retried (1 s, 2 s, 4 s); 4xx never (same request, same answer)
  * every failure becomes a named error (errors.py) with advice
  * pagination follows the Link: rel="next" header, with a page limit
  * the transport is injectable, so tests use a fake GitHub (no network, no token)
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import GitHubAuthError, GitHubNotFound, GitHubRateLimited, GitHubRequestFailed

API = "https://api.github.com"
API_VERSION = "2022-11-28"
USER_AGENT = "perfhound-gateway"

# transport(method, url, headers, timeout) -> (status, headers, body); raises OSError on network failure
Transport = Callable[[str, str, dict, float], "tuple[int, dict, bytes]"]


def urllib_transport(method: str, url: str, headers: dict, timeout: float,
                     body: bytes | None = None) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers.items()) if e.headers else {}, e.read()


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    data: Any = None
    etag: str | None = None

    @property
    def not_modified(self) -> bool:
        return self.status == 304


@dataclass
class GitHubClient:
    token: str | None = None
    transport: Transport = urllib_transport
    base_url: str = API
    timeout: float = 30.0
    max_retries: int = 3
    sleep: Callable[[float], None] = time.sleep
    requests: int = 0                       # every request sent (incl. retries)
    rate_remaining: int | None = None
    rate_limit: int | None = None
    rate_reset: int | None = None
    _log: list = field(default_factory=list, repr=False)

    def get(self, path: str, params: dict | None = None, *, etag: str | None = None,
            accept: str = "application/vnd.github+json") -> Response:
        url = path if path.startswith("http") else self.base_url + path
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        headers = {"Accept": accept, "X-GitHub-Api-Version": API_VERSION, "User-Agent": USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if etag:
            headers["If-None-Match"] = etag
        return self._send("GET", url, headers, None, path)

    def graphql(self, query: str, variables: dict | None = None) -> Response:
        """POST /graphql. GitHub's GraphQL API needs a token (no anonymous access).
        Errors inside a 200 answer are left in resp.data["errors"] for the caller."""
        headers = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
        return self._send("POST", self.base_url + "/graphql", headers, payload, "/graphql")

    def _send(self, method: str, url: str, headers: dict, payload: bytes | None, path: str) -> Response:
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                if payload is None:
                    status, raw_headers, body = self.transport(method, url, headers, self.timeout)
                else:
                    status, raw_headers, body = self.transport(method, url, headers, self.timeout, body=payload)
            except OSError as e:                                 # network down, DNS, reset, timeout
                if attempt < self.max_retries:
                    self.requests += 1
                    self.sleep(delay)
                    delay *= 2
                    continue
                raise GitHubRequestFailed(f"cannot reach GitHub ({e}) - check the internet connection") from None
            self.requests += 1
            h = {k.lower(): v for k, v in raw_headers.items()}
            self._note_rate(h)
            if status in (500, 502, 503, 504) and attempt < self.max_retries:
                self.sleep(delay)
                delay *= 2
                continue
            return self._handle(status, h, body, path)
        raise GitHubRequestFailed("unreachable")

    def get_json(self, path: str, params: dict | None = None) -> Any:
        return self.get(path, params).data

    def paginate(self, path: str, params: dict | None = None, *, max_pages: int = 10) -> list:
        items: list = []
        url, p = path, {**(params or {}), "per_page": 100}
        for _ in range(max_pages):
            resp = self.get(url, p)
            items.extend(resp.data or [])
            nxt = _next_link(resp.headers.get("link", ""))
            if not nxt:
                break
            url, p = nxt, None
        return items

    # -- internals ---------------------------------------------------------------------------
    def _note_rate(self, h: dict) -> None:
        for attr, key in (("rate_remaining", "x-ratelimit-remaining"), ("rate_limit", "x-ratelimit-limit"),
                          ("rate_reset", "x-ratelimit-reset")):
            if h.get(key, "").isdigit():
                setattr(self, attr, int(h[key]))

    def _handle(self, status: int, h: dict, body: bytes, path: str) -> Response:
        if status == 304:
            return Response(304, h, None, h.get("etag"))
        text = body.decode("utf-8", "replace") if body else ""
        if 200 <= status < 300:
            try:
                data = json.loads(text) if text else None
            except ValueError:
                data = text
            return Response(status, h, data, h.get("etag"))
        message = _message(text)
        if status == 401:
            raise GitHubAuthError("GitHub rejected the token (wrong, expired or revoked) - run "
                                  "`python -m perfhound.gateway login` again")
        if status == 404:
            raise GitHubNotFound(f"{path}: not found - check the link, or the token cannot see this private repository")
        if status in (403, 429) and (h.get("x-ratelimit-remaining") == "0" or "rate limit" in message.lower()
                                     or "retry-after" in h):
            reset = int(h["x-ratelimit-reset"]) if h.get("x-ratelimit-reset", "").isdigit() else None
            if reset is None and h.get("retry-after", "").isdigit():
                reset = int(time.time()) + int(h["retry-after"])
            raise GitHubRateLimited("GitHub request limit reached" + ("" if self.token else
                                    " (no token: 60/hour - `python -m perfhound.gateway login` raises it to 5,000)"), reset)
        if status >= 500:
            raise GitHubRequestFailed(f"GitHub server error {status} on {path} (retried {self.max_retries} times)")
        raise GitHubRequestFailed(f"GitHub answered {status} on {path}: {message[:200]}")


def _message(text: str) -> str:
    try:
        return str(json.loads(text).get("message", text))
    except (ValueError, AttributeError):
        return text


def _next_link(link: str) -> str | None:
    m = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
    return m.group(1) if m else None
