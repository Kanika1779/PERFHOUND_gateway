"""Cache (gateway part 8a): SQLite store so a commit is processed only once.

Rule 3 of the guide: a commit SHA is a hash of its content, so data derived
ONLY from the commit never goes stale and is stored forever. Two caveats:

* Some values depend on more than the SHA - e.g. the diff depends on the
  truncation limit, function lists depend on the analyzer version. Those
  settings are part of the `kind` key, so changing them is a cache miss,
  never a wrong answer.
* `position` depends on the requested range, so it is NOT cached; the
  gateway sets it from the range on every request.

GitHub PR data can change (title edits, labels), so it is stored with an
ETag and an expiry (default 24h) - used by the GitHub provider (Step 9).

The cache is an optimization: if the database is unusable (corrupt, locked,
read-only disk) a warning is emitted and the gateway keeps working without it.

Location: $PERFHOUND_CACHE_DIR/cache.db, else ~/.perfhound/cache.db.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

CACHE_SCHEMA_VERSION = 1
DEFAULT_PR_TTL_SECONDS = 24 * 3600


class CacheWarning(UserWarning):
    """The cache could not be used; the gateway continues without it."""


def default_cache_path() -> Path:
    base = os.environ.get("PERFHOUND_CACHE_DIR")
    return (Path(base) if base else Path.home() / ".perfhound") / "cache.db"


@dataclass(frozen=True)
class CachedValue:
    payload: Any
    etag: str | None
    fetched_at: float


class Cache:
    """Key = (repo_id, sha, kind). Values are JSON."""

    def __init__(self, path: str | Path | None = None, *, pr_ttl: float = DEFAULT_PR_TTL_SECONDS) -> None:
        self.path = Path(path) if path else default_cache_path()
        self.pr_ttl = pr_ttl
        self._conn: sqlite3.Connection | None = None
        self.disabled = False
        try:
            self._open()
        except (sqlite3.Error, OSError) as exc:
            self._disable(f"cannot open cache at {self.path}: {exc}")

    # -- setup ---------------------------------------------------------------

    def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")       # readers don't block a writer
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is not None and int(row[0]) != CACHE_SCHEMA_VERSION:
                conn.execute("DROP TABLE IF EXISTS entries")   # old layout: start fresh
            conn.execute(
                """CREATE TABLE IF NOT EXISTS entries (
                       repo_id    TEXT NOT NULL,
                       sha        TEXT NOT NULL,
                       kind       TEXT NOT NULL,
                       payload    TEXT NOT NULL,
                       etag       TEXT,
                       fetched_at REAL NOT NULL,
                       PRIMARY KEY (repo_id, sha, kind))"""
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(CACHE_SCHEMA_VERSION),)
            )
            conn.commit()
        except sqlite3.DatabaseError:
            conn.close()
            raise
        self._conn = conn

    def _disable(self, reason: str) -> None:
        warnings.warn(f"{reason}; continuing without cache", CacheWarning, stacklevel=3)
        self.disabled = True
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
        self._conn = None

    # -- commit-derived data (never expires) ---------------------------------

    def get_many(self, repo_id: str, shas: Iterable[str], kind: str) -> dict[str, Any]:
        """sha -> payload for every sha that is cached."""
        shas = list(dict.fromkeys(shas))
        if self._conn is None or not shas:
            return {}
        out: dict[str, Any] = {}
        try:
            for i in range(0, len(shas), 500):   # SQLite limits the number of '?' parameters
                chunk = shas[i:i + 500]
                q = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT sha, payload FROM entries WHERE repo_id=? AND kind=? AND sha IN ({q})",
                    (repo_id, kind, *chunk),
                ).fetchall()
                for sha, payload in rows:
                    out[sha] = json.loads(payload)
        except (sqlite3.Error, ValueError) as exc:
            self._disable(f"cache read failed: {exc}")
            return {}
        return out

    def put_many(self, repo_id: str, items: dict[str, Any], kind: str) -> None:
        if self._conn is None or not items:
            return
        now = time.time()
        try:
            with self._conn:   # one transaction
                self._conn.executemany(
                    "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, NULL, ?)",
                    [(repo_id, sha, kind, json.dumps(p, ensure_ascii=False), now) for sha, p in items.items()],
                )
        except sqlite3.Error as exc:
            self._disable(f"cache write failed: {exc}")

    # -- data that can change (GitHub) ---------------------------------------

    def get_volatile(self, repo_id: str, key: str, kind: str) -> tuple[CachedValue | None, bool]:
        """Return (value, is_fresh). A stale value is still returned so its
        ETag can be sent to GitHub (a 304 reply is free)."""
        if self._conn is None:
            return None, False
        try:
            row = self._conn.execute(
                "SELECT payload, etag, fetched_at FROM entries WHERE repo_id=? AND sha=? AND kind=?",
                (repo_id, key, kind),
            ).fetchone()
        except sqlite3.Error as exc:
            self._disable(f"cache read failed: {exc}")
            return None, False
        if row is None:
            return None, False
        value = CachedValue(json.loads(row[0]), row[1], row[2])
        return value, (time.time() - value.fetched_at) < self.pr_ttl

    def put_volatile(self, repo_id: str, key: str, kind: str, payload: Any, etag: str | None) -> None:
        if self._conn is None:
            return
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, ?)",
                    (repo_id, key, kind, json.dumps(payload, ensure_ascii=False), etag, time.time()),
                )
        except sqlite3.Error as exc:
            self._disable(f"cache write failed: {exc}")

    # -- housekeeping --------------------------------------------------------

    def count(self, repo_id: str | None = None) -> int:
        if self._conn is None:
            return 0
        if repo_id is None:
            return self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM entries WHERE repo_id=?", (repo_id,)).fetchone()[0]

    def clear(self, repo_id: str | None = None) -> None:
        if self._conn is None:
            return
        with self._conn:
            if repo_id is None:
                self._conn.execute("DELETE FROM entries")
            else:
                self._conn.execute("DELETE FROM entries WHERE repo_id=?", (repo_id,))

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
