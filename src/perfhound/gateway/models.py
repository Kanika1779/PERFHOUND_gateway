"""Gateway output contract: CandidateCommit and its parts.

Every module after the gateway works ONLY with these records. Change this
file carefully: a field change breaks every consumer.

Design notes
------------
* All records are frozen (immutable). Collections are tuples, not lists,
  so a record can never be modified after the gateway hands it out and it
  stays hashable (usable as a dict key / in sets).
* Timestamps must be timezone-aware. Git stores an offset per commit; a
  naive datetime would silently lose it.
* JSON round-trip is exact: CandidateCommit.from_json(c.to_json()) == c.
  The snapshot store (Step 8) relies on this.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SCHEMA_VERSION = 1

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# How a file changed in a commit (subset of git's --name-status letters).
FILE_STATUSES = frozenset({"A", "M", "D", "R", "C", "T"})

# Where a CandidateCommit's data came from.
SOURCES = frozenset({"local", "local+github", "snapshot"})


def _check_sha(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA_RE.match(value):
        raise ValueError(f"{name} must be a full 40-char lowercase hex SHA, got {value!r}")


@dataclass(frozen=True)
class FileChange:
    """One file touched by a commit.

    additions / deletions are None for binary files (git prints '-').
    old_path is set only for renames / copies.
    """

    path: str
    status: str = "M"
    additions: int | None = 0
    deletions: int | None = 0
    old_path: str | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("FileChange.path must not be empty")
        if self.status not in FILE_STATUSES:
            raise ValueError(f"FileChange.status must be one of {sorted(FILE_STATUSES)}, got {self.status!r}")
        for name in ("additions", "deletions"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"FileChange.{name} must be >= 0 or None")
        if self.status in ("R", "C") and not self.old_path:
            raise ValueError("rename/copy FileChange needs old_path")

    @property
    def is_binary(self) -> bool:
        return self.additions is None or self.deletions is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
            "old_path": self.old_path,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileChange":
        return cls(
            path=d["path"],
            status=d.get("status", "M"),
            additions=d.get("additions", 0),
            deletions=d.get("deletions", 0),
            old_path=d.get("old_path"),
        )


@dataclass(frozen=True)
class PRInfo:
    """Pull-request metadata from GitHub (things git itself does not store)."""

    number: int
    title: str
    body: str = ""
    labels: tuple[str, ...] = ()
    url: str = ""

    def __post_init__(self) -> None:
        if self.number <= 0:
            raise ValueError("PRInfo.number must be positive")
        object.__setattr__(self, "labels", tuple(self.labels))

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "labels": list(self.labels),
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PRInfo":
        return cls(
            number=d["number"],
            title=d["title"],
            body=d.get("body", ""),
            labels=tuple(d.get("labels", ())),
            url=d.get("url", ""),
        )


@dataclass(frozen=True)
class CandidateCommit:
    """One commit between `good` and `bad`, normalized for the rest of the system.

    position: 0 = first commit after `good`, increasing towards `bad`.
    changed/added/deleted_functions: fully-qualified names such as
        "sympy.core.basic.Basic.__hash__" (filled by the Code Analyzer, Step 5).
    unanalyzed_files: Python files the analyzer could NOT parse (syntax error,
        too large...). Non-empty means the function lists may be incomplete -
        "no functions changed" and "unknown" are different things.
    pr: None when there is no internet / no token / repo not on GitHub.
        Consumers MUST handle None.
    """

    sha: str
    parent: str
    position: int
    message: str
    author: str
    timestamp: datetime
    files: tuple[FileChange, ...] = ()
    changed_functions: tuple[str, ...] = ()
    added_functions: tuple[str, ...] = ()
    deleted_functions: tuple[str, ...] = ()
    unanalyzed_files: tuple[str, ...] = ()
    diff: str = ""
    diff_truncated: bool = False
    pr: PRInfo | None = None
    source: str = "local"

    def __post_init__(self) -> None:
        _check_sha(self.sha, "sha")
        _check_sha(self.parent, "parent")
        if self.position < 0:
            raise ValueError("position must be >= 0")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {sorted(SOURCES)}, got {self.source!r}")
        # Accept lists from callers but always store tuples (immutability).
        for name in ("files", "changed_functions", "added_functions", "deleted_functions", "unanalyzed_files"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def short_sha(self) -> str:
        return self.sha[:7]

    @property
    def subject(self) -> str:
        """First line of the commit message."""
        return self.message.split("\n", 1)[0]

    # ---- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sha": self.sha,
            "parent": self.parent,
            "position": self.position,
            "message": self.message,
            "author": self.author,
            "timestamp": self.timestamp.isoformat(),
            "files": [f.to_dict() for f in self.files],
            "changed_functions": list(self.changed_functions),
            "added_functions": list(self.added_functions),
            "deleted_functions": list(self.deleted_functions),
            "unanalyzed_files": list(self.unanalyzed_files),
            "diff": self.diff,
            "diff_truncated": self.diff_truncated,
            "pr": self.pr.to_dict() if self.pr else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CandidateCommit":
        version = d.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported CandidateCommit schema_version {version} (expected {SCHEMA_VERSION})")
        return cls(
            sha=d["sha"],
            parent=d["parent"],
            position=d["position"],
            message=d["message"],
            author=d["author"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            files=tuple(FileChange.from_dict(f) for f in d.get("files", ())),
            changed_functions=tuple(d.get("changed_functions", ())),
            added_functions=tuple(d.get("added_functions", ())),
            deleted_functions=tuple(d.get("deleted_functions", ())),
            unanalyzed_files=tuple(d.get("unanalyzed_files", ())),
            diff=d.get("diff", ""),
            diff_truncated=d.get("diff_truncated", False),
            pr=PRInfo.from_dict(d["pr"]) if d.get("pr") else None,
            source=d.get("source", "local"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "CandidateCommit":
        return cls.from_dict(json.loads(text))


def candidates_to_json(candidates: list[CandidateCommit] | tuple[CandidateCommit, ...]) -> str:
    """Serialize a whole candidate list (used by the Snapshot Store)."""
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "candidates": [c.to_dict() for c in candidates]},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def candidates_from_json(text: str) -> list[CandidateCommit]:
    data = json.loads(text)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot schema_version {data.get('schema_version')}")
    return [CandidateCommit.from_dict(d) for d in data["candidates"]]
