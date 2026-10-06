"""Snapshot Store (gateway part 8b) + Sanitizers.

Rule 7 of the guide: experiments must run from a FROZEN candidate list.
Live data changes (PR edits, force-pushes, a newer analyzer), so every
experiment records exactly what the localizer saw:

    snap = gw.snapshot("experiments/run_042/candidates.json", "v1.2", "HEAD")
    later:  snap = Snapshot.load("experiments/run_042/candidates.json")
            snap.candidates  -> identical CandidateCommit list

The file contains a SHA-256 of its candidates; loading a file that was
edited or damaged raises SnapshotError instead of silently using it.
Writes are atomic (temp file + rename), so a crash never leaves half a file.

Sanitizers (experiment mode)
----------------------------
Injected regressions must not be recognizable by metadata (message,
author, date) - otherwise an LLM ranker "finds" them without understanding
code, and the results are meaningless. Principles:

1. A sanitizer is applied to EVERY candidate, never only to injected ones
   (a lone neutral message would itself be a tell).
2. Code is the signal under test: diffs and function lists are only
   touched by explicit, user-given redaction patterns.
3. The sanitizer's description is stored in the snapshot (auditable).
4. Use find_tells() to check for remaining leaks BEFORE running experiments.

The ground truth (which commit is injected) must never be stored in the
candidates file - keep it in a separate file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from .errors import GatewayError
from .gitcmd import run_git
from .models import SCHEMA_VERSION, CandidateCommit, PRInfo

SNAPSHOT_FORMAT_VERSION = 1


class SnapshotError(GatewayError):
    """Snapshot file is missing, damaged, edited, or from an unknown version."""


# =========================================================================
# sanitizers
# =========================================================================

class Sanitizer(Protocol):
    def __call__(self, candidates: list[CandidateCommit]) -> list[CandidateCommit]: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RedactPatterns:
    """Replace regex matches with a placeholder in messages, PR text and
    (optionally) diffs - for ALL candidates. Case-insensitive."""

    patterns: tuple[str, ...]
    replacement: str = "[redacted]"
    include_diff: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "patterns", tuple(self.patterns))
        for p in self.patterns:
            re.compile(p)   # fail early on a bad pattern

    def _sub(self, text: str) -> str:
        for p in self.patterns:
            text = re.sub(p, self.replacement, text, flags=re.IGNORECASE)
        return text

    def __call__(self, candidates: list[CandidateCommit]) -> list[CandidateCommit]:
        out = []
        for c in candidates:
            pr = c.pr
            if pr is not None:
                pr = PRInfo(pr.number, self._sub(pr.title), self._sub(pr.body),
                            tuple(self._sub(l) for l in pr.labels), pr.url)
            out.append(replace(
                c,
                message=self._sub(c.message),
                diff=self._sub(c.diff) if self.include_diff else c.diff,
                pr=pr,
            ))
        return out

    def describe(self) -> dict[str, Any]:
        return {"name": "RedactPatterns", "patterns": list(self.patterns),
                "replacement": self.replacement, "include_diff": self.include_diff}


@dataclass(frozen=True)
class AnonymizeMetadata:
    """Same author for everyone; timestamps replaced by an evenly spaced
    sequence in range order (keeps order, removes date outliers)."""

    author: str = "anonymous"
    start: datetime = datetime(2000, 1, 1, tzinfo=timezone.utc)
    step: timedelta = timedelta(hours=1)

    def __call__(self, candidates: list[CandidateCommit]) -> list[CandidateCommit]:
        return [replace(c, author=self.author, timestamp=self.start + c.position * self.step) for c in candidates]

    def describe(self) -> dict[str, Any]:
        return {"name": "AnonymizeMetadata", "author": self.author,
                "start": self.start.isoformat(), "step_seconds": self.step.total_seconds()}


@dataclass(frozen=True)
class Compose:
    """Apply several sanitizers in order."""

    steps: tuple[Any, ...]

    def __call__(self, candidates: list[CandidateCommit]) -> list[CandidateCommit]:
        for s in self.steps:
            candidates = s(candidates)
        return candidates

    def describe(self) -> dict[str, Any]:
        return {"name": "Compose", "steps": [s.describe() for s in self.steps]}


DEFAULT_TELL_WORDS = ("inject", "injected", "regression", "synthetic", "perfhound", "artificial", "slowdown")


_SUBJECT_SHAPES = (
    ("merge-commit style", lambda s: s.startswith(("Merge pull request #", "Merge branch ", "Merge remote-tracking"))),
    ("squash-merge style '(#n)'", lambda s: bool(re.search(r"\(#\d+\)\s*$", s))),
)


@dataclass(frozen=True)
class Tell:
    sha: str
    field: str
    reason: str


def find_tells(
    candidates: Sequence[CandidateCommit],
    suspects: Iterable[str],
    words: Sequence[str] = DEFAULT_TELL_WORDS,
) -> list[Tell]:
    """Ways in which `suspects` (e.g. injected commits) stand out from the rest.

    Heuristic leak check - an empty result does NOT prove there is no leak,
    a non-empty one proves there is.
    """
    suspects = set(suspects)
    others = [c for c in candidates if c.sha not in suspects]
    tells: list[Tell] = []
    author_counts = Counter(c.author for c in candidates)
    word_re = {w: re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words}

    def has(word: str, c: CandidateCommit, where: str) -> bool:
        if where == "message":
            text = c.message + ("\n" + c.pr.title + "\n" + c.pr.body if c.pr else "")
        else:
            text = "\n".join(l for l in c.diff.split("\n") if l.startswith("+") and not l.startswith("+++"))
        return bool(word_re[word].search(text))

    times = [c.timestamp for c in candidates]
    span = (max(times) - min(times)) if len(times) > 1 else timedelta(0)

    for c in candidates:
        if c.sha not in suspects:
            continue
        if others and author_counts[c.author] == sum(1 for x in candidates if x.sha in suspects and x.author == c.author):
            tells.append(Tell(c.sha, "author", f"author {c.author!r} appears only on suspect commits"))
        for where in ("message", "diff"):
            for w in words:
                if has(w, c, where) and not any(has(w, o, where) for o in others):
                    tells.append(Tell(c.sha, where, f"word {w!r} appears only in suspect commits"))
        # message SHAPE: e.g. every main-line commit is "Merge pull request #..." or ends in "(#123)"
        if len(others) >= 3:
            for label, shape in _SUBJECT_SHAPES:
                share = sum(1 for o in others if shape(o.subject)) / len(others)
                if share >= 0.8 and not shape(c.subject):
                    tells.append(Tell(c.sha, "message", f"subject does not look like the others ({label}: "
                                                        f"{share:.0%} of other commits)"))
            # timezone offset nobody else uses
            if c.timestamp.utcoffset() not in {o.timestamp.utcoffset() for o in others}:
                tells.append(Tell(c.sha, "timestamp", f"timezone {c.timestamp.strftime('%z')} used by no other commit"))
        # date out of order relative to its neighbours on the main line
        idx = candidates.index(c)
        prev_t = candidates[idx - 1].timestamp if idx > 0 else None
        next_t = candidates[idx + 1].timestamp if idx + 1 < len(candidates) else None
        if (prev_t and c.timestamp < prev_t) or (next_t and c.timestamp > next_t):
            tells.append(Tell(c.sha, "timestamp", "timestamp is out of order with neighbouring commits"))
        elif span and others and len(candidates) >= 4:
            gaps = sorted(abs((b.timestamp - a.timestamp).total_seconds()) for a, b in zip(candidates, candidates[1:]))
            median = gaps[len(gaps) // 2] or 1
            own = max(abs((c.timestamp - t).total_seconds()) for t in (prev_t, next_t) if t)
            if own > 50 * median:
                tells.append(Tell(c.sha, "timestamp", "unusually large time gap to neighbouring commits"))
    return tells


# =========================================================================
# snapshot
# =========================================================================

def _content_hash(candidate_dicts: list[dict]) -> str:
    canonical = json.dumps(candidate_dicts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    candidates: tuple[CandidateCommit, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))

    @property
    def content_hash(self) -> str:
        return _content_hash([c.to_dict() for c in self.candidates])

    # -- io ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dicts = [c.to_dict() for c in self.candidates]
        doc = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "candidate_schema_version": SCHEMA_VERSION,
            "content_sha256": _content_hash(dicts),
            "meta": self.meta,
            "candidates": dicts,
        }
        text = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)   # atomic on Windows and POSIX
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    @classmethod
    def load(cls, path: str | Path, *, mark_source: bool = False) -> "Snapshot":
        """Load and verify a snapshot. mark_source=True sets source="snapshot"
        on every record (by default records are exactly as saved)."""
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise SnapshotError(f"snapshot {path} does not exist") from None
        except (OSError, ValueError) as exc:
            raise SnapshotError(f"snapshot {path} is not valid JSON: {exc}") from None
        if doc.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            raise SnapshotError(f"unsupported snapshot format_version {doc.get('format_version')!r}")
        dicts = doc.get("candidates")
        if not isinstance(dicts, list) or _content_hash(dicts) != doc.get("content_sha256"):
            raise SnapshotError(f"snapshot {path} was modified or damaged (content hash mismatch)")
        try:
            cands = [CandidateCommit.from_dict(d) for d in dicts]
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotError(f"snapshot {path} contains an invalid record: {exc}") from None
        if mark_source:
            cands = [replace(c, source="snapshot") for c in cands]
        return cls(tuple(cands), doc.get("meta", {}))

    # -- checks --------------------------------------------------------------

    def missing_commits(self, repo: str | Path) -> list[str]:
        """SHAs of this snapshot that do not exist in `repo` (e.g. after a force-push)."""
        if not self.candidates:
            return []
        proc = run_git(repo, "cat-file", "--batch-check", input="\n".join(c.sha for c in self.candidates) + "\n")
        return [line.split()[0] for line in proc.stdout.splitlines() if line.endswith(" missing")]


def build_snapshot(
    candidates: list[CandidateCommit],
    meta: dict[str, Any],
    sanitizer: Callable[[list[CandidateCommit]], list[CandidateCommit]] | None = None,
) -> Snapshot:
    meta = dict(meta)
    if sanitizer is not None:
        before = len(candidates)
        candidates = sanitizer(list(candidates))
        if len(candidates) != before:
            raise SnapshotError("a sanitizer must not add or remove candidates")
        meta["sanitizer"] = sanitizer.describe() if hasattr(sanitizer, "describe") else {"name": repr(sanitizer)}
    else:
        meta["sanitizer"] = None
    return Snapshot(tuple(candidates), meta)
