"""Source Adapters: every outside source enters Perfhound through one.

An adapter knows ONE format (a dataset, a results database, a file) and
yields RegressionCase records. Nothing after the adapter knows where a case
came from - adding a source never changes the gateway or the localizer.

    from perfhound.gateway.sources import open_source
    for case in open_source("json", path="cases.json").cases():
        ...

Built in:  "yaml"  - perfhound.yaml: any repo + good + bad + setup + benchmark
           "json"  - cases in Perfhound's own schema (JSON list or JSONL)
           "git"   - one ad-hoc case: repo URL/path + good + bad
Dataset adapters (SWE-fficiency, asv-runner, JMH ...) register the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from ..cases import BenchmarkSpec, RegressionCase


class SourceAdapter(Protocol):
    name: str

    def cases(self, limit: int | None = None) -> Iterator[RegressionCase]: ...


_REGISTRY: dict[str, Callable[..., SourceAdapter]] = {}


def register_source(name: str, factory: Callable[..., SourceAdapter]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"source {name!r} is already registered")
    _REGISTRY[name] = factory


def available_sources() -> list[str]:
    return sorted(_REGISTRY)


def open_source(name: str, **options: Any) -> SourceAdapter:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown source {name!r}; available: {available_sources()}") from None
    return factory(**options)


class JsonCasesAdapter:
    """Cases already in Perfhound's schema: a JSON list, {"cases": [...]}, or JSON Lines."""

    name = "json"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def cases(self, limit: int | None = None) -> Iterator[RegressionCase]:
        text = self.path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            items = data["cases"] if isinstance(data, dict) and "cases" in data else data
            if isinstance(items, dict):   # a single case object
                items = [items]
        except json.JSONDecodeError:       # JSON Lines: one case per line
            items = [json.loads(line) for line in text.splitlines() if line.strip()]
        for i, item in enumerate(items):
            if limit is not None and i >= limit:
                return
            try:
                yield RegressionCase.from_dict(item)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{self.path}: case #{i} is invalid: {exc}") from None


class GitRangeAdapter:
    """One ad-hoc case from the command line / VS Code: repo + good + bad (no ground truth)."""

    name = "git"

    def __init__(self, repo: str, good: str, bad: str, language: str = "python",
                 benchmark: BenchmarkSpec | None = None, case_id: str | None = None) -> None:
        self.case = RegressionCase(
            case_id=case_id or f"git:{repo}@{good}..{bad}", source="git", repo=repo, language=language,
            good=good, bad=bad, benchmark=benchmark,
        )

    def cases(self, limit: int | None = None) -> Iterator[RegressionCase]:
        if limit is None or limit > 0:
            yield self.case


register_source("json", JsonCasesAdapter)
register_source("git", GitRangeAdapter)


def _load_builtin_adapters() -> None:
    """Dataset adapters register themselves on import."""
    from . import asv_runner, swefficiency, yaml_spec  # noqa: F401


_load_builtin_adapters()
