"""`perfhound.yaml`: describe a case for ANY git repository.

    # perfhound.yaml
    repo: https://github.com/jhy/jsoup        # URL or local path (relative to this file)
    # or instead of repo + good + bad:  compare: https://github.com/jhy/jsoup/compare/jsoup-1.17.1...master
    language: java                            # python | java
    good: jsoup-1.17.1                        # before the change
    bad: master                               # after the change
    direction: slower                         # slower (default) | faster
    benchmark:
      name: parse_big_html
      setup: mvn -q -DskipTests package       # once per commit, optional
      command: java -cp target/classes Bench  # one sample = one run
      metric: {type: wall_time}               # or {type: regex, pattern: "took ([0-9.]+)"} / {type: json, key: mean}
    # evaluation only (never shown to the localizer):
    # culprit: 1a2b3c...
    # ground_truth: reported

A file holds one case, or several under `cases:`; keys under `defaults:` apply
to every case. Unknown keys are an ERROR (a typo like `benchmak:` must not be
silently ignored).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import yaml

from ..cases import BenchmarkSpec, RegressionCase
from ..fetcher import is_remote
from . import register_source

_CASE_KEYS = {"name", "repo", "compare", "language", "good", "bad", "direction", "benchmark", "culprit",
              "ground_truth", "regression_type", "expected_magnitude", "metadata"}
_BENCH_KEYS = {"name", "command", "setup", "metric", "unit", "higher_is_better", "framework", "params", "workload"}


class SpecError(ValueError):
    """perfhound.yaml is invalid; the message names the file, case and field."""


def _check_keys(where: str, data: dict, allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _benchmark(where: str, raw: Any) -> BenchmarkSpec | None:
    if raw is None:
        return None
    if isinstance(raw, str):                 # shorthand: benchmark: "python bench.py"
        raw = {"command": raw}
    if not isinstance(raw, dict):
        raise SpecError(f"{where}.benchmark must be a command string or a mapping")
    _check_keys(f"{where}.benchmark", raw, _BENCH_KEYS)
    if not raw.get("command") and not raw.get("workload"):
        raise SpecError(f"{where}.benchmark needs a 'command' (or an inline 'workload')")
    name = raw.get("name") or re.sub(r"\s+", " ", str(raw.get("command") or "workload"))[:60]
    try:
        return BenchmarkSpec(
            name=name, framework=raw.get("framework", "command"), command=raw.get("command"),
            workload=raw.get("workload"), params=raw.get("params") or {}, unit=raw.get("unit", "seconds"),
            higher_is_better=bool(raw.get("higher_is_better", False)), setup=raw.get("setup"),
            metric=raw.get("metric") or {"type": "wall_time"},
        )
    except ValueError as exc:
        raise SpecError(f"{where}.benchmark: {exc}") from None


def case_from_spec(raw: dict, *, base_dir: Path, index: int = 0, source: str = "yaml") -> RegressionCase:
    where = f"case #{index}" + (f" ({raw.get('name')})" if isinstance(raw, dict) and raw.get("name") else "")
    if not isinstance(raw, dict):
        raise SpecError(f"{where}: must be a mapping")
    _check_keys(where, raw, _CASE_KEYS)
    if raw.get("compare"):                   # compare: https://github.com/o/r/compare/GOOD...BAD
        from ..github.errors import InvalidGitHubLink
        from ..github.links import parse_compare

        if any(raw.get(k) for k in ("repo", "good", "bad")):
            raise SpecError(f"{where}: give either 'compare' or 'repo' + 'good' + 'bad', not both")
        try:
            ref, good, bad = parse_compare(str(raw["compare"]))
        except InvalidGitHubLink as exc:
            raise SpecError(f"{where}: {exc}") from None
        raw = {**{k: v for k, v in raw.items() if k != "compare"}, "repo": ref.url, "good": good, "bad": bad}
    for key in ("repo", "good", "bad"):
        if not raw.get(key):
            raise SpecError(f"{where}: '{key}' is required")
    repo = str(raw["repo"])
    if not is_remote(repo) and not Path(repo).is_absolute():
        repo = str((base_dir / repo).resolve())
    culprit = raw.get("culprit")
    truth = raw.get("ground_truth") or ("reported" if culprit else "none")
    name = raw.get("name") or f"{Path(repo.rstrip('/')).name}@{raw['good']}..{raw['bad']}"
    try:
        return RegressionCase(
            case_id=f"{source}:{name}", source=source, repo=repo, language=str(raw.get("language", "python")),
            good=str(raw["good"]), bad=str(raw["bad"]), culprit=str(culprit) if culprit else None,
            ground_truth=truth, regression_type=raw.get("regression_type"),
            expected_magnitude=raw.get("expected_magnitude"), benchmark=_benchmark(where, raw.get("benchmark")),
            metadata=raw.get("metadata") or {}, direction=raw.get("direction", "slower"),
        )
    except ValueError as exc:
        raise SpecError(f"{where}: {exc}") from None


def load_spec(path: str | Path) -> list[RegressionCase]:
    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SpecError(f"{path} does not exist") from None
    except yaml.YAMLError as exc:
        raise SpecError(f"{path} is not valid YAML: {exc}") from None
    if not isinstance(doc, dict):
        raise SpecError(f"{path}: expected a mapping at the top level")
    base = path.parent
    if "cases" in doc:
        _check_keys(str(path), doc, {"cases", "defaults"})
        defaults = doc.get("defaults") or {}
        return [case_from_spec({**defaults, **c}, base_dir=base, index=i) for i, c in enumerate(doc["cases"] or [])]
    return [case_from_spec(doc, base_dir=base)]


class YamlSpecAdapter:
    """Cases from a perfhound.yaml file - the way to bring ANY repository in."""

    name = "yaml"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def cases(self, limit: int | None = None) -> Iterator[RegressionCase]:
        for i, case in enumerate(load_spec(self.path)):
            if limit is not None and i >= limit:
                return
            yield case


register_source("yaml", YamlSpecAdapter)
