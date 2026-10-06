"""RegressionCase: one localization problem, whatever source it came from.

Every Source Adapter (SWE-fficiency, asv-runner, JMH data, a JSON file, a
plain git URL ...) turns its own format into RegressionCase records - the
schema from the synopsis: project, benchmark, good commit, bad commit,
culprit commit, regression type, expected magnitude, observations.

Ground truth is kept OUT of what the localizer sees: call
`case.for_localizer()` before handing a case to the prioritizer /
scheduler - it drops the culprit (and anything else that reveals it).
"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field, replace
from typing import Any

CASE_SCHEMA_VERSION = 1

LANGUAGES = frozenset({"python", "java"})
METRIC_TYPES = frozenset({"wall_time", "regex", "json"})
# What changed between good and bad: a slowdown (regression) or a speedup (e.g. an optimization).
DIRECTIONS = frozenset({"slower", "faster"})
# How trustworthy `culprit` is:
#   injected - we created the regression ourselves (exact)
#   reported - maintainers confirmed it (issue / PR / paper)
#   detected - an automatic detector flagged it (heuristic, may be wrong)
#   none     - no ground truth (real-world run, culprit unknown)
GROUND_TRUTH_KINDS = frozenset({"injected", "reported", "detected", "none"})


@dataclass(frozen=True)
class BenchmarkSpec:
    """What to measure. `framework` decides which runner executes it."""

    name: str                       # e.g. "frame_methods.Apply.time_apply", "org.x.MyBench.run"
    framework: str                  # "command" | "script" | "pytest-benchmark" | "asv" | "jmh"
    command: str | None = None      # how to run it inside a checkout (shell command)
    workload: str | None = None     # inline workload source (SWE-fficiency)
    params: dict[str, Any] = field(default_factory=dict)
    unit: str = "seconds"
    higher_is_better: bool = False  # True for throughput (JMH thrpt mode)
    setup: str | None = None        # build step run once per commit before measuring, e.g. "pip install -e ."
    # How to read one sample:  {"type": "wall_time"}  (runner times the command, default)
    #                          {"type": "regex", "pattern": "took ([0-9.]+) s"}
    #                          {"type": "json", "key": "mean"}   (command prints JSON)
    metric: dict[str, Any] = field(default_factory=lambda: {"type": "wall_time"})

    def __post_init__(self) -> None:
        if not self.name or not self.framework:
            raise ValueError("BenchmarkSpec needs name and framework")
        kind = self.metric.get("type")
        if kind not in METRIC_TYPES:
            raise ValueError(f"metric type must be one of {sorted(METRIC_TYPES)}, got {kind!r}")
        if kind == "regex":
            pattern = self.metric.get("pattern", "")
            if re.compile(pattern).groups != 1:
                raise ValueError("regex metric needs exactly one capture group for the number")
        if kind == "json" and not self.metric.get("key"):
            raise ValueError("json metric needs a 'key'")


@dataclass(frozen=True)
class Observation:
    """Already-recorded measurements of the benchmark at one commit
    (asv / JMH datasets). Lets SPRT be replayed without running anything."""

    commit: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        if not self.commit:
            raise ValueError("Observation needs a commit")


@dataclass(frozen=True)
class RegressionCase:
    case_id: str                    # unique, "<source>:<id>"
    source: str                     # adapter name
    repo: str                       # git URL or local path
    language: str
    good: str                       # commit BEFORE the change (fast for a regression)
    bad: str                        # commit AFTER the change
    culprit: str | None = None
    ground_truth: str = "none"
    regression_type: str | None = None
    # relative change of the measured time, |t_bad - t_good| / t_good  (0.15 = 15 % slower / faster)
    expected_magnitude: float | None = None
    benchmark: BenchmarkSpec | None = None
    observations: tuple[Observation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    direction: str = "slower"       # "slower" (regression) or "faster" (speedup)

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}, got {self.direction!r}")
        for name in ("case_id", "source", "repo", "good", "bad"):
            if not getattr(self, name):
                raise ValueError(f"RegressionCase.{name} is required")
        if self.language not in LANGUAGES:
            raise ValueError(f"language must be one of {sorted(LANGUAGES)}, got {self.language!r}")
        if self.ground_truth not in GROUND_TRUTH_KINDS:
            raise ValueError(f"ground_truth must be one of {sorted(GROUND_TRUTH_KINDS)}")
        if self.ground_truth != "none" and not self.culprit:
            raise ValueError("a case with ground truth needs a culprit")
        if self.ground_truth == "none" and self.culprit:
            raise ValueError("culprit given but ground_truth is 'none'")
        object.__setattr__(self, "observations", tuple(self.observations))

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth != "none"

    def for_localizer(self) -> "RegressionCase":
        """Copy without anything that reveals the answer."""
        meta = {k: v for k, v in self.metadata.items() if not k.startswith("truth_")}
        # case ids are built from dataset ids / positions ("...dask-10356__real_n20_k13"): opaque id instead
        opaque = "case-" + hashlib.sha1(self.case_id.encode("utf-8")).hexdigest()[:12]
        return replace(self, case_id=opaque, culprit=None, ground_truth="none", regression_type=None,
                       expected_magnitude=None, metadata=meta)

    # ---- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        b = self.benchmark
        return {
            "schema_version": CASE_SCHEMA_VERSION,
            "case_id": self.case_id, "source": self.source, "repo": self.repo,
            "language": self.language, "good": self.good, "bad": self.bad,
            "culprit": self.culprit, "ground_truth": self.ground_truth,
            "regression_type": self.regression_type, "expected_magnitude": self.expected_magnitude,
            "direction": self.direction,
            "benchmark": None if b is None else {
                "name": b.name, "framework": b.framework, "command": b.command, "workload": b.workload,
                "params": b.params, "unit": b.unit, "higher_is_better": b.higher_is_better,
                "setup": b.setup, "metric": b.metric,
            },
            "observations": [{"commit": o.commit, "values": list(o.values)} for o in self.observations],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RegressionCase":
        if d.get("schema_version", CASE_SCHEMA_VERSION) != CASE_SCHEMA_VERSION:
            raise ValueError(f"unsupported RegressionCase schema_version {d.get('schema_version')}")
        b = d.get("benchmark")
        return cls(
            case_id=d["case_id"], source=d["source"], repo=d["repo"], language=d["language"],
            good=d["good"], bad=d["bad"], culprit=d.get("culprit"),
            ground_truth=d.get("ground_truth", "none"), regression_type=d.get("regression_type"),
            expected_magnitude=d.get("expected_magnitude"),
            direction=d.get("direction", "slower"),
            benchmark=None if not b else BenchmarkSpec(
                name=b["name"], framework=b["framework"], command=b.get("command"), workload=b.get("workload"),
                params=b.get("params") or {}, unit=b.get("unit", "seconds"),
                higher_is_better=b.get("higher_is_better", False),
                setup=b.get("setup"), metric=b.get("metric") or {"type": "wall_time"},
            ),
            observations=tuple(Observation(o["commit"], tuple(o["values"])) for o in d.get("observations", ())),
            metadata=d.get("metadata") or {},
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RegressionCase":
        return cls.from_dict(json.loads(text))
