import json

import pytest

from perfhound.gateway import BenchmarkSpec, Observation, RegressionCase
from perfhound.gateway.sources import (
    GitRangeAdapter, JsonCasesAdapter, available_sources, open_source, register_source,
)


def make_case(**kw) -> RegressionCase:
    fields = dict(
        case_id="swefficiency:sympy-25591#k11", source="swefficiency", repo="https://github.com/sympy/sympy",
        language="python", good="a" * 40, bad="b" * 40, culprit="c" * 40, ground_truth="injected",
        regression_type="extra_loop", expected_magnitude=0.25,
        benchmark=BenchmarkSpec("satask", "script", workload="print(1)", params={"n": 3}),
        observations=[Observation("a" * 40, [1.0, 1.1]), Observation("b" * 40, (1.4,))],
        metadata={"task": "sympy__sympy-25591", "truth_position": 11},
    )
    fields.update(kw)
    return RegressionCase(**fields)


def test_round_trip():
    c = make_case()
    assert RegressionCase.from_json(c.to_json()) == c
    assert c.observations[1].values == (1.4,)


def test_for_localizer_hides_the_answer():
    c = make_case().for_localizer()
    assert c.culprit is None and c.ground_truth == "none"
    assert c.regression_type is None and c.expected_magnitude is None
    assert "truth_position" not in c.metadata and c.metadata["task"] == "sympy__sympy-25591"
    assert "c" * 40 not in c.to_json()
    assert c.case_id.startswith("case-") and c.case_id != make_case().case_id
    assert (c.good, c.bad, c.benchmark) == (make_case().good, make_case().bad, make_case().benchmark)


@pytest.mark.parametrize("kw", [
    {"language": "rust"},
    {"ground_truth": "maybe"},
    {"culprit": None},                         # ground truth without culprit
    {"ground_truth": "none"},                  # culprit without ground truth
    {"good": ""},
])
def test_invalid_cases(kw):
    with pytest.raises(ValueError):
        make_case(**kw)


def test_java_case_without_ground_truth():
    c = make_case(language="java", culprit=None, ground_truth="none",
                  benchmark=BenchmarkSpec("org.x.B.run", "jmh", unit="ops/s", higher_is_better=True))
    assert not c.has_ground_truth
    assert RegressionCase.from_dict(c.to_dict()) == c


@pytest.mark.parametrize("layout", ["list", "object", "jsonl", "single"])
def test_json_adapter_layouts(tmp_path, layout):
    cases = [make_case(), make_case(case_id="x:2", culprit=None, ground_truth="none")]
    p = tmp_path / "cases.json"
    if layout == "list":
        p.write_text(json.dumps([c.to_dict() for c in cases]), encoding="utf-8")
    elif layout == "object":
        p.write_text(json.dumps({"cases": [c.to_dict() for c in cases]}, indent=2), encoding="utf-8")
    elif layout == "jsonl":
        p.write_text("\n".join(c.to_json() for c in cases) + "\n", encoding="utf-8")
    else:
        p.write_text(json.dumps(cases[0].to_dict()), encoding="utf-8")
        cases = cases[:1]
    assert list(open_source("json", path=p).cases()) == cases
    assert len(list(JsonCasesAdapter(p).cases(limit=1))) == 1


def test_json_adapter_reports_which_case_is_bad(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps([make_case().to_dict(), {"case_id": "broken"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="case #1"):
        list(JsonCasesAdapter(p).cases())


def test_git_adapter_and_registry():
    (c,) = open_source("git", repo="https://github.com/jhy/jsoup", good="v1", bad="v2", language="java").cases()
    assert (c.source, c.language, c.ground_truth) == ("git", "java", "none")
    assert {"git", "json"} <= set(available_sources())
    with pytest.raises(ValueError, match="unknown source"):
        open_source("nope")
    with pytest.raises(ValueError, match="already registered"):
        register_source("json", JsonCasesAdapter)
    assert list(GitRangeAdapter(".", "a", "b").cases(limit=0)) == []
