from datetime import datetime, timedelta, timezone

import pytest

from perfhound.gateway.models import (
    CandidateCommit,
    FileChange,
    PRInfo,
    candidates_from_json,
    candidates_to_json,
)

SHA_A = "a3f9c21" + "0" * 33
SHA_B = "7be01d4" + "1" * 33
IST = timezone(timedelta(hours=5, minutes=30))


def make_candidate(**overrides) -> CandidateCommit:
    fields = dict(
        sha=SHA_A,
        parent=SHA_B,
        position=3,
        message="Refactor cache lookup\n\nLonger body with ünïcödé.",
        author="Kanika",
        timestamp=datetime(2026, 10, 4, 11, 30, tzinfo=IST),
        files=[
            FileChange("sympy/core/basic.py", "M", 5, 11),
            FileChange("new_name.py", "R", 0, 0, old_path="old_name.py"),
            FileChange("logo.png", "A", None, None),
        ],
        changed_functions=["sympy.core.basic.Basic.__hash__"],
        added_functions=["sympy.core.basic.helper"],
        deleted_functions=[],
        diff="@@ -74,7 +74,7 @@\n-    old\n+    new\n",
        diff_truncated=True,
        pr=PRInfo(25591, "Speed up hashing", "body", labels=["performance"], url="https://x"),
        source="local+github",
    )
    fields.update(overrides)
    return CandidateCommit(**fields)


def test_json_round_trip_is_exact():
    c = make_candidate()
    back = CandidateCommit.from_json(c.to_json())
    assert back == c
    assert back.timestamp.utcoffset() == timedelta(hours=5, minutes=30)


def test_round_trip_without_pr():
    c = make_candidate(pr=None, source="local")
    assert CandidateCommit.from_json(c.to_json()) == c


def test_list_round_trip():
    cs = [make_candidate(), make_candidate(position=4, pr=None, source="local")]
    assert candidates_from_json(candidates_to_json(cs)) == cs


def test_lists_are_stored_as_tuples_and_record_is_hashable():
    c = make_candidate()
    assert isinstance(c.files, tuple)
    assert isinstance(c.changed_functions, tuple)
    assert isinstance(c.pr.labels, tuple)
    hash(c)  # must not raise


def test_record_is_immutable():
    c = make_candidate()
    with pytest.raises(AttributeError):
        c.position = 9  # type: ignore[misc]


def test_helpers():
    c = make_candidate()
    assert c.short_sha == "a3f9c21"
    assert c.subject == "Refactor cache lookup"
    assert c.files[2].is_binary
    assert not c.files[0].is_binary


@pytest.mark.parametrize(
    "overrides",
    [
        {"sha": "abc"},  # short sha
        {"sha": SHA_A.upper()},  # uppercase
        {"parent": "not-a-sha"},
        {"position": -1},
        {"timestamp": datetime(2026, 1, 1)},  # naive datetime
        {"source": "gitlab"},
    ],
)
def test_invalid_candidate_rejected(overrides):
    with pytest.raises(ValueError):
        make_candidate(**overrides)


def test_invalid_file_change_rejected():
    with pytest.raises(ValueError):
        FileChange("a.py", "X")
    with pytest.raises(ValueError):
        FileChange("a.py", "R")  # rename without old_path
    with pytest.raises(ValueError):
        FileChange("a.py", "M", -1, 0)


def test_unknown_schema_version_rejected():
    d = make_candidate().to_dict()
    d["schema_version"] = 99
    with pytest.raises(ValueError):
        CandidateCommit.from_dict(d)
