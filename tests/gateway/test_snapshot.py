import json
import os
import shutil
import subprocess
from datetime import timedelta

import pytest

from perfhound.gateway import Gateway, Snapshot, SnapshotError
from perfhound.gateway.snapshot import AnonymizeMetadata, Compose, RedactPatterns, build_snapshot, find_tells

from .fixture_repo import BAD_TAG, GOOD_TAG


@pytest.fixture
def gw(fixture_repo, tmp_path):
    return Gateway(fixture_repo.path, cache=tmp_path / "c.db")


def test_round_trip_is_identical(gw, tmp_path):
    original = gw.get_candidates(GOOD_TAG, BAD_TAG)
    path = tmp_path / "experiments" / "run_042" / "candidates.json"
    saved = gw.snapshot(path, GOOD_TAG, BAD_TAG)
    loaded = Snapshot.load(path)
    assert list(loaded.candidates) == original          # the guide's "Done kab?"
    assert loaded.meta == saved.meta
    assert loaded.content_hash == saved.content_hash


def test_meta_records_provenance(gw, fixture_repo, tmp_path):
    snap = gw.snapshot(tmp_path / "s.json", GOOD_TAG, BAD_TAG)
    m = snap.meta
    assert (m["good"], m["bad"]) == (fixture_repo.sha("initial"), fixture_repo.sha("slow_add"))
    assert (m["good_ref"], m["bad_ref"]) == (GOOD_TAG, BAD_TAG)
    assert m["settings"]["max_diff_lines"] == 400 and m["sanitizer"] is None
    assert m["repo_id"] == gw.repo_id and m["created_at"]


def test_mark_source(gw, tmp_path):
    gw.snapshot(tmp_path / "s.json", GOOD_TAG, BAD_TAG)
    assert {c.source for c in Snapshot.load(tmp_path / "s.json", mark_source=True).candidates} == {"snapshot"}


def test_loading_needs_no_repo(fresh_fixture_repo, tmp_path):
    path = tmp_path / "s.json"
    Gateway(fresh_fixture_repo.path, cache=False).snapshot(path, GOOD_TAG, BAD_TAG)
    shutil.rmtree(fresh_fixture_repo.path, onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)))
    assert len(Snapshot.load(path).candidates) == 7


def test_edited_snapshot_is_rejected(gw, tmp_path):
    path = tmp_path / "s.json"
    gw.snapshot(path, GOOD_TAG, BAD_TAG)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["candidates"][6]["message"] = "Cache lookups (definitely the culprit)"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SnapshotError, match="modified or damaged"):
        Snapshot.load(path)


@pytest.mark.parametrize("content", ["", "{not json", '{"format_version": 99}'])
def test_broken_files_are_rejected(tmp_path, content):
    path = tmp_path / "s.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SnapshotError):
        Snapshot.load(path)
    with pytest.raises(SnapshotError, match="does not exist"):
        Snapshot.load(tmp_path / "nope.json")


def test_failed_save_leaves_existing_file_intact(gw, tmp_path):
    path = tmp_path / "s.json"
    gw.snapshot(path, GOOD_TAG, BAD_TAG)
    before = path.read_bytes()

    def broken(cands):
        raise RuntimeError("sanitizer crashed")

    with pytest.raises(RuntimeError):
        gw.snapshot(path, GOOD_TAG, BAD_TAG, sanitizer=broken)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_missing_commits_after_history_rewrite(fresh_fixture_repo, tmp_path):
    repo = fresh_fixture_repo
    path = tmp_path / "s.json"
    snap = Gateway(repo.path, cache=False).snapshot(path, GOOD_TAG, BAD_TAG)
    assert snap.missing_commits(repo.path) == []
    other = tmp_path / "other"
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    assert len(Snapshot.load(path).missing_commits(other)) == 7


# ------------------------------------------------------------------ sanitizers

def test_redact_patterns_applies_to_all_and_keeps_code_signal(gw):
    cands = gw.get_candidates(GOOD_TAG, BAD_TAG)
    out = RedactPatterns((r"\brobust\b", r"\bint\b"), include_diff=False)(cands)
    assert out[6].message == "Make add() more [redacted]"
    assert out[0].message == "Coerce inputs to [redacted] in add()"     # non-suspect commit too
    assert [c.diff for c in out] == [c.diff for c in cands]               # code untouched
    assert [c.changed_functions for c in out] == [c.changed_functions for c in cands]


def test_anonymize_metadata(gw):
    cands = gw.get_candidates(GOOD_TAG, BAD_TAG)
    out = AnonymizeMetadata()(cands)
    assert {c.author for c in out} == {"anonymous"}
    assert all(b.timestamp - a.timestamp == timedelta(hours=1) for a, b in zip(out, out[1:]))
    assert [(c.sha, c.diff, c.changed_functions) for c in out] == [(c.sha, c.diff, c.changed_functions) for c in cands]


def test_sanitizer_is_recorded_and_must_not_drop_commits(gw, tmp_path):
    san = Compose((RedactPatterns(("inject",)), AnonymizeMetadata()))
    snap = gw.snapshot(tmp_path / "s.json", GOOD_TAG, BAD_TAG, sanitizer=san)
    assert snap.meta["sanitizer"]["name"] == "Compose"
    assert snap.meta["sanitizer"]["steps"][0]["patterns"] == ["inject"]
    assert Snapshot.load(tmp_path / "s.json").meta["sanitizer"] == snap.meta["sanitizer"]
    with pytest.raises(SnapshotError, match="add or remove"):
        build_snapshot(gw.get_candidates(GOOD_TAG, BAD_TAG), {}, lambda cs: cs[:-1])


def test_find_tells_catches_a_careless_injection_and_sanitizing_removes_it(fresh_fixture_repo, tmp_path):
    repo = fresh_fixture_repo
    path = repo.path / "mathops.py"
    path.write_text(path.read_text(encoding="utf-8") +
                    "\n\n# perfhound: injected regression\ndef helper():\n    return sum(range(10**6))\n",
                    encoding="utf-8", newline="\n")
    # careless injector: own author, telling message, commit date = today (fixture dates are 2025)
    repo.git("commit", "-q", "-a", "--author=perfhound-injector <i@x>", "-m", "Inject synthetic regression #3")
    gw = Gateway(repo.path, cache=False)
    cands = gw.get_candidates(GOOD_TAG, "HEAD")
    injected = cands[-1].sha

    tells = find_tells(cands, [injected])
    fields = {t.field for t in tells}
    assert {"author", "message", "diff"} <= fields
    assert find_tells(cands, [cands[2].sha]) == []      # an ordinary commit has no tells

    san = Compose((RedactPatterns((r"#.*perfhound.*", r"\binject\w*", r"\bsynthetic\b", r"\bregression\b")),
                   AnonymizeMetadata()))
    clean = san(cands)
    assert find_tells(clean, [injected]) == []
    assert clean[-1].added_functions == ("mathops.helper",)        # the real signal survives
    assert "+    return sum(range(10**6))" in clean[-1].diff


def test_find_tells_timestamp_out_of_order():
    from perfhound.gateway.models import CandidateCommit
    from datetime import datetime, timezone

    def c(i, hour):
        return CandidateCommit(sha=f"{i:040x}", parent=f"{i + 100:040x}", position=i, message="m", author="a",
                               timestamp=datetime(2025, 1, 1, hour, tzinfo=timezone.utc))

    cands = [c(0, 1), c(1, 2), c(2, 0), c(3, 4)]
    assert [t.field for t in find_tells(cands, [cands[2].sha])] == ["timestamp"]
