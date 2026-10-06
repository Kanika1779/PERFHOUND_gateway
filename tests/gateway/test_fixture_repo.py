"""Sanity checks on the fixture repo itself, so later test failures
can be blamed on gateway code, not on a broken fixture."""

import time

from .fixture_repo import BAD_TAG, FIRST_PARENT_ORDER, GOOD_TAG, build_fixture_repo


def test_tags_point_to_expected_commits(fixture_repo):
    assert fixture_repo.git("rev-parse", GOOD_TAG + "^{commit}") == fixture_repo.sha("initial")
    assert fixture_repo.git("rev-parse", BAD_TAG + "^{commit}") == fixture_repo.sha("slow_add")


def test_first_parent_range_has_seven_commits_in_order(fixture_repo):
    out = fixture_repo.git("rev-list", "--first-parent", "--reverse", f"{GOOD_TAG}..{BAD_TAG}")
    assert out.split() == fixture_repo.first_parent_shas()
    assert len(FIRST_PARENT_ORDER) == 7


def test_full_range_includes_feature_branch_commit(fixture_repo):
    out = fixture_repo.git("rev-list", f"{GOOD_TAG}..{BAD_TAG}").split()
    assert len(out) == 8
    assert fixture_repo.sha("feature_square") in out


def test_merge_commit_has_two_parents(fixture_repo):
    parents = fixture_repo.git("rev-list", "--parents", "-n", "1", fixture_repo.sha("merge")).split()[1:]
    assert parents == [fixture_repo.sha("docs"), fixture_repo.sha("feature_square")]


def test_rename_is_detected_by_git(fixture_repo):
    out = fixture_repo.git("show", "--name-status", "-M", "--format=", fixture_repo.sha("rename"))
    assert out.startswith("R100") and "calc.py" in out and "mathops.py" in out


def test_binary_file_shows_dash_in_numstat(fixture_repo):
    out = fixture_repo.git("show", "--numstat", "--format=", fixture_repo.sha("docs"))
    assert "-\t-\tdata.bin" in out


def test_working_tree_is_clean(fixture_repo):
    assert fixture_repo.git("status", "--porcelain") == ""


def test_shas_are_deterministic(tmp_path, fixture_repo):
    other = build_fixture_repo(tmp_path / "again")
    assert other.shas == fixture_repo.shas


def test_build_is_fast(tmp_path):
    start = time.perf_counter()
    build_fixture_repo(tmp_path / "timed")
    elapsed = time.perf_counter() - start
    print(f"\nfixture repo build: {elapsed:.2f}s")
    assert elapsed < 5  # target is ~1s; loose bound so slow Windows disks don't flake
