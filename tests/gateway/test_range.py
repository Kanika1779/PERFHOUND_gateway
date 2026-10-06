import warnings

import pytest

from perfhound.gateway.errors import InvalidRangeError, NotAGitRepoError, RangeWarning, UnknownRefError
from perfhound.gateway.range import find_repo_root, resolve_range, resolve_ref

from .fixture_repo import BAD_TAG, GOOD_TAG


def resolve_quietly(*args, **kwargs):
    """resolve_range that fails the test if any RangeWarning is emitted."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RangeWarning)
        return resolve_range(*args, **kwargs)


def test_resolves_tags_to_first_parent_list_in_order(fixture_repo):
    r = resolve_quietly(fixture_repo.path, GOOD_TAG, BAD_TAG)
    assert list(r.shas) == fixture_repo.first_parent_shas()
    assert len(r) == 7
    assert r.good == fixture_repo.sha("initial")
    assert r.bad == fixture_repo.sha("slow_add")
    assert (r.good_ref, r.bad_ref) == (GOOD_TAG, BAD_TAG)


def test_merged_branch_commit_is_not_listed_separately(fixture_repo):
    r = resolve_quietly(fixture_repo.path, GOOD_TAG, BAD_TAG)
    assert fixture_repo.sha("feature_square") not in r.shas
    assert fixture_repo.sha("merge") in r.shas


def test_position_of(fixture_repo):
    r = resolve_quietly(fixture_repo.path, GOOD_TAG, BAD_TAG)
    assert r.position_of(fixture_repo.sha("tweak_add")) == 0
    assert r.position_of(fixture_repo.sha("slow_add")) == 6


@pytest.mark.parametrize("good,bad,expected_len", [
    ("v1.0", "HEAD", 7),
    ("v1.0", "main", 7),
    ("HEAD~3", "HEAD", 3),
])
def test_accepts_branch_head_and_relative_refs(fixture_repo, good, bad, expected_len):
    assert len(resolve_quietly(fixture_repo.path, good, bad)) == expected_len


def test_accepts_short_shas(fixture_repo):
    r = resolve_quietly(fixture_repo.path, fixture_repo.sha("rename")[:7], fixture_repo.sha("circle_area")[:8])
    assert list(r.shas) == [fixture_repo.sha("delete_sub"), fixture_repo.sha("circle_area")]


def test_works_from_a_subfolder(fixture_repo, tmp_path):
    sub = fixture_repo.path / "nested"
    sub.mkdir(exist_ok=True)
    r = resolve_quietly(sub, GOOD_TAG, BAD_TAG)
    assert r.repo.resolve() == fixture_repo.path.resolve()


def test_not_a_repo(tmp_path):
    with pytest.raises(NotAGitRepoError):
        find_repo_root(tmp_path)
    with pytest.raises(NotAGitRepoError):
        resolve_range(tmp_path / "missing", "a", "b")


@pytest.mark.parametrize("ref", ["v9.9", "deadbeef", "", "  ", "--all", "-n1"])
def test_unknown_or_dangerous_ref(fixture_repo, ref):
    with pytest.raises(UnknownRefError):
        resolve_ref(fixture_repo.path, ref)


def test_same_commit_is_empty_range(fixture_repo):
    with pytest.raises(InvalidRangeError, match="same commit"):
        resolve_range(fixture_repo.path, BAD_TAG, "HEAD")


def test_swapped_good_and_bad_gives_hint(fixture_repo):
    with pytest.raises(InvalidRangeError, match="swap"):
        resolve_range(fixture_repo.path, BAD_TAG, GOOD_TAG)


def test_unrelated_branches(fixture_repo):
    # feature_square and docs both branch off circle_area; neither contains the other
    with pytest.raises(InvalidRangeError, match="not an ancestor"):
        resolve_range(fixture_repo.path, fixture_repo.sha("feature_square"), fixture_repo.sha("docs"))


def test_good_on_side_branch_warns(fixture_repo):
    with pytest.warns(RangeWarning, match="first-parent"):
        r = resolve_range(fixture_repo.path, fixture_repo.sha("feature_square"), BAD_TAG)
    assert list(r.shas) == [fixture_repo.sha(n) for n in ("docs", "merge", "slow_add")]


def test_large_range_warns(fixture_repo):
    with pytest.warns(RangeWarning, match="7 commits"):
        resolve_range(fixture_repo.path, GOOD_TAG, BAD_TAG, warn_threshold=5)


def test_max_commits_is_enforced(fixture_repo):
    with pytest.raises(InvalidRangeError, match="max_commits"):
        resolve_range(fixture_repo.path, GOOD_TAG, BAD_TAG, max_commits=5)


def test_shallow_clone_warns(fixture_repo, tmp_path):
    import subprocess

    clone = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "3", fixture_repo.path.as_uri(), str(clone)],
        check=True, capture_output=True,
    )
    with pytest.warns(RangeWarning, match="shallow"):
        resolve_range(clone, "HEAD~2", "HEAD")


def test_does_not_modify_the_repo(fixture_repo):
    before = fixture_repo.git("status", "--porcelain")
    head_before = fixture_repo.git("rev-parse", "HEAD")
    resolve_quietly(fixture_repo.path, GOOD_TAG, BAD_TAG)
    assert fixture_repo.git("status", "--porcelain") == before
    assert fixture_repo.git("rev-parse", "HEAD") == head_before
