import sqlite3
import subprocess
import time

import pytest

from perfhound.gateway import Gateway
from perfhound.gateway.analyzer import CodeAnalyzer
from perfhound.gateway.api import repo_identity
from perfhound.gateway.cache import Cache, CacheWarning
from perfhound.gateway.local_git import LocalGitProvider

from .fixture_repo import BAD_TAG, GOOD_TAG


@pytest.fixture
def cache(tmp_path):
    c = Cache(tmp_path / "cache.db")
    yield c
    c.close()


def forbid_git_reads(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should have come from the cache")

    monkeypatch.setattr(LocalGitProvider, "get_commits", boom)
    monkeypatch.setattr(CodeAnalyzer, "analyze", boom)


def test_second_run_comes_entirely_from_cache(fixture_repo, cache, monkeypatch):
    first = Gateway(fixture_repo.path, cache=cache).get_candidates(GOOD_TAG, BAD_TAG)
    gw = Gateway(fixture_repo.path, cache=cache)
    assert gw.last_stats is None

    forbid_git_reads(monkeypatch)
    second = gw.get_candidates(GOOD_TAG, BAD_TAG)
    assert second == first
    assert (gw.last_stats.commits, gw.last_stats.commit_cache_hits, gw.last_stats.function_cache_hits) == (7, 7, 7)


def test_only_new_commits_are_read(fresh_fixture_repo, cache, monkeypatch):
    repo = fresh_fixture_repo
    Gateway(repo.path, cache=cache).get_candidates(GOOD_TAG, BAD_TAG)
    for i in range(2):
        repo.git("commit", "-q", "--allow-empty", "-m", f"new {i}")

    asked = []
    real = LocalGitProvider.get_commits
    monkeypatch.setattr(LocalGitProvider, "get_commits",
                        lambda self, r, shas=None: asked.append(list(shas)) or real(self, r, shas))
    gw = Gateway(repo.path, cache=cache)
    cs = gw.get_candidates(GOOD_TAG, "HEAD")
    assert len(cs) == 9 and gw.last_stats.commit_cache_hits == 7
    assert asked == [[cs[7].sha, cs[8].sha]]


def test_positions_follow_the_requested_range_not_the_cache(fixture_repo, cache):
    Gateway(fixture_repo.path, cache=cache).get_candidates(GOOD_TAG, BAD_TAG)
    cached = Gateway(fixture_repo.path, cache=cache).get_candidates(fixture_repo.sha("rename"), BAD_TAG)
    uncached = Gateway(fixture_repo.path, cache=False).get_candidates(fixture_repo.sha("rename"), BAD_TAG)
    assert [c.position for c in cached] == [0, 1, 2, 3, 4]
    assert cached == uncached


def test_different_diff_limit_is_a_miss_not_a_stale_hit(fixture_repo, cache):
    Gateway(fixture_repo.path, cache=cache).get_candidates(GOOD_TAG, BAD_TAG)
    gw = Gateway(fixture_repo.path, cache=cache, max_diff_lines=3)
    cs = gw.get_candidates(GOOD_TAG, BAD_TAG)
    assert gw.last_stats.commit_cache_hits == 0
    assert cs[-1].diff_truncated


def test_cache_disabled(fixture_repo, tmp_path):
    gw = Gateway(fixture_repo.path, cache=False)
    assert len(gw.get_candidates(GOOD_TAG, BAD_TAG)) == 7
    assert gw.last_stats.commit_cache_hits == 0


def test_corrupt_database_warns_and_gateway_still_works(fixture_repo, tmp_path):
    db = tmp_path / "cache.db"
    db.write_bytes(b"this is not a sqlite database" * 100)
    with pytest.warns(CacheWarning):
        gw = Gateway(fixture_repo.path, cache=db)
    assert len(gw.get_candidates(GOOD_TAG, BAD_TAG)) == 7
    assert gw.cache.disabled


def test_damaged_entry_is_treated_as_miss(fixture_repo, cache):
    gw = Gateway(fixture_repo.path, cache=cache)
    good = gw.get_candidates(GOOD_TAG, BAD_TAG)
    cache._conn.execute("UPDATE entries SET payload = '{\"oops\": 1}'")
    cache._conn.commit()
    again = Gateway(fixture_repo.path, cache=cache)
    assert again.get_candidates(GOOD_TAG, BAD_TAG) == good
    assert again.last_stats.commit_cache_hits == 0


def test_old_cache_schema_is_dropped(tmp_path):
    db = tmp_path / "cache.db"
    with Cache(db) as c:
        c.put_many("r", {"a" * 40: {"x": 1}}, "k")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value='0' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with Cache(db) as c:
        assert c.count() == 0


def test_volatile_entries_expire_but_keep_etag(tmp_path):
    with Cache(tmp_path / "c.db", pr_ttl=0.2) as c:
        assert c.get_volatile("r", "pr:1", "pr") == (None, False)
        c.put_volatile("r", "pr:1", "pr", {"title": "Speed up"}, etag='W/"abc"')
        value, fresh = c.get_volatile("r", "pr:1", "pr")
        assert fresh and value.payload == {"title": "Speed up"} and value.etag == 'W/"abc"'
        time.sleep(0.3)
        value, fresh = c.get_volatile("r", "pr:1", "pr")
        assert not fresh and value.etag == 'W/"abc"'   # stale, but ETag usable for a free 304


def test_many_keys_are_chunked(tmp_path):
    with Cache(tmp_path / "c.db") as c:
        items = {f"{i:040x}": {"i": i} for i in range(1200)}
        c.put_many("r", items, "k")
        got = c.get_many("r", list(items), "k")
        assert got == items


def test_repo_identity_survives_clone_and_differs_between_repos(fixture_repo, tmp_path):
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(fixture_repo.path), str(clone)], check=True)
    assert repo_identity(clone) == repo_identity(fixture_repo.path)
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    subprocess.run(["git", "-C", str(other), "-c", "user.name=x", "-c", "user.email=x@x",
                    "commit", "-q", "--allow-empty", "-m", "root"], check=True)
    assert repo_identity(other) != repo_identity(fixture_repo.path)
