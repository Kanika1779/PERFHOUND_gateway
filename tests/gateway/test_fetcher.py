import re
import subprocess
from pathlib import Path

import pytest

from perfhound.gateway import Gateway, RegressionCase
from perfhound.gateway import local_git
from perfhound.gateway.errors import GatewayError
from perfhound.gateway.fetcher import FetchError, RepoFetcher, local_dir_for, validate_url
from perfhound.gateway.gitcmd import is_partial_clone

from .fixture_repo import BAD_TAG, GOOD_TAG


@pytest.fixture
def remote(fresh_fixture_repo):
    """A fixture repo that serves partial clones over file://, like GitHub does over https."""
    r = fresh_fixture_repo
    r.git("config", "uploadpack.allowFilter", "true")
    r.git("config", "uploadpack.allowAnySHA1InWant", "true")
    r.url = r.path.as_uri()
    return r


@pytest.mark.parametrize("url", [
    "ext::sh -c touch% /tmp/pwned", "fd::17", "-oProxyCommand=evil", "ftp://example.com/x.git",
])
def test_dangerous_or_unknown_urls_are_refused(url):
    with pytest.raises(FetchError):
        validate_url(url)


@pytest.mark.parametrize("url", [
    "https://github.com/pandas-dev/pandas", "https://github.com/pandas-dev/pandas.git",
    "git@github.com:pandas-dev/pandas.git", "ssh://git@github.com/pandas-dev/pandas",
])
def test_github_urls_map_to_one_folder(url, tmp_path):
    validate_url(url)
    assert local_dir_for(url, tmp_path) == tmp_path / "github.com" / "pandas-dev" / "pandas"


def test_local_path_is_used_as_is(fixture_repo, tmp_path):
    f = RepoFetcher(tmp_path / "repos")
    assert f.fetch(str(fixture_repo.path)) == fixture_repo.path
    assert not (tmp_path / "repos").exists()
    with pytest.raises(FetchError):
        f.fetch(str(tmp_path / "missing"))


def test_clone_is_partial_without_checkout_and_reused(remote, tmp_path):
    f = RepoFetcher(tmp_path / "repos")
    path = f.fetch(remote.url)
    assert is_partial_clone(path)
    assert [p.name for p in path.iterdir()] == [".git"]            # no working files
    assert f.fetch(remote.url) == path                              # second call: no new clone
    assert not [p for p in path.parent.iterdir() if ".tmp-" in p.name]


def test_failed_clone_leaves_nothing(tmp_path):
    f = RepoFetcher(tmp_path / "repos")
    (tmp_path / "repos").mkdir()
    with pytest.raises(FetchError, match="could not clone"):
        f.fetch((tmp_path / "no_such_repo").as_uri())
    assert not [p for p in (tmp_path / "repos").rglob("*") if ".tmp-" in p.name]


def test_ensure_commits_fetches_new_history(remote, tmp_path):
    f = RepoFetcher(tmp_path / "repos")
    path = f.fetch(remote.url)
    remote.git("commit", "-q", "--allow-empty", "-m", "pushed later")
    new_sha = remote.git("rev-parse", "HEAD")
    assert f.ensure_commits(path, [GOOD_TAG, new_sha]) == []
    assert f.ensure_commits(path, ["0" * 40]) == ["0" * 40]


def _lazy_fetches(trace: Path) -> int:
    if not trace.exists():
        return 0
    return len(re.findall(r"run_command: .*fetch origin", trace.read_text(encoding="utf-8", errors="replace")))


@pytest.mark.parametrize("prefetch", [True, False])
def test_partial_clone_blobs_come_in_one_batch(remote, tmp_path, monkeypatch, prefetch):
    case = RegressionCase(case_id="t:1", source="test", repo=remote.url, language="python",
                          good=GOOD_TAG, bad=BAD_TAG)
    gw = Gateway.for_case(case, fetcher=RepoFetcher(tmp_path / "repos"), cache=False)
    if not prefetch:   # control experiment: prove the test can see lazy fetches
        monkeypatch.setattr(local_git, "prefetch_blobs", lambda repo, shas: 0)
    trace = tmp_path / "git_trace.txt"
    monkeypatch.setenv("GIT_TRACE", str(trace))
    got = gw.candidates_for(case)
    monkeypatch.delenv("GIT_TRACE")

    expected = Gateway(remote.path, cache=False).get_candidates(GOOD_TAG, BAD_TAG)
    assert got == expected                       # same data as from the full repo
    if prefetch:
        assert _lazy_fetches(trace) == 0
    else:
        assert _lazy_fetches(trace) > 1


def test_for_case_reports_missing_commits(remote, tmp_path):
    case = RegressionCase(case_id="t:2", source="test", repo=remote.url, language="python",
                          good=GOOD_TAG, bad="f" * 40)
    with pytest.raises(GatewayError, match="not found"):
        Gateway.for_case(case, fetcher=RepoFetcher(tmp_path / "repos"), cache=False)


def test_stale_half_clones_are_removed(remote, tmp_path):
    import os, time
    f = RepoFetcher(tmp_path / "repos")
    dest = local_dir_for(remote.url, tmp_path / "repos")
    dest.parent.mkdir(parents=True)
    old = dest.parent / f".{dest.name}.tmp-dead0001"
    young = dest.parent / f".{dest.name}.tmp-busy0002"
    for p in (old, young):
        (p / "objects").mkdir(parents=True)
    os.utime(old, (time.time() - 7 * 3600,) * 2)
    f.fetch(remote.url)
    assert not old.exists() and young.exists()


def test_relative_base_dir_works(remote, tmp_path, monkeypatch):
    """Found on the first real multi-repo run: base_dir="repos" cloned into repos/repos/..."""
    monkeypatch.chdir(tmp_path)
    path = RepoFetcher("repos").fetch(remote.url)
    assert path.is_absolute() and (path / ".git").exists()
    assert path == (tmp_path / "repos").resolve() / path.relative_to((tmp_path / "repos").resolve())


# -- clone folders never leave the repos folder ------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://github.com/../../../Documents", "https://github.com/acme/..", "https://github.com/./x",
    "git@github.com:../../Documents", "https://../outside", "https://github.com/acme/.../x",
])
def test_urls_with_dot_segments_are_refused(url, tmp_path):
    with pytest.raises(FetchError):
        local_dir_for(url, tmp_path / "repos")


def test_remove_never_deletes_outside_the_repos_folder(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "notes.txt").write_text("important")
    (tmp_path / "repos" / "github.com").mkdir(parents=True)     # exists after any GitHub clone
    with pytest.raises(FetchError):
        RepoFetcher(tmp_path / "repos").remove("https://github.com/../../victim")
    assert (victim / "notes.txt").read_text() == "important"


def test_file_urls_still_work_with_dots_in_the_path(remote, tmp_path):
    (remote.path.parent / "x").mkdir()
    odd = remote.path.parent / "x" / ".." / remote.path.name     # file URL paths are hashed, never used
    path = RepoFetcher(tmp_path / "repos").fetch("file://" + odd.as_posix())
    assert (path / ".git").exists() and path.is_relative_to((tmp_path / "repos").resolve())



@pytest.mark.parametrize("url", ["file://C:/Users/me/x/../repo",      # Windows: drive in the host slot
                                 "file://server/share/../repo"])      # UNC-style host
def test_file_urls_with_a_host_are_hashed_too(url, tmp_path):
    dest = local_dir_for(url, tmp_path / "repos")
    assert dest.parent.name == "local" and dest.name.startswith("_")

# -- clones follow upstream ---------------------------------------------------------------------

def _branch(remote):
    return remote.git("symbolic-ref", "--short", "HEAD")


def test_branch_names_follow_upstream(remote, tmp_path):
    f = RepoFetcher(tmp_path / "repos", refresh_seconds=0)
    path = f.fetch(remote.url)
    branch = _branch(remote)
    remote.git("commit", "-q", "--allow-empty", "-m", "pushed later")
    new_sha = remote.git("rev-parse", "HEAD")
    assert run_git_out(path, "rev-parse", branch) != new_sha          # the clone is behind
    assert f.ensure_commits(path, [GOOD_TAG, branch]) == []
    assert run_git_out(path, "rev-parse", branch) == new_sha          # ... until a name is asked for
    assert run_git_out(path, "rev-parse", "HEAD") == new_sha


def test_moved_tags_follow_upstream(remote, tmp_path):
    f = RepoFetcher(tmp_path / "repos", refresh_seconds=0)
    path = f.fetch(remote.url)
    remote.git("commit", "-q", "--allow-empty", "-m", "release")
    remote.git("tag", "-f", BAD_TAG)
    assert f.ensure_commits(path, [BAD_TAG]) == []
    assert run_git_out(path, "rev-parse", BAD_TAG + "^{commit}") == remote.git("rev-parse", "HEAD")


def test_refresh_is_rate_limited_and_full_shas_need_no_network(remote, tmp_path, monkeypatch):
    f = RepoFetcher(tmp_path / "repos", refresh_seconds=3600)
    path = f.fetch(remote.url)
    sha = run_git_out(path, "rev-parse", GOOD_TAG + "^{commit}")
    calls = []
    real_update = RepoFetcher.update
    monkeypatch.setattr(RepoFetcher, "update", lambda self, p, **kw: calls.append(kw) or real_update(self, p, **kw))
    assert f.ensure_commits(path, [sha]) == [] and calls == []        # full SHA present: no network
    assert f.update(path) is False                                    # just cloned: still fresh
    assert f.update(path, max_age=0) is True


def test_local_only_branches_survive_a_refresh(remote, tmp_path):
    f = RepoFetcher(tmp_path / "repos", refresh_seconds=0)
    path = f.fetch(remote.url)
    run_git_out(path, "branch", "perfhound/case-1", GOOD_TAG)
    remote.git("commit", "-q", "--allow-empty", "-m", "pushed later")
    assert f.update(path, max_age=0)
    assert run_git_out(path, "rev-parse", "perfhound/case-1") == run_git_out(path, "rev-parse", GOOD_TAG + "^{commit}")


def test_offline_refresh_warns_and_keeps_the_copy(remote, tmp_path):
    from perfhound.gateway.fetcher import FetchWarning
    f = RepoFetcher(tmp_path / "repos", refresh_seconds=0)
    path = f.fetch(remote.url)
    branch = _branch(remote)
    before = run_git_out(path, "rev-parse", branch)
    run_git_out(path, "remote", "set-url", "origin", (tmp_path / "gone").as_uri())
    with pytest.warns(FetchWarning, match="existing copy"):
        assert f.update(path, max_age=0) is False
    with pytest.warns(FetchWarning):
        assert f.ensure_commits(path, [branch]) == []
    assert run_git_out(path, "rev-parse", branch) == before


def test_users_own_repo_is_never_refreshed(fresh_fixture_repo, tmp_path):
    f = RepoFetcher(tmp_path / "repos")
    assert not f.is_managed(fresh_fixture_repo.path)
    with pytest.raises(FetchError, match="not a clone made by RepoFetcher"):
        f.update(fresh_fixture_repo.path)


def run_git_out(path, *args):
    from perfhound.gateway.gitcmd import git_out
    return git_out(path, *args)
