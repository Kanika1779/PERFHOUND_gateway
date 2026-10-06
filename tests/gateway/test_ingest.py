"""Ingest: repository link -> folder of JSON. Git side against the fixture repo (served over file://,
like GitHub serves https); GitHub side against a FAKE GitHub (no network, no token)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from perfhound.gateway.errors import GatewayError
from perfhound.gateway.fetcher import RepoFetcher
from perfhound.gateway.github import RepoRef
from perfhound.gateway.ingest import ingest, load, resolve_link
from perfhound.gateway.local_git import parse_git_date, read_commit_info

from .fixture_repo import BAD_TAG, GOOD_TAG
from .test_github import FakeGitHub, client

SHOP = RepoRef("acme", "shop")
T0 = datetime(2025, 10, 4, 4, 0, tzinfo=timezone.utc)          # fixture repo: first commit
Z = lambda dt: dt.isoformat().replace("+00:00", "Z")


@pytest.fixture
def remote(fresh_fixture_repo):
    r = fresh_fixture_repo
    r.git("config", "uploadpack.allowFilter", "true")
    r.git("config", "uploadpack.allowAnySHA1InWant", "true")
    r.url = r.path.as_uri()
    return r


def lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run(remote, tmp_path, **kw):
    kw.setdefault("github", False)
    return ingest(remote.url, tmp_path / "out", fetcher=RepoFetcher(tmp_path / "repos"), cache=False,
                  log=lambda s: None, **kw)


# -- links ------------------------------------------------------------------------------------

def test_link_forms():
    for link in ("https://github.com/acme/shop", "https://github.com/acme/shop.git",
                 "https://github.com/acme/shop/tree/main/src", "git@github.com:acme/shop.git", "acme/shop"):
        assert resolve_link(link) == ("https://github.com/acme/shop", SHOP), link
    assert resolve_link("https://gitlab.com/acme/shop.git") == ("https://gitlab.com/acme/shop.git", None)
    for bad in ("", "not a link", "ftp://example.com/x.git", "https://github.com/acme"):
        with pytest.raises(GatewayError):
            resolve_link(bad)


# -- git side ---------------------------------------------------------------------------------

def test_every_commit_is_stored_with_diff_functions_and_parents(remote, tmp_path):
    res = run(remote, tmp_path)
    commits = lines(res.folder / "commits.jsonl")
    repo = json.loads((res.folder / "repo.json").read_text(encoding="utf-8"))
    every = remote.git("rev-list", "HEAD").split()
    root = remote.sha("initial")
    assert len(commits) == len(every) - 1 == res.commits                       # all but the root commit
    assert [r["sha"] for r in repo["root_commits"]] == [root]
    assert [c["position"] for c in commits] == list(range(len(commits)))
    seen = {root}
    for c in commits:                                                         # parents before children
        assert set(c["parents"]) <= seen and c["parent"] == c["parents"][0]
        seen.add(c["sha"])
    by_sha = {c["sha"]: c for c in commits}
    tweak = by_sha[remote.sha("tweak_add")]
    assert "int(a)" in tweak["diff"] and tweak["changed_functions"] == ["calc.add"]
    assert by_sha[remote.sha("feature_square")]["mainline"] is False           # came in through the merge
    assert sum(c["mainline"] for c in commits) == len(remote.git("rev-list", "--first-parent", "HEAD").split()) - 1
    assert len(by_sha[remote.sha("merge")]["parents"]) == 2
    assert repo["tags"][GOOD_TAG] == root and repo["tags"][BAD_TAG] == remote.sha("slow_add")
    assert repo["head"] == remote.git("rev-parse", "HEAD") and repo["complete"]["commits"] is True


def test_limit_keeps_the_newest_commits(remote, tmp_path):
    res = run(remote, tmp_path, limit=3)
    commits = lines(res.folder / "commits.jsonl")
    newest = remote.git("log", "--topo-order", "--format=%H", "-3").split()
    assert [c["sha"] for c in commits] == newest[::-1]


def test_rerun_picks_up_new_upstream_commits(remote, tmp_path):
    run(remote, tmp_path)
    remote.git("commit", "-q", "--allow-empty", "-m", "pushed later")
    res = run(remote, tmp_path)
    assert lines(res.folder / "commits.jsonl")[-1]["message"] == "pushed later"


def test_broken_time_zones_in_old_history(remote, tmp_path):
    """requests (2011) has commits with time zone "+051800"; git prints it as "+518:00"."""
    parent = remote.git("rev-parse", "HEAD")
    tree = remote.git("rev-parse", "HEAD^{tree}")
    raw = (f"tree {tree}\nparent {parent}\nauthor A <a@x> 1313584730 +051800\n"
           f"committer A <a@x> 1313584730 +051800\n\nodd time zone\n")
    import subprocess
    sha = subprocess.run(["git", "hash-object", "-w", "-t", "commit", "--literally", "--stdin"], cwd=remote.path,
                         input=raw.encode(), capture_output=True, check=True).stdout.decode().strip()
    remote.git("update-ref", "refs/heads/" + remote.git("symbolic-ref", "--short", "HEAD"), sha)
    [info] = read_commit_info(remote.path, [sha])
    assert info.timestamp == datetime.fromtimestamp(1313584730, timezone.utc)
    res = run(remote, tmp_path)
    assert lines(res.folder / "commits.jsonl")[-1]["committed_at"] == "2011-08-17T12:38:50+00:00"
    assert parse_git_date("2024-03-01T10:00:00+05:30", "1709267400").utcoffset() == timedelta(hours=5, minutes=30)


# -- GitHub side ------------------------------------------------------------------------------

def gh_pull(n, merged_sha=None, created=T0, merged=None, updated=None):
    return {"number": n, "title": f"PR {n}", "body": f"Fixes #{n + 100}", "state": "closed" if merged else "open",
            "labels": [{"name": "perf"}], "user": {"login": "dev"}, "created_at": Z(created),
            "updated_at": Z(updated or created), "closed_at": Z(merged) if merged else None,
            "merged_at": Z(merged) if merged else None, "merge_commit_sha": merged_sha,
            "base": {"ref": "main"}, "head": {"ref": f"b{n}"}, "html_url": f"https://github.com/acme/shop/pull/{n}"}


def gh_issue(n, created=T0, closed=None, updated=None, comments=0):
    return {"number": n, "title": f"Issue {n}", "body": "it is slow", "state": "closed" if closed else "open",
            "labels": [], "user": {"login": "user"}, "created_at": Z(created), "updated_at": Z(updated or created),
            "closed_at": Z(closed) if closed else None, "comments": comments,
            "html_url": f"https://github.com/acme/shop/issues/{n}"}


def page_links(page, last):
    nxt = f'<https://api.github.com/repos/acme/shop/pulls?page={page + 1}>; rel="next", ' if page < last else ""
    return {"Link": nxt + f'<https://api.github.com/repos/acme/shop/pulls?page={last}>; rel="last"'}


def fake_shop(remote):
    fake = FakeGitHub()
    fake.add("/repos/acme/shop", {"full_name": "acme/shop", "default_branch": "main", "stargazers_count": 3})
    merge = remote.sha("merge")
    fake.add("/repos/acme/shop/pulls", [gh_pull(1, merge, merged=T0 + timedelta(hours=7)), gh_pull(2)],
             headers=page_links(1, 2))
    fake.add("/repos/acme/shop/pulls", [gh_pull(3, created=T0 + timedelta(days=30))], headers=page_links(2, 2))
    fake.add("/repos/acme/shop/issues", [gh_issue(101, comments=1), gh_issue(102, created=T0 + timedelta(days=40)),
                                         {**gh_issue(2), "pull_request": {"url": "..."}}])
    fake.add("/repos/acme/shop/issues/101/comments",
             [{"user": {"login": "maint"}, "created_at": Z(T0 + timedelta(hours=1)), "body": "confirmed"}])
    return fake


def test_github_prs_issues_and_comments_are_stored(remote, tmp_path):
    fake = fake_shop(remote)
    res = run(remote, tmp_path, github=True, github_repo=SHOP, client=client(fake), comments=True)
    pulls = lines(res.folder / "pull_requests.jsonl")
    issues = lines(res.folder / "issues.jsonl")
    assert [p["number"] for p in pulls] == [1, 2, 3]                            # both pages
    assert [i["number"] for i in issues] == [101, 102]                          # the PR in the issues list: dropped
    assert issues[0]["comment_list"] == [{"author": "maint", "created_at": Z(T0 + timedelta(hours=1)),
                                          "body": "confirmed"}]
    merge = {c["sha"]: c for c in lines(res.folder / "commits.jsonl")}[remote.sha("merge")]
    assert merge["pr"]["number"] == 1 and merge["source"] == "local+github"     # commit -> its PR
    repo = json.loads((res.folder / "repo.json").read_text(encoding="utf-8"))
    assert res.complete and repo["complete"]["github"] is True and repo["github_repo"]["stargazers_count"] == 3
    assert "last_complete_at" in repo["github_sync"]


def test_rate_limit_keeps_what_it_got_and_the_next_run_continues(remote, tmp_path):
    fake = FakeGitHub()
    fake.add("/repos/acme/shop", {"full_name": "acme/shop"})
    fake.add("/repos/acme/shop/pulls", [gh_pull(1), gh_pull(2)], headers=page_links(1, 2))
    fake.add("/repos/acme/shop/pulls", {"message": "API rate limit exceeded"}, status=403,
             headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1900000000"})
    res = run(remote, tmp_path, github=True, github_repo=SHOP, client=client(fake))
    assert not res.complete and any("limit" in w for w in res.warnings)
    assert [p["number"] for p in lines(res.folder / "pull_requests.jsonl")] == [1, 2]
    repo = json.loads((res.folder / "repo.json").read_text(encoding="utf-8"))
    assert repo["github_sync"]["pulls_next_page"] == 2 and repo["complete"]["github"] is False
    assert len(lines(res.folder / "commits.jsonl")) == res.commits > 0         # git data is complete anyway

    fake = FakeGitHub()                                                         # an hour later
    fake.add("/repos/acme/shop", {"full_name": "acme/shop"})
    fake.add("/repos/acme/shop/pulls", [gh_pull(3)], headers=page_links(2, 2))
    fake.add("/repos/acme/shop/issues", [gh_issue(101)])
    res = run(remote, tmp_path, github=True, github_repo=SHOP, client=client(fake))
    assert res.complete and [p["number"] for p in lines(res.folder / "pull_requests.jsonl")] == [1, 2, 3]
    assert "page=2" in fake.calls[1][0]                                         # resumed, not restarted

    fake = FakeGitHub()                                                         # later still: incremental
    fake.add("/repos/acme/shop", {"full_name": "acme/shop"})
    fake.add("/repos/acme/shop/pulls", [gh_pull(2, updated=datetime.now(timezone.utc)),
                                        gh_pull(1, updated=T0)])
    fake.add("/repos/acme/shop/issues", [gh_issue(103)])
    res = run(remote, tmp_path, github=True, github_repo=SHOP, client=client(fake))
    assert res.complete and res.pull_requests == 3 and res.issues == 2
    assert "sort=updated" in fake.calls[1][0] and "since=" in fake.calls[2][0]


def test_load_until_hides_the_future(remote, tmp_path):
    fake = fake_shop(remote)
    res = run(remote, tmp_path, github=True, github_repo=SHOP, client=client(fake), comments=True)
    merge = remote.sha("merge")
    landed = next(c for c in lines(res.folder / "commits.jsonl") if c["sha"] == merge)["committed_at"]
    until = datetime.fromisoformat(landed)
    view = load(res.folder, until=until)
    assert view["commits"][-1]["sha"] == merge                                  # nothing committed after `until`
    assert [p["number"] for p in view["pull_requests"]] == [1, 2]               # PR 3 opened 30 days later
    assert [i["number"] for i in view["issues"]] == [101]                       # issue 102 opened later
    full = load(res.folder)
    assert len(full["commits"]) > len(view["commits"]) and len(full["issues"]) == 2
    early = load(res.folder, until=T0 + timedelta(minutes=30))
    assert early["issues"][0]["comment_list"] == [] and early["issues"][0]["comments"] == 0
    assert all(p["merged_at"] is None and p["state"] == "open" for p in early["pull_requests"])


def test_token_commands_dispatch_without_a_link(monkeypatch, capsys):
    from perfhound.gateway import __main__ as cli
    calls = []
    monkeypatch.setattr(cli, "token_command", lambda action, token=None: calls.append((action, token)) or 0)
    assert cli.main(["login", "--token", "ghp_x"]) == 0 and cli.main(["whoami"]) == 0
    assert calls == [("login", "ghp_x"), ("whoami", None)]
    assert cli.main(["not a link"]) == 1 and "not a repository link" in capsys.readouterr().err


def test_newer_git_prints_utc_as_z():
    assert parse_git_date("2026-10-06T08:02:46Z", "1791273766") == datetime(2026, 10, 6, 8, 2, 46, tzinfo=timezone.utc)
    assert parse_git_date("2026-10-06T08:02:46Z").utcoffset() == timedelta(0)
