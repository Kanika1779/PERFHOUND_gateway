"""Range fetch (good..bad only, live) + GraphQL batch lookups, against the fixture repo and a FAKE
GitHub (no network, no token)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from perfhound.gateway import Gateway
from perfhound.gateway.errors import GatewayError
from perfhound.gateway.fetcher import RepoFetcher
from perfhound.gateway.github import GitHubClient, GitHubGraphQLProvider, RepoRef
from perfhound.gateway.models import CandidateCommit
from perfhound.gateway.range_fetch import fetch_range, resolve_input, summary_markdown

from .fixture_repo import BAD_TAG, GOOD_TAG

SHOP = RepoRef("acme", "shop")
T0 = datetime(2025, 10, 4, 4, 0, tzinfo=timezone.utc)
Z = lambda dt: dt.isoformat().replace("+00:00", "Z")
quiet = lambda s: None


class FakeGraphQL:
    """transport(method, url, headers, timeout, body=None) answering POST /graphql from a sha -> PRs table."""

    def __init__(self, prs_by_sha=None, error=None):
        self.prs_by_sha = prs_by_sha or {}
        self.error = error
        self.calls = []

    def __call__(self, method, url, headers, timeout, body=None):
        q = json.loads(body)
        self.calls.append(q)
        if self.error:
            return 200, {}, json.dumps({"data": {"repository": None}, "errors": [self.error]}).encode()
        import re
        repo = {}
        for alias, sha in re.findall(r'(c\d+): object\(oid: "([0-9a-f]{40})"\)', q["query"]):
            prs = self.prs_by_sha.get(sha)
            repo[alias] = None if prs is None else {"associatedPullRequests": {"nodes": prs}}
        return 200, {}, json.dumps({"data": {"repository": repo}}).encode()


def gql_pr(number, merge_sha, merged, issues=(), title=None):
    return {"number": number, "title": title or f"PR {number}", "body": "details",
            "url": f"https://github.com/acme/shop/pull/{number}", "mergedAt": Z(merged),
            "mergeCommit": {"oid": merge_sha}, "labels": {"nodes": [{"name": "perf"}]},
            "closingIssuesReferences": {"nodes": [
                {"number": n, "title": t, "body": "it is slow", "createdAt": Z(c), "url": "u"} for n, t, c in issues]}}


def commit(sha, hours):
    return CandidateCommit(sha=sha, parent="0" * 40, position=hours, message="m", author="dev",
                           timestamp=T0 + timedelta(hours=hours))


def gql_client(fake, token="ghp_test"):
    return GitHubClient(token, transport=fake, sleep=lambda s: None)


# -- input forms ------------------------------------------------------------------------------

def test_inputs(tmp_path):
    assert resolve_input("https://github.com/acme/shop/compare/v1.0...main", None, None) == \
        ("https://github.com/acme/shop", "v1.0", "main", SHOP)
    assert resolve_input("https://github.com/acme/shop", "a", "b") == ("https://github.com/acme/shop", "a", "b", SHOP)
    assert resolve_input(str(tmp_path), "a", None) == (str(tmp_path), "a", None, None)       # a folder: as-is
    with pytest.raises(GatewayError):
        resolve_input("not a link", None, None)


# -- git side ---------------------------------------------------------------------------------

def test_range_of_a_folder_on_disk(fixture_repo, tmp_path):
    out = tmp_path / "r.json"
    res = fetch_range(str(fixture_repo.path), GOOD_TAG, BAD_TAG, out=out, github=False, cache=False, log=quiet)
    assert json.loads(out.read_text(encoding="utf-8")) == res
    assert [c["sha"] for c in res["commits"]] == fixture_repo.first_parent_shas()
    assert res["good"]["sha"] == fixture_repo.sha("initial") and res["bad"]["sha"] == fixture_repo.sha("slow_add")
    assert res["mode"] == "real" and res["count"] == 7 and all(c["committed_at"] for c in res["commits"])
    slow = res["commits"][-1]
    assert slow["changed_functions"] == ["mathops.add"] and "for" in slow["diff"]
    md = summary_markdown(res)
    assert "7 commits between `v1.0` and `v1.1`" in md and "`mathops.add`" in md


def test_range_from_a_link_is_cloned_and_kept_current(fresh_fixture_repo, tmp_path):
    r = fresh_fixture_repo
    r.git("config", "uploadpack.allowFilter", "true")
    r.git("config", "uploadpack.allowAnySHA1InWant", "true")
    f = RepoFetcher(tmp_path / "repos", refresh_seconds=0)
    branch = r.git("symbolic-ref", "--short", "HEAD")
    res = fetch_range(r.path.as_uri(), GOOD_TAG, branch, out=None, github=False, fetcher=f, cache=False, log=quiet)
    assert res["count"] == 7
    r.git("commit", "-q", "--allow-empty", "-m", "pushed later")
    res = fetch_range(r.path.as_uri(), GOOD_TAG, branch, out=None, github=False, fetcher=f, cache=False, log=quiet)
    assert res["count"] == 8 and res["commits"][-1]["message"] == "pushed later"


def test_no_good_commit_means_just_bad(fixture_repo):
    zero = "0" * 40                                      # GitHub's "before" on the first push of a branch
    res = fetch_range(str(fixture_repo.path), zero, BAD_TAG, out=None, github=False, cache=False, log=quiet)
    assert res["count"] == 1 and res["commits"][0]["sha"] == fixture_repo.sha("slow_add")
    assert any("no good commit" in w for w in res["warnings"])


def test_pull_request_range_starts_at_the_merge_base(fixture_repo):
    # 'feature' branched off before 'docs'; from the merge base only feature_square is left
    res = fetch_range(str(fixture_repo.path), fixture_repo.sha("docs"), fixture_repo.sha("feature_square"),
                      out=None, github=False, cache=False, from_merge_base=True, log=quiet)
    assert [c["sha"] for c in res["commits"]] == [fixture_repo.sha("feature_square")]


def test_big_ranges_are_processed_in_pieces(fixture_repo):
    whole = Gateway(fixture_repo.path, cache=False).get_candidates(GOOD_TAG, BAD_TAG)
    pieces = Gateway(fixture_repo.path, cache=False, batch_size=2).get_candidates(GOOD_TAG, BAD_TAG)
    assert pieces == whole and [c.position for c in pieces] == list(range(7))


# -- GitHub: GraphQL, many commits per request ---------------------------------------------------

def test_graphql_batches_commits_and_brings_linked_issues(tmp_path):
    a, b, c = "a" * 40, "b" * 40, "c" * 40
    fake = FakeGraphQL({
        a: [gql_pr(10, a, T0 + timedelta(hours=1), issues=[(45, "Checkout is slow", T0 - timedelta(days=2))])],
        b: [],
        # c: unknown to GitHub (not pushed) -> None
    })
    prov = GitHubGraphQLProvider(gql_client(fake), per_request=2)
    out, st = prov.enrich(SHOP, "r", [commit(a, 1), commit(b, 2), commit(c, 3)])
    assert len(fake.calls) == 2                                          # 3 commits, 2 per request
    assert fake.calls[0]["variables"] == {"owner": "acme", "name": "shop"}
    assert out[0].pr.number == 10 and out[0].pr.labels == ("perf",)
    assert "Issue #45: Checkout is slow" in out[0].pr.body and "it is slow" in out[0].pr.body
    assert out[1].pr is None and out[2].pr is None and st.with_pr == 1 and st.issues_used == 1


def test_graphql_real_mode_keeps_everything_evaluation_mode_hides_the_future():
    a = "a" * 40
    late_issue = (46, "Slow since #10", T0 + timedelta(days=5))
    fake = FakeGraphQL({a: [gql_pr(10, a, T0, issues=[late_issue]),
                            gql_pr(12, "f" * 40, T0 + timedelta(days=9), title="Revert #10")]})
    real, _ = GitHubGraphQLProvider(gql_client(fake)).enrich(SHOP, "r", [commit(a, 1)], cutoff=None)
    assert "Slow since #10" in real[0].pr.body                          # real users get every clue
    ev, st = GitHubGraphQLProvider(gql_client(fake)).enrich(SHOP, "r", [commit(a, 1)], cutoff=T0 + timedelta(hours=2))
    assert ev[0].pr.number == 10 and "Slow since" not in ev[0].pr.body
    assert st.issues_skipped_after_cutoff == 1


def test_graphql_errors_never_block(tmp_path):
    a = "a" * 40
    fake = FakeGraphQL(error={"type": "NOT_FOUND", "message": "Could not resolve to a Repository"})
    out, st = GitHubGraphQLProvider(gql_client(fake)).enrich(SHOP, "r", [commit(a, 1)])
    assert out[0].pr is None and "does not know acme/shop" in st.warnings[0]
    fake = FakeGraphQL(error={"type": "RATE_LIMITED", "message": "API rate limit exceeded"})
    out, st = GitHubGraphQLProvider(gql_client(fake)).enrich(SHOP, "r", [commit(a, 1)])
    assert out[0].pr is None and "limit" in st.warnings[0]


def test_graphql_answers_are_cached(tmp_path):
    from perfhound.gateway.cache import Cache
    a = "a" * 40
    fake = FakeGraphQL({a: [gql_pr(10, a, T0)]})
    cache = Cache(tmp_path / "c.db")
    GitHubGraphQLProvider(gql_client(fake), cache).enrich(SHOP, "r", [commit(a, 1)])
    out, st = GitHubGraphQLProvider(gql_client(fake), cache).enrich(SHOP, "r", [commit(a, 1)])
    assert len(fake.calls) == 1 and st.cache_hits == 1 and out[0].pr.number == 10


def test_range_with_github_through_the_whole_path(fixture_repo, monkeypatch, tmp_path):
    monkeypatch.setenv("PERFHOUND_CACHE_DIR", str(tmp_path))          # own GitHub cache: no answers from other tests
    slow = fixture_repo.sha("slow_add")
    fake = FakeGraphQL({slow: [gql_pr(77, slow, T0 + timedelta(days=1), issues=[(5, "add() got slow", T0)])]})
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/shop")                # as inside GitHub Actions
    res = fetch_range(str(fixture_repo.path), GOOD_TAG, BAD_TAG, out=None, client=gql_client(fake),
                      cache=False, log=quiet)
    assert res["github"] == "acme/shop" and res["stats"]["with_pr"] == 1
    assert res["commits"][-1]["pr"]["number"] == 77 and "add() got slow" in res["commits"][-1]["pr"]["body"]
    assert "#77" in summary_markdown(res)


def test_cli_range(fixture_repo, tmp_path, capsys):
    from perfhound.gateway.__main__ import main
    out, summary = tmp_path / "r.json", tmp_path / "summary.md"
    assert main(["range", str(fixture_repo.path), "--good", GOOD_TAG, "--bad", BAD_TAG, "--no-github",
                 "--out", str(out), "--summary", str(summary)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["count"] == 7
    assert "7 commits" in summary.read_text(encoding="utf-8")
    assert main(["range", str(fixture_repo.path), "--good", "no-such-tag", "--no-github", "--out", str(out)]) == 1
    assert "no-such-tag" in capsys.readouterr().err


def test_first_commit_of_a_repository_has_nothing_to_compare(tmp_path):
    """The very first push of a new repository: GitHub's "before" is all zeros and HEAD has no parent."""
    from perfhound.gateway.gitcmd import run_git
    repo = tmp_path / "new"
    repo.mkdir()
    run_git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    run_git(repo, "add", "-A")
    run_git(repo, "-c", "user.name=d", "-c", "user.email=d@x", "commit", "-q", "-m", "first")
    out = tmp_path / "r.json"
    res = fetch_range(str(repo), "0" * 40, "HEAD", out=out, github=False, cache=False, log=quiet)
    assert res["count"] == 0 and "first commit" in res["warnings"][0]
    assert json.loads(out.read_text(encoding="utf-8"))["count"] == 0
    assert "Perfhound: 0 commits" in summary_markdown(res)


def test_graphql_refused_falls_back_to_rest(fixture_repo, monkeypatch, tmp_path):
    """A token that may not use GraphQL on this repository: PRs still come, through REST."""
    monkeypatch.setenv("PERFHOUND_CACHE_DIR", str(tmp_path))
    slow = fixture_repo.sha("slow_add")

    def transport(method, url, headers, timeout, body=None):
        if method == "POST":                       # GraphQL: refused
            return 200, {}, json.dumps({"data": {"repository": None}, "errors": [
                {"type": "FORBIDDEN", "message": "Resource not accessible by integration"}]}).encode()
        if f"/commits/{slow}/pulls" in url:        # REST: works
            return 200, {}, json.dumps([{"number": 77, "title": "Slow add", "body": "", "labels": [],
                                         "html_url": "u", "merged_at": Z(T0), "merge_commit_sha": slow}]).encode()
        return 200, {}, b"[]"

    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/shop")
    res = fetch_range(str(fixture_repo.path), GOOD_TAG, BAD_TAG, out=None, client=gql_client(transport),
                      cache=False, log=quiet)
    assert res["commits"][-1]["pr"]["number"] == 77 and res["stats"]["with_pr"] == 1
    assert any("used REST instead" in w for w in res["warnings"])
