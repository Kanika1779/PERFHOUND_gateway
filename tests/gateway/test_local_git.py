import socket
from datetime import timedelta

import pytest

from perfhound.gateway import CandidateCommit, Gateway
from perfhound.gateway import local_git
from perfhound.gateway.local_git import LocalGitProvider, truncate_diff
from perfhound.gateway.range import resolve_range

from .fixture_repo import BAD_TAG, FIRST_PARENT_ORDER, GOOD_TAG


@pytest.fixture(scope="module")
def candidates(fixture_repo):
    return Gateway(fixture_repo.path).get_candidates(GOOD_TAG, BAD_TAG)


def by_name(fixture_repo, candidates, name) -> CandidateCommit:
    sha = fixture_repo.sha(name)
    return next(c for c in candidates if c.sha == sha)


def test_returns_all_range_commits_in_order(fixture_repo, candidates):
    assert [c.sha for c in candidates] == fixture_repo.first_parent_shas()
    assert [c.position for c in candidates] == list(range(7))
    assert candidates[0].parent == fixture_repo.sha("initial")
    for prev, cur in zip(candidates, candidates[1:]):
        assert cur.parent == prev.sha  # unbroken first-parent chain


def test_metadata(fixture_repo, candidates):
    c = by_name(fixture_repo, candidates, "tweak_add")
    assert c.message == "Coerce inputs to int in add()"
    assert c.author == "Test Author"
    assert c.timestamp.utcoffset() == timedelta(hours=5, minutes=30)
    assert c.source == "local" and c.pr is None
    assert c.changed_functions == ("calc.add",)  # from the Code Analyzer (Step 5)


def test_normal_change(fixture_repo, candidates):
    c = by_name(fixture_repo, candidates, "tweak_add")
    assert [(f.path, f.status, f.additions, f.deletions) for f in c.files] == [("calc.py", "M", 1, 1)]
    assert "-    return a + b" in c.diff and "+    return int(a) + int(b)" in c.diff


def test_rename(fixture_repo, candidates):
    (f,) = by_name(fixture_repo, candidates, "rename").files
    assert (f.status, f.old_path, f.path, f.additions, f.deletions) == ("R", "calc.py", "mathops.py", 0, 0)


def test_function_delete(fixture_repo, candidates):
    c = by_name(fixture_repo, candidates, "delete_sub")
    (f,) = c.files
    assert f.path == "mathops.py" and f.additions == 0 and f.deletions == 4  # 2 blank lines + def + return
    assert "-def sub(a, b):" in c.diff


def test_added_and_binary_files(fixture_repo, candidates):
    files = {f.path: f for f in by_name(fixture_repo, candidates, "docs").files}
    assert files["README.md"].status == "A" and files["README.md"].additions == 3
    assert files["data.bin"].is_binary and files["data.bin"].status == "A"


def test_merge_commit_is_diffed_against_first_parent(fixture_repo, candidates):
    c = by_name(fixture_repo, candidates, "merge")
    assert c.parent == fixture_repo.sha("docs")
    assert [(f.path, f.status) for f in c.files] == [("square.py", "A")]
    assert "+class Square:" in c.diff


def test_regression_commit_diff(fixture_repo, candidates):
    c = by_name(fixture_repo, candidates, "slow_add")
    assert "+    for _ in range(1000):" in c.diff
    assert not c.diff_truncated


def test_records_survive_json_round_trip(candidates):
    for c in candidates:
        assert CandidateCommit.from_json(c.to_json()) == c


def test_uses_exactly_two_git_processes(fixture_repo, monkeypatch):
    calls = []
    real = local_git.run_git

    def counting(*args, **kwargs):
        calls.append(args[1:3])
        return real(*args, **kwargs)

    monkeypatch.setattr(local_git, "run_git", counting)
    r = resolve_range(fixture_repo.path, GOOD_TAG, BAD_TAG)
    LocalGitProvider(fixture_repo.path).get_commits(r)
    assert len(calls) == 2


def test_subset_keeps_requested_order_and_range_positions(fixture_repo):
    r = resolve_range(fixture_repo.path, GOOD_TAG, BAD_TAG)
    wanted = [fixture_repo.sha("slow_add"), fixture_repo.sha("rename")]
    got = LocalGitProvider(fixture_repo.path).get_commits(r, wanted)
    assert [c.sha for c in got] == wanted
    assert [c.position for c in got] == [6, 1]


def test_diff_truncation_by_lines(fixture_repo):
    cs = Gateway(fixture_repo.path, max_diff_lines=3).get_candidates(GOOD_TAG, BAD_TAG)
    slow = cs[-1]
    assert slow.diff_truncated
    assert len(slow.diff.splitlines()) == 3


def test_diff_truncation_by_chars():
    text, cut = truncate_diff("x" * 100 + "\n", max_lines=400, max_chars=10)
    assert cut and text == "x" * 10
    assert truncate_diff("a\nb\n", 400, 100) == ("a\nb\n", False)


def test_line_count_ignores_exotic_line_separators():
    # \x0c (form feed) and \u2028 are NOT newlines for git; they must not
    # count as extra lines or split a diff line in two.
    diff = "+a\x0cb\n+c\u2028d\n"
    assert truncate_diff(diff, max_lines=2, max_chars=1000) == (diff, False)


def test_works_without_network(fixture_repo, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("gateway tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    assert len(Gateway(fixture_repo.path).get_candidates(GOOD_TAG, BAD_TAG)) == 7


def test_unicode_message_and_filename_and_empty_commit(fresh_fixture_repo):
    repo = fresh_fixture_repo
    (repo.path / "données.py").write_text("def café():\n    return 'é'\n", encoding="utf-8", newline="\n")
    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "Ajouter café ✓\n\nCorps du message.")
    repo.git("commit", "-q", "--allow-empty", "-m", "Empty commit")
    cs = Gateway(repo.path).get_candidates(BAD_TAG, "HEAD")
    uni, empty = cs
    assert uni.message == "Ajouter café ✓\n\nCorps du message."
    assert uni.subject == "Ajouter café ✓"
    assert [f.path for f in uni.files] == ["données.py"]
    assert "+def café():" in uni.diff
    assert empty.files == () and empty.diff == ""


def test_does_not_modify_repo(fixture_repo):
    before = (fixture_repo.git("status", "--porcelain"), fixture_repo.git("rev-parse", "HEAD"))
    Gateway(fixture_repo.path).get_candidates(GOOD_TAG, BAD_TAG)
    assert (fixture_repo.git("status", "--porcelain"), fixture_repo.git("rev-parse", "HEAD")) == before


def test_crlf_and_lone_cr_survive_exactly(fresh_fixture_repo):
    """Windows regression: text-mode pipes turned \r\n into \n (and \n into \r\n on stdin)."""
    repo = fresh_fixture_repo
    (repo.path / "win.py").write_bytes(b"def f():\r\n    return 1\r\n")
    repo.git("add", "win.py")
    repo.git("commit", "-q", "-m", "crlf file")
    (repo.path / "win.py").write_bytes(b"def f():\r\n    return 2\r\n")
    repo.git("commit", "-qam", "Message with a lone\rcarriage return")
    first, second = Gateway(repo.path, cache=False).get_candidates(BAD_TAG, "HEAD")
    assert "+    return 2\r\n" in second.diff
    assert "\r" in second.message
    assert second.changed_functions == ("win.f",)
