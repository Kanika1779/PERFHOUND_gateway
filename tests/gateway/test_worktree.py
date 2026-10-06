import hashlib
import os

import pytest

from perfhound.gateway import Gateway
from perfhound.gateway.errors import GatewayError, UnknownRefError
from perfhound.gateway.worktree import WorktreeManager, _try_lock

from .fixture_repo import BAD_TAG, GOOD_TAG


def snapshot_user_state(repo) -> dict:
    """Everything a user could notice: files (bytes), HEAD, branches, index, stash, status."""
    files = {}
    for root, dirs, names in os.walk(repo.path):
        dirs[:] = [d for d in dirs if d != ".git"]
        for n in names:
            p = os.path.join(root, n)
            with open(p, "rb") as fh:
                files[os.path.relpath(p, repo.path)] = hashlib.sha256(fh.read()).hexdigest()
    return {
        "files": files,
        "head": repo.git("rev-parse", "HEAD"),
        "symbolic_head": repo.git("symbolic-ref", "-q", "HEAD"),
        "branches": repo.git("for-each-ref", "--format=%(refname) %(objectname)", "refs/heads", "refs/tags"),
        "index": repo.git("ls-files", "--stage"),
        "status": repo.git("status", "--porcelain", "--ignored"),
        "stash": repo.git("stash", "list"),
    }


@pytest.fixture
def dirty_repo(fresh_fixture_repo):
    """User repo with uncommitted work of every kind."""
    repo = fresh_fixture_repo
    (repo.path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    repo.git("add", ".gitignore")
    repo.git("commit", "-qm", "ignore logs")
    (repo.path / "mathops.py").write_text("# my unsaved experiment\n", encoding="utf-8")   # modified
    (repo.path / "shapes.py").write_text("# staged change\n", encoding="utf-8")
    repo.git("add", "shapes.py")                                                          # staged
    (repo.path / "notes.txt").write_text("todo\n", encoding="utf-8")                      # untracked
    (repo.path / "debug.log").write_text("ignored\n", encoding="utf-8")                   # ignored
    return repo


def test_twenty_checkouts_never_touch_user_folder(dirty_repo, tmp_path):
    repo = dirty_repo
    before = snapshot_user_state(repo)
    shas = [c.sha for c in Gateway(repo.path).get_candidates(GOOD_TAG, BAD_TAG, analyze=False)]
    shas = [repo.sha("initial")] + shas

    with WorktreeManager(repo.path, base_dir=tmp_path / "wt") as wt:
        paths = set()
        for i in range(20):
            sha = shas[i % len(shas)]
            path = wt.checkout(sha)
            paths.add(path)
            # worktree really is at that commit, with exactly that commit's files
            assert repo.git("-C", str(path), "rev-parse", "HEAD") == sha
            expected = set(repo.git("ls-tree", "-r", "--name-only", sha).split("\n"))
            actual = {
                os.path.relpath(os.path.join(r, n), path).replace(os.sep, "/")
                for r, d, ns in os.walk(path) for n in ns if ".git" not in r.split(os.sep) and n != ".git"
            }
            assert actual == expected
        assert len(paths) == 1                       # one folder, reused
        assert snapshot_user_state(repo) == before   # mid-run: untouched

    assert snapshot_user_state(repo) == before       # after close: untouched
    assert repo.git("worktree", "list", "--porcelain").count("worktree ") == 1
    assert not (repo.path / ".git" / "worktrees").exists() or not any((repo.path / ".git" / "worktrees").iterdir())


def test_worktree_lives_outside_user_repo(fresh_fixture_repo):
    with Gateway(fresh_fixture_repo.path).worktree() as wt:
        path = wt.checkout(BAD_TAG)
        assert fresh_fixture_repo.path.resolve() not in path.resolve().parents
        assert (path / "mathops.py").exists()
    assert not path.exists()


def test_leftovers_from_previous_commit_are_removed(fresh_fixture_repo, tmp_path):
    repo = fresh_fixture_repo
    with WorktreeManager(repo.path, base_dir=tmp_path / "wt") as wt:
        path = wt.checkout(repo.sha("docs"))
        (path / "build_output.bin").write_bytes(b"junk")                 # untracked artifact
        (path / "__pycache__").mkdir()
        (path / "__pycache__" / "x.pyc").write_bytes(b"junk")
        (path / "README.md").write_text("benchmark edited me", encoding="utf-8")   # tracked edit
        wt.checkout(repo.sha("merge"))
        assert not (path / "build_output.bin").exists()
        assert not (path / "__pycache__").exists()
        assert (path / "README.md").read_text(encoding="utf-8").startswith("# Fixture repo")
        assert (path / "square.py").exists()


def test_no_branches_created_and_worktree_is_detached(fresh_fixture_repo, tmp_path):
    repo = fresh_fixture_repo
    branches = repo.git("branch", "--list")
    with WorktreeManager(repo.path, base_dir=tmp_path / "wt") as wt:
        path = wt.checkout(GOOD_TAG)
        assert repo.git("-C", str(path), "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"  # detached
        assert repo.git("branch", "--list") == branches


def test_unknown_ref(fresh_fixture_repo, tmp_path):
    with WorktreeManager(fresh_fixture_repo.path, base_dir=tmp_path / "wt") as wt:
        with pytest.raises(UnknownRefError):
            wt.checkout("does-not-exist")


def test_closed_on_exception_and_close_is_idempotent(fresh_fixture_repo, tmp_path):
    with pytest.raises(RuntimeError):
        with WorktreeManager(fresh_fixture_repo.path, base_dir=tmp_path / "wt") as wt:
            path = wt.checkout(BAD_TAG)
            raise RuntimeError("benchmark crashed")
    assert not path.exists()
    wt.close()  # second close: no error
    with pytest.raises(GatewayError):
        wt.checkout(BAD_TAG)


def test_crashed_run_is_cleaned_up_by_next_run(fresh_fixture_repo, tmp_path):
    repo = fresh_fixture_repo
    base = tmp_path / "wt"
    crashed = WorktreeManager(repo.path, base_dir=base)
    stale_path = crashed.checkout(BAD_TAG)
    # simulate the process dying: OS releases the lock, nothing else is cleaned
    crashed._lock_fh.close()
    crashed._lock_fh = None
    crashed._closed = True
    assert stale_path.exists()

    with WorktreeManager(repo.path, base_dir=base) as fresh:
        assert fresh.removed_stale == [stale_path]
        assert not stale_path.exists()
        assert repo.git("worktree", "list", "--porcelain").count("worktree ") == 1


def test_live_worktree_of_another_run_is_not_removed(fresh_fixture_repo, tmp_path):
    repo = fresh_fixture_repo
    base = tmp_path / "wt"
    with WorktreeManager(repo.path, base_dir=base) as first:
        p1 = first.checkout(BAD_TAG)
        with WorktreeManager(repo.path, base_dir=base) as second:
            p2 = second.checkout(GOOD_TAG)
            assert second.removed_stale == []
            assert p1.exists() and p1 != p2
            assert first.checkout(GOOD_TAG) == p1   # first still works
        assert p1.exists() and not p2.exists()


def test_lock_helper(tmp_path):
    f = tmp_path / "x.lock"
    with open(f, "a+b") as a, open(f, "a+b") as b:
        assert _try_lock(a)
        assert not _try_lock(b)
