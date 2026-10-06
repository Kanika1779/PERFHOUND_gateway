"""Fixtures shared by every test folder (gateway, rag, ...)."""

import os

import pytest

from gateway.fixture_repo import FixtureRepo, build_fixture_repo   # tests/ is on sys.path (rootdir conftest)


@pytest.fixture(scope="session")
def fixture_repo(tmp_path_factory) -> FixtureRepo:
    """Shared, READ-ONLY fixture repo (built once per test session).

    Tests that modify the repo (checkout, worktrees...) must use
    `fresh_fixture_repo` instead.
    """
    return build_fixture_repo(tmp_path_factory.mktemp("fixture_repo"))


@pytest.fixture
def fresh_fixture_repo(tmp_path) -> FixtureRepo:
    """A private copy of the fixture repo for tests that mutate it."""
    return build_fixture_repo(tmp_path / "repo")


@pytest.fixture(scope="session", autouse=True)
def isolated_cache_dir(tmp_path_factory):
    """Never let tests write to the real ~/.perfhound/cache.db."""
    old = os.environ.get("PERFHOUND_CACHE_DIR")
    os.environ["PERFHOUND_CACHE_DIR"] = str(tmp_path_factory.mktemp("perfhound_cache"))
    yield
    if old is None:
        os.environ.pop("PERFHOUND_CACHE_DIR", None)
    else:
        os.environ["PERFHOUND_CACHE_DIR"] = old
