"""Builds a tiny, deterministic git repo for gateway tests (~1 second).

History (main, oldest -> newest). Tags: v1.0 = good, v1.1 = bad.

    initial ........ calc.py (add, sub) + shapes.py (Circle)     <- tag v1.0
    tweak_add ...... normal change: body of calc.add
    rename ......... calc.py -> mathops.py (pure rename)
    delete_sub ..... function delete: mathops.sub removed
    circle_area .... class method change: shapes.Circle.area
    docs ........... non-Python changes: README.md + binary data.bin
    merge .......... merge commit of branch 'feature'
                       feature_square (on 'feature'): new file square.py
    slow_add ....... "regression": mathops.add gets a pointless loop  <- tag v1.1

So v1.0..v1.1 has 7 first-parent commits (8 including feature_square).

Commit dates are fixed and global/system git config is ignored, so the
SHAs are identical on every machine. Tests may rely on that, but should
prefer `repo.sha("name")` over hard-coded SHAs.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

GOOD_TAG = "v1.0"
BAD_TAG = "v1.1"
FIRST_PARENT_ORDER = ["tweak_add", "rename", "delete_sub", "circle_area", "docs", "merge", "slow_add"]

_BASE_EPOCH = 1_759_550_400  # 2025-10-04T04:00:00Z
_TZ = "+0530"


@dataclass
class FixtureRepo:
    path: Path
    shas: dict[str, str] = field(default_factory=dict)
    good: str = GOOD_TAG
    bad: str = BAD_TAG

    def sha(self, name: str) -> str:
        return self.shas[name]

    def git(self, *args: str) -> str:
        return _git(self.path, *args)

    def first_parent_shas(self) -> list[str]:
        return [self.shas[n] for n in FIRST_PARENT_ORDER]


def git_env() -> dict[str, str]:
    """Environment that ignores the user's own git config (autocrlf, hooks, signing...)."""
    env = dict(os.environ)
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="Test Author",
        GIT_AUTHOR_EMAIL="author@example.com",
        GIT_COMMITTER_NAME="Test Author",
        GIT_COMMITTER_EMAIL="author@example.com",
        GIT_TERMINAL_PROMPT="0",
    )
    return env


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env or git_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


class _Builder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tick = 0
        self.shas: dict[str, str] = {}

    def _env(self) -> dict[str, str]:
        env = git_env()
        date = f"{_BASE_EPOCH + self.tick * 3600} {_TZ}"
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
        self.tick += 1
        return env

    def git(self, *args: str) -> str:
        return _git(self.root, *args)

    def commit(self, name: str, message: str) -> str:
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "--no-verify", "-m", message, env=self._env())
        self.shas[name] = _git(self.root, "rev-parse", "HEAD")
        return self.shas[name]

    def merge(self, name: str, branch: str, message: str) -> str:
        _git(self.root, "merge", "-q", "--no-ff", "--no-verify", "-m", message, branch, env=self._env())
        self.shas[name] = _git(self.root, "rev-parse", "HEAD")
        return self.shas[name]


CALC_V0 = '''\
def add(a, b):
    return a + b


def sub(a, b):
    return a - b
'''

CALC_V1 = '''\
def add(a, b):
    return int(a) + int(b)


def sub(a, b):
    return a - b
'''

CALC_V2 = '''\
def add(a, b):
    return int(a) + int(b)
'''

CALC_SLOW = '''\
def add(a, b):
    total = 0
    for _ in range(1000):
        total = int(a) + int(b)
    return total
'''

SHAPES_V0 = '''\
class Circle:
    def __init__(self, r):
        self.r = r

    def area(self):
        return 3.14 * self.r * self.r

    def perimeter(self):
        return 2 * 3.14 * self.r
'''

SHAPES_V1 = '''\
import math


class Circle:
    def __init__(self, r):
        self.r = r

    def area(self):
        return math.pi * self.r ** 2

    def perimeter(self):
        return 2 * 3.14 * self.r
'''

SQUARE = '''\
class Square:
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side * self.side
'''


def build_fixture_repo(root: Path) -> FixtureRepo:
    """Create the fixture repo inside `root` (must be empty or not exist)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    b = _Builder(root)
    b.git("init", "-q", "-b", "main")
    b.git("config", "core.autocrlf", "false")
    b.git("config", "commit.gpgsign", "false")

    _write(root / "calc.py", CALC_V0)
    _write(root / "shapes.py", SHAPES_V0)
    b.commit("initial", "Initial commit: calc and shapes")
    b.git("tag", GOOD_TAG)

    _write(root / "calc.py", CALC_V1)
    b.commit("tweak_add", "Coerce inputs to int in add()")

    b.git("mv", "calc.py", "mathops.py")
    b.commit("rename", "Rename calc.py to mathops.py")

    _write(root / "mathops.py", CALC_V2)
    b.commit("delete_sub", "Remove unused sub()")

    _write(root / "shapes.py", SHAPES_V1)
    b.commit("circle_area", "Use math.pi in Circle.area")

    b.git("checkout", "-q", "-b", "feature")
    _write(root / "square.py", SQUARE)
    b.commit("feature_square", "Add Square shape")
    b.git("checkout", "-q", "main")

    _write(root / "README.md", "# Fixture repo\n\nUsed by Perfhound tests.\n")
    (root / "data.bin").write_bytes(bytes(range(256)))
    b.commit("docs", "Add README and sample binary data")

    b.merge("merge", "feature", "Merge branch 'feature'")

    _write(root / "mathops.py", CALC_SLOW)
    b.commit("slow_add", "Make add() more robust")  # innocent-looking message, real slowdown
    b.git("tag", BAD_TAG)

    return FixtureRepo(path=root, shas=b.shas)
