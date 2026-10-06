"""Code Analyzer (gateway part 6): which functions did a commit change?

Languages: Python (this file, stdlib `ast`) and Java (lang_java.py,
tree-sitter). Add a language = one class with `extensions`, `parse()` and
`qualifier()`; see PythonSupport.


Approach: parse the file BEFORE (first parent) and AFTER the commit with
Python's `ast` module and compare each function's normalized syntax tree.

Why AST comparison instead of mapping diff line numbers to functions:
* comment / whitespace / docstring-only edits are NOT reported (they
  cannot change performance), line-mapping would report them;
* a function that only moved inside the file is NOT reported;
* no diff parsing (git quotes unusual paths in patch headers).
Cost: we know WHICH function changed, not which line.

Naming follows Python's __qualname__, prefixed by the module:
    pkg.mod.func                 top-level function
    pkg.mod.Class.method         method
    pkg.mod.outer.<locals>.inner nested function
Two pseudo-entries can appear in changed_functions:
    pkg.mod.<module>             module-level code changed (imports, constants...)
    pkg.mod.Class                class-level code changed (attributes, bases, decorators)

A change inside a nested function also marks every enclosing function as
changed (their source really did change).

Files that fail to parse (e.g. syntax newer than the running Python) are
skipped and listed in FunctionChanges.skipped_files - never silently.
"""

from __future__ import annotations

import ast
import copy
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .gitcmd import read_blobs
from .models import CandidateCommit

DEFAULT_MAX_FILE_BYTES = 2_000_000
ANALYZER_VERSION = 2   # v2: Java support. Bump when analysis output changes -> old cache entries are ignored
_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class FunctionChanges:
    changed: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    skipped_files: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# pure helpers (no git) - easy to unit test
# --------------------------------------------------------------------------

def path_to_module(path: str) -> str:
    """'sympy/core/basic.py' -> 'sympy.core.basic'; 'pkg/__init__.py' -> 'pkg'.

    A leading 'src/' is dropped (src-layout), matching how the code is imported.
    """
    p = path.replace("\\", "/")
    if p.startswith("src/"):
        p = p[4:]
    if p.endswith(".py"):
        p = p[:-3]
    parts = [x for x in p.split("/") if x]
    if parts and parts[-1] == "__init__" and len(parts) > 1:
        parts = parts[:-1]
    return ".".join(parts)


_BODY_OWNERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)


def _without_docstring(body: list[ast.stmt]) -> list:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:] or [ast.Pass()]
    return body


def _fingerprint(node) -> str:
    """Structure of the code, ignoring comments, formatting, docstrings and positions.

    Like ast.dump(include_attributes=False) but skips docstrings while
    serializing, so the tree never has to be copied or rewritten.
    """
    if isinstance(node, ast.AST):
        parts = [type(node).__name__]
        for name, value in ast.iter_fields(node):
            if name == "body" and isinstance(node, _BODY_OWNERS):
                value = _without_docstring(value)
            elif name == "type_comment":
                continue
            parts.append(_fingerprint(value))
        return "(" + " ".join(parts) + ")"
    if isinstance(node, list):
        return "[" + " ".join(_fingerprint(x) for x in node) + "]"
    return repr(node)


@dataclass(frozen=True)
class _Entry:
    raw: str            # exact source text (fast path: equal text => equal code)
    node: Any           # syntax node, fingerprinted lazily, only when the raw text differs


@dataclass
class Definitions:
    """Everything defined in one file (any language).

    Python keys: 'func', 'Class.method', 'outer.<locals>.inner',
    'Class' (class-level code only) and '<module>' (module-level code only).
    Java keys are already fully qualified, see lang_java.py.
    """

    entries: dict[str, _Entry]
    classes: frozenset[str]
    fingerprint_fn: Callable[[Any], str] = field(default=None, repr=False)   # type: ignore[assignment]
    namespace: str = ""            # Java package; unused for Python
    _fps: dict[str, str] = field(default_factory=dict, repr=False)

    def fingerprint(self, key: str) -> str:
        if key not in self._fps:
            fn = self.fingerprint_fn or _fingerprint
            self._fps[key] = fn(self.entries[key].node)
        return self._fps[key]

    @property
    def fingerprints(self) -> dict[str, str]:   # convenience for tests / debugging
        return {k: self.fingerprint(k) for k in self.entries}


def _child_statements(node: ast.stmt) -> list[ast.stmt]:
    """Statements nested in if / for / while / with / try / match blocks."""
    out: list[ast.stmt] = []
    for name in ("body", "orelse", "finalbody"):
        out.extend(s for s in getattr(node, name, None) or [] if isinstance(s, ast.stmt))
    for handler in getattr(node, "handlers", None) or []:
        out.extend(handler.body)
    for case in getattr(node, "cases", None) or []:   # match statement
        out.extend(case.body)
    return out


def _residual(stmts: list[ast.stmt]) -> list[ast.stmt]:
    return [s for s in stmts if not isinstance(s, (*_FUNC_TYPES, ast.ClassDef))]


def extract_definitions(source: bytes) -> Definitions:
    """Parse a file and index every function / class body / module body.

    Raises SyntaxError / ValueError when the source cannot be parsed.
    """
    with warnings.catch_warnings():
        # analyzed repos contain e.g. "\d" in normal strings: Python 3.12+ prints a
        # SyntaxWarning per occurrence while parsing - noise for our users, not our problem
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        tree = ast.parse(source)
    lines = source.decode("utf-8", "replace").split("\n")
    entries: dict[str, _Entry] = {}
    classes: set[str] = set()

    def text(node: ast.AST) -> str:
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        return "\n".join(lines[first - 1:node.end_lineno])

    def walk(stmts: list[ast.stmt], prefix: str) -> None:
        for node in stmts:
            if isinstance(node, _FUNC_TYPES):
                name = prefix + node.name
                entries[name] = _Entry(text(node), node)
                walk(node.body, name + ".<locals>.")
            elif isinstance(node, ast.ClassDef):
                name = prefix + node.name
                classes.add(name)
                shell = copy.copy(node)
                shell.body = _residual(node.body)
                entries[name] = _Entry(text(node), shell)
                walk(node.body, name + ".")
            else:
                walk(_child_statements(node), prefix)

    walk(tree.body, "")
    residual = _residual(tree.body)
    entries["<module>"] = _Entry("\x00".join(text(s) for s in residual),
                                 ast.Module(body=residual, type_ignores=[]))
    return Definitions(entries, frozenset(classes), _fingerprint)


_EMPTY = Definitions({}, frozenset())


def python_qualifier(module: str) -> Callable[[str], str]:
    return lambda key: f"{module}.{key}" if module else key


def diff_parsed(old: Definitions, new: Definitions, qualify: Callable[[str], str] | str) -> FunctionChanges:
    """Compare two parsed versions of one file. `qualify` turns a key into
    the reported name (a plain string is treated as a Python module name)."""
    classes = old.classes | new.classes
    q = python_qualifier(qualify) if isinstance(qualify, str) else qualify

    changed, added, deleted = [], [], []
    for key in sorted(set(old.entries) | set(new.entries)):
        in_old, in_new = key in old.entries, key in new.entries
        if in_old and in_new:
            if old.entries[key].raw != new.entries[key].raw and old.fingerprint(key) != new.fingerprint(key):
                changed.append(q(key))
        elif key in classes or key == "<module>":
            continue  # a new / removed class or file shows up through its functions
        elif in_new:
            added.append(q(key))
        else:
            deleted.append(q(key))
    return FunctionChanges(tuple(changed), tuple(added), tuple(deleted))


def diff_definitions(old_source: bytes | None, new_source: bytes | None, module: str) -> FunctionChanges:
    """Compare one file before/after. None = file did not exist on that side.

    Raises SyntaxError / ValueError if either side cannot be parsed.
    """
    old = extract_definitions(old_source) if old_source is not None else _EMPTY
    new = extract_definitions(new_source) if new_source is not None else _EMPTY
    return diff_parsed(old, new, module)


# --------------------------------------------------------------------------
# git-backed analyzer
# --------------------------------------------------------------------------

class PythonSupport:
    """Python: `ast` from the standard library; names from the file path."""

    name = "python"
    extensions = (".py",)

    def parse(self, source: bytes) -> Definitions:
        return extract_definitions(source)

    def qualifier(self, path: str, old: Definitions, new: Definitions) -> Callable[[str], str]:
        return python_qualifier(path_to_module(path))


def default_languages() -> list:
    langs: list = [PythonSupport()]
    from .lang_java import JavaSupport   # imported lazily: needs tree-sitter
    langs.append(JavaSupport())
    return langs


class CodeAnalyzer:
    """Fills changed/added/deleted_functions for CandidateCommits.

    Language is chosen by file extension (see `languages`). All needed file
    versions for all commits are fetched with ONE `git cat-file --batch`.
    """

    def __init__(self, repo: str | Path, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES, languages=None) -> None:
        self.repo = Path(repo)
        self.max_file_bytes = max_file_bytes
        self.languages = list(languages) if languages is not None else default_languages()
        self._by_ext = {ext: lang for lang in self.languages for ext in lang.extensions}

    def language_for(self, path: str):
        for ext, lang in self._by_ext.items():
            if path.endswith(ext):
                return lang
        return None

    def _sides(self, c: CandidateCommit):
        """(language, old_spec | None, new_spec | None, naming_path) per supported source file."""
        for f in c.files:
            old_path = f.old_path or f.path
            lang = self.language_for(f.path) or self.language_for(old_path)
            if lang is None or f.is_binary:
                continue
            old_spec = None if f.status == "A" else f"{c.parent}:{old_path}"
            new_spec = None if f.status == "D" else f"{c.sha}:{f.path}"
            # renamed files are named after the NEW path, so a pure rename
            # reports no function changes
            naming_path = f.path if self.language_for(f.path) is lang else old_path
            yield lang, old_spec, new_spec, naming_path

    def analyze(self, commits: Sequence[CandidateCommit]) -> dict[str, FunctionChanges]:
        specs = [s for c in commits for _, o, n, _ in self._sides(c) for s in (o, n) if s]
        if specs:
            from .local_git import prefetch_blobs   # no-op unless partial clone
            prefetch_blobs(self.repo, [c.sha for c in commits if any(True for _ in self._sides(c))])
        blobs = read_blobs(self.repo, specs)

        # A file version is usually the "after" of one commit AND the "before"
        # of the next: parse each distinct content only once.
        parsed: dict[tuple[str, bytes], Definitions | Exception] = {}

        def parse(lang, src: bytes | None) -> Definitions:
            if src is None:
                return _EMPTY
            key = (lang.name, src)
            if key not in parsed:
                try:
                    parsed[key] = lang.parse(src)
                except (SyntaxError, ValueError, RecursionError) as exc:
                    parsed[key] = exc
            got = parsed[key]
            if isinstance(got, Exception):
                raise got
            return got

        result: dict[str, FunctionChanges] = {}
        for c in commits:
            changed: set[str] = set()
            added: set[str] = set()
            deleted: set[str] = set()
            skipped: list[str] = []
            for lang, old_spec, new_spec, path in self._sides(c):
                old_src = blobs.get(old_spec) if old_spec else None
                new_src = blobs.get(new_spec) if new_spec else None
                if (old_spec and old_src is None) or (new_spec and new_src is None):
                    skipped.append(path)
                    continue
                if any(s is not None and len(s) > self.max_file_bytes for s in (old_src, new_src)):
                    skipped.append(path)
                    continue
                try:
                    old_defs, new_defs = parse(lang, old_src), parse(lang, new_src)
                    fc = diff_parsed(old_defs, new_defs, lang.qualifier(path, old_defs, new_defs))
                except (SyntaxError, ValueError, RecursionError):
                    skipped.append(path)
                    continue
                changed.update(fc.changed)
                added.update(fc.added)
                deleted.update(fc.deleted)
            result[c.sha] = FunctionChanges(
                tuple(sorted(changed)), tuple(sorted(added)), tuple(sorted(deleted)), tuple(skipped)
            )
        return result

    def enrich(self, commits: Sequence[CandidateCommit]) -> list[CandidateCommit]:
        changes = self.analyze(commits)
        return [
            replace(
                c,
                changed_functions=changes[c.sha].changed,
                added_functions=changes[c.sha].added,
                deleted_functions=changes[c.sha].deleted,
                unanalyzed_files=changes[c.sha].skipped_files,
            )
            for c in commits
        ]
