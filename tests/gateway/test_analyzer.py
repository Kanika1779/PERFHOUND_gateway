import textwrap

import pytest

from perfhound.gateway import Gateway
from perfhound.gateway import analyzer as analyzer_mod
from perfhound.gateway.analyzer import CodeAnalyzer, diff_definitions, extract_definitions, path_to_module
from perfhound.gateway.gitcmd import read_blobs

from .fixture_repo import BAD_TAG, GOOD_TAG


def src(text: str) -> bytes:
    return textwrap.dedent(text).encode("utf-8")


def diff(old: str | None, new: str | None, module: str = "m"):
    return diff_definitions(src(old) if old is not None else None, src(new) if new is not None else None, module)


# ---------------------------------------------------------------- naming

@pytest.mark.parametrize("path,module", [
    ("sympy/core/basic.py", "sympy.core.basic"),
    ("pkg/__init__.py", "pkg"),
    ("setup.py", "setup"),
    ("src/perfhound/gateway/api.py", "perfhound.gateway.api"),
    ("__init__.py", "__init__"),
])
def test_path_to_module(path, module):
    assert path_to_module(path) == module


def test_qualnames():
    defs = extract_definitions(src("""
        def f(): pass
        async def g(): pass
        class A:
            x = 1
            def m(self):
                def inner(): pass
            class B:
                def n(self): pass
        if True:
            def conditional(): pass
        try:
            import fast
        except ImportError:
            def fallback(): pass
    """))
    assert set(defs.fingerprints) == {
        "f", "g", "A", "A.m", "A.m.<locals>.inner", "A.B", "A.B.n", "conditional", "fallback", "<module>",
    }
    assert defs.classes == {"A", "A.B"}


# ---------------------------------------------------------------- what counts as a change

BASE = """
    import os

    def f(x):
        return x + 1

    def g(x):
        return x * 2
"""


def test_body_change_is_changed():
    fc = diff(BASE, BASE.replace("x + 1", "x + 2"))
    assert fc.changed == ("m.f",) and fc.added == () and fc.deleted == ()


@pytest.mark.parametrize("edit", [
    lambda s: s.replace("return x + 1", "return x + 1  # comment"),       # comment
    lambda s: s.replace("return x + 1", "return (x  +  1)"),               # formatting
    lambda s: s.replace("def f(x):\n", 'def f(x):\n        """Doc."""\n'),    # docstring
    lambda s: s.replace("\n\n    def g", "\n\n\n\n\n    def g"),          # blank lines
])
def test_cosmetic_edits_are_not_changes(edit):
    new = edit(BASE)
    assert new != BASE
    assert diff(BASE, new) == analyzer_mod.FunctionChanges()


def test_moving_a_function_is_not_a_change():
    swapped = """
        import os

        def g(x):
            return x * 2

        def f(x):
            return x + 1
    """
    assert diff(BASE, swapped) == analyzer_mod.FunctionChanges()


def test_added_and_deleted_functions():
    fc = diff(BASE, BASE.replace("def g(x):\n        return x * 2", "def h(x):\n        return x * 2"))
    assert fc.added == ("m.h",) and fc.deleted == ("m.g",) and fc.changed == ()


def test_decorator_change_is_a_change():
    old = "def f():\n    pass\n"
    new = "import functools\n@functools.lru_cache\ndef f():\n    pass\n"
    fc = diff(old, new)
    assert "m.f" in fc.changed and "m.<module>" in fc.changed


def test_signature_change_is_a_change():
    assert diff("def f(a): return a\n", "def f(a, b=0): return a\n").changed == ("m.f",)


def test_nested_change_marks_inner_and_enclosing():
    old = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    fc = diff(old, old.replace("return 1", "return 2"))
    assert fc.changed == ("m.outer", "m.outer.<locals>.inner")


def test_method_and_class_level_changes():
    old = "class A:\n    size = 10\n    def m(self):\n        return self.size\n"
    assert diff(old, old.replace("return self.size", "return self.size * 2")).changed == ("m.A.m",)
    assert diff(old, old.replace("size = 10", "size = 10_000")).changed == ("m.A",)


def test_module_level_change():
    assert diff(BASE, BASE.replace("import os", "import os, sys")).changed == ("m.<module>",)


def test_new_file_and_deleted_file():
    code = "class S:\n    def area(self): pass\ndef helper(): pass\n"
    added = diff(None, code)
    assert added.added == ("m.S.area", "m.helper") and added.changed == () and added.deleted == ()
    gone = diff(code, None)
    assert gone.deleted == ("m.S.area", "m.helper") and gone.changed == () and gone.added == ()


def test_syntax_error_raises():
    with pytest.raises(SyntaxError):
        diff("def f(): pass\n", "def f(:\n")


# ---------------------------------------------------------------- git integration

def test_fixture_repo_function_changes(fixture_repo):
    cs = {c.sha: c for c in Gateway(fixture_repo.path).get_candidates(GOOD_TAG, BAD_TAG)}

    def fc(name):
        c = cs[fixture_repo.sha(name)]
        return c.changed_functions, c.added_functions, c.deleted_functions

    assert fc("tweak_add") == (("calc.add",), (), ())
    assert fc("rename") == ((), (), ())                       # pure rename: nothing changed
    assert fc("delete_sub") == ((), (), ("mathops.sub",))
    assert fc("circle_area") == (("shapes.<module>", "shapes.Circle.area"), (), ())  # + `import math`
    assert fc("docs") == ((), (), ())                          # no Python files
    assert fc("merge") == ((), ("square.Square.__init__", "square.Square.area"), ())
    assert fc("slow_add") == (("mathops.add",), (), ())        # the hidden regression
    assert all(c.unanalyzed_files == () for c in cs.values())


def test_analyze_false_skips_analysis(fixture_repo):
    cs = Gateway(fixture_repo.path).get_candidates(GOOD_TAG, BAD_TAG, analyze=False)
    assert all(c.changed_functions == () for c in cs)


def test_rename_with_edit_reports_change_under_new_module(fresh_fixture_repo):
    repo = fresh_fixture_repo
    repo.git("mv", "mathops.py", "ops.py")
    p = repo.path / "ops.py"
    p.write_text(p.read_text(encoding="utf-8").replace("range(1000)", "range(10)"), encoding="utf-8", newline="\n")
    repo.git("commit", "-qam", "Rename and fix")
    (c,) = Gateway(repo.path).get_candidates(BAD_TAG, "HEAD")
    assert c.files[0].status == "R"
    assert (c.changed_functions, c.added_functions, c.deleted_functions) == (("ops.add",), (), ())


def test_unparseable_file_is_reported_not_hidden(fresh_fixture_repo):
    repo = fresh_fixture_repo
    (repo.path / "broken.py").write_text("def f(:\n", encoding="utf-8")
    (repo.path / "mathops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8", newline="\n")
    repo.git("add", "-A")
    repo.git("commit", "-qm", "Broken file + fix add")
    (c,) = Gateway(repo.path).get_candidates(BAD_TAG, "HEAD")
    assert c.unanalyzed_files == ("broken.py",)
    assert c.changed_functions == ("mathops.add",)   # other files still analyzed


def test_non_utf8_file_with_coding_cookie(fresh_fixture_repo):
    repo = fresh_fixture_repo
    (repo.path / "legacy.py").write_bytes("# -*- coding: latin-1 -*-\ndef caf\u00e9(): return 1\n".encode("latin-1"))
    repo.git("add", "-A")
    repo.git("commit", "-qm", "legacy file")
    (c,) = Gateway(repo.path).get_candidates(BAD_TAG, "HEAD")
    assert c.added_functions == ("legacy.caf\u00e9",)


def test_analyzer_uses_one_git_process(fixture_repo, monkeypatch):
    calls = []
    real = analyzer_mod.read_blobs
    monkeypatch.setattr(analyzer_mod, "read_blobs", lambda *a, **k: calls.append(1) or real(*a, **k))
    commits = Gateway(fixture_repo.path).get_candidates(GOOD_TAG, BAD_TAG, analyze=False)
    CodeAnalyzer(fixture_repo.path).analyze(commits)
    assert len(calls) == 1


def test_read_blobs_handles_missing_and_binary(fixture_repo):
    head = fixture_repo.sha("slow_add")
    got = read_blobs(fixture_repo.path, [f"{head}:mathops.py", f"{head}:nope.py", f"{head}:data.bin"])
    assert got[f"{head}:mathops.py"].startswith(b"def add")
    assert got[f"{head}:nope.py"] is None
    assert got[f"{head}:data.bin"] == bytes(range(256))


def test_parsing_old_code_prints_no_syntax_warnings(recwarn):
    """Real repos (sympy, dask) contain '\\d' in plain strings: Python 3.12+ warns while parsing."""
    fc = diff('def f():\n    return "\\d+"\n', 'def f():\n    return "\\d*"\n')
    assert fc.changed == ("m.f",)
    assert not [w for w in recwarn if issubclass(w.category, (SyntaxWarning, DeprecationWarning))]
