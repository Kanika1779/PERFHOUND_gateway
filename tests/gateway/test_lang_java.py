import textwrap

import pytest

from perfhound.gateway import Gateway
from perfhound.gateway.analyzer import FunctionChanges, diff_parsed
from perfhound.gateway.lang_java import JavaSupport

from .fixture_repo import BAD_TAG

JAVA = JavaSupport()


def j(text: str) -> bytes:
    return textwrap.dedent(text).encode("utf-8")


def diff(old, new, path="src/main/java/com/x/Calc.java") -> FunctionChanges:
    o = JAVA.parse(j(old)) if old is not None else JAVA.parse(b"")
    n = JAVA.parse(j(new)) if new is not None else JAVA.parse(b"")
    return diff_parsed(o, n, JAVA.qualifier(path, o, n))


BASE = """\
package com.x;

import java.util.List;

/** Calculator. */
public class Calc<T> {
    private int n = 1;

    static { System.out.println("init"); }

    public Calc(int n) { this.n = n; }

    public int add(int a, int b) { return a + b + n; }

    public String add(String a, List<String> b) { return a + b; }

    static class Helper {
        int twice(int x) { return 2 * x; }
    }

    void run() { Runnable r = () -> System.out.println(1); r.run(); }
}
"""


def test_names_follow_jvm_conventions():
    keys = set(JAVA.parse(j(BASE)).entries)
    assert keys == {
        "com.x.Calc", "com.x.Calc.<clinit>", "com.x.Calc.<init>(int)",
        "com.x.Calc.add(int,int)", "com.x.Calc.add(String,List)",   # overloads, generics erased
        "com.x.Calc$Helper", "com.x.Calc$Helper.twice(int)", "com.x.Calc.run()", "<module>",
    }


def test_body_change_hits_only_that_overload():
    fc = diff(BASE, BASE.replace("return a + b + n;", "return a + b + n + 0;"))
    assert fc == FunctionChanges(("com.x.Calc.add(int,int)",), (), ())


@pytest.mark.parametrize("edit", [
    lambda s: s.replace("/** Calculator. */", "/** Calculator, now with more docs. */"),   # javadoc
    lambda s: s.replace("{ return 2 * x; }", "{\n            // doubled\n            return 2 * x;\n        }"),
    lambda s: s.replace("public int add(int a, int b)", "public int add(int a,   int b)"),     # formatting
])
def test_comments_javadoc_and_formatting_are_ignored(edit):
    new = edit(BASE)
    assert new != BASE
    assert diff(BASE, new) == FunctionChanges()


def test_constructor_static_block_nested_and_lambda():
    assert diff(BASE, BASE.replace("this.n = n;", "this.n = n * 2;")).changed == ("com.x.Calc.<init>(int)",)
    assert diff(BASE, BASE.replace('"init"', '"boot"')).changed == ("com.x.Calc.<clinit>",)
    assert diff(BASE, BASE.replace("2 * x", "3 * x")).changed == ("com.x.Calc$Helper.twice(int)",)
    assert diff(BASE, BASE.replace("println(1)", "println(2)")).changed == ("com.x.Calc.run()",)


def test_class_level_and_import_changes():
    assert diff(BASE, BASE.replace("private int n = 1;", "private int n = 1000;")).changed == ("com.x.Calc",)
    assert diff(BASE, BASE.replace("import java.util.List;", "import java.util.List;\nimport java.util.Map;")).changed \
        == ("com.x.Calc.<imports>",)


def test_added_deleted_and_signature_change():
    fc = diff(BASE, BASE.replace("public int add(int a, int b)", "public int add(long a, int b)"))
    assert fc.added == ("com.x.Calc.add(long,int)",) and fc.deleted == ("com.x.Calc.add(int,int)",)


def test_moving_methods_is_not_a_change():
    a = "class A {\n  void f() { g(); }\n  void g() { }\n}\n"
    b = "class A {\n  void g() { }\n  void f() { g(); }\n}\n"
    assert diff(a, b) == FunctionChanges()


def test_interface_enum_record_and_varargs():
    src = """\
    package p;
    interface Shape { double area(); default String name() { return "s"; } }
    enum Op { ADD; int apply(int a, int b) { return a + b; } }
    record Point(int x, int y) { Point { if (x < 0) throw new IllegalArgumentException(); } }
    class V { void log(String fmt, Object... args) { } void arr(int[] a, String s[]) { } }
    """
    keys = set(JAVA.parse(j(src)).entries)
    assert {"p.Shape.area()", "p.Shape.name()", "p.Op.apply(int,int)", "p.Point.<init>(int,int)",
            "p.V.log(String,Object...)", "p.V.arr(int[],String[])"} <= keys


def test_default_package_and_operator_change():
    a = "class A { int f(int x) { return x + 1; } }\n"
    assert diff(a, a.replace("x + 1", "x - 1"), path="A.java").changed == ("A.f(int)",)


def test_syntax_error_raises():
    with pytest.raises(SyntaxError):
        JAVA.parse(b"class A { void f( { }")


def test_java_through_the_gateway(fresh_fixture_repo):
    repo = fresh_fixture_repo
    path = repo.path / "src" / "main" / "java" / "com" / "x"
    path.mkdir(parents=True)
    (path / "Calc.java").write_text(BASE, encoding="utf-8", newline="\n")
    repo.git("add", "-A")
    repo.git("commit", "-qm", "Add Java calc")
    (path / "Calc.java").write_text(BASE.replace("return a + b + n;", "for (int i = 0; i < 1000; i++) {} return a + b + n;"),
                                    encoding="utf-8", newline="\n")
    (path / "Broken.java").write_text("class Broken { void f( { }", encoding="utf-8")
    repo.git("add", "-A")
    repo.git("commit", "-qm", "Slow add + broken file")

    added, slow = Gateway(repo.path, cache=False).get_candidates(BAD_TAG, "HEAD")
    assert "com.x.Calc.add(int,int)" in added.added_functions
    assert slow.changed_functions == ("com.x.Calc.add(int,int)",)
    assert slow.unanalyzed_files == ("src/main/java/com/x/Broken.java",)


def test_mixed_python_and_java_commit(fresh_fixture_repo):
    repo = fresh_fixture_repo
    (repo.path / "A.java").write_text("class A { int f() { return 1; } }\n", encoding="utf-8")
    (repo.path / "mathops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8", newline="\n")
    repo.git("add", "-A")
    repo.git("commit", "-qm", "both languages")
    (c,) = Gateway(repo.path, cache=False).get_candidates(BAD_TAG, "HEAD")
    assert c.added_functions == ("A.f()",)
    assert c.changed_functions == ("mathops.add",)
