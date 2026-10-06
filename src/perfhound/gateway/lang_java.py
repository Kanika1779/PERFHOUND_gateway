"""Java support for the Code Analyzer (tree-sitter-java).

Same idea as Python: compare a fingerprint of every method before/after a
commit; comments, Javadoc and formatting are ignored.

Names follow what JVM stack traces and profilers (JFR, async-profiler) show,
so they can later be matched against profiler output:

    com.x.Calc.add(int,List)          method; parameter types with generics erased
    com.x.Calc.<init>(int)            constructor
    com.x.Calc.<clinit>               all static { } blocks of the class
    com.x.Outer$Inner.run()           nested type -> '$' (JVM binary name)
    com.x.Calc                        class-level code: fields, instance
                                      initializers, extends/implements, annotations
    com.x.Calc.<imports>              package / import lines of Calc.java

Overloads are distinct (`add(int)` vs `add(String)`). Lambdas, anonymous and
local classes are part of the enclosing method (a change there marks that
method). A file with a syntax error raises SyntaxError, like Python.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Callable

from .analyzer import Definitions, _Entry

try:
    import tree_sitter_java as _tsjava
    from tree_sitter import Language, Parser

    _LANGUAGE = Language(_tsjava.language())
except ImportError:  # pragma: no cover - reported when a Java file is met
    _LANGUAGE = None

_TYPE_DECLS = {
    "class_declaration", "interface_declaration", "enum_declaration",
    "record_declaration", "annotation_type_declaration",
}
_BODIES = {"class_body", "interface_body", "enum_body", "enum_body_declarations", "annotation_type_body"}
_MEMBER_CALLABLES = {"method_declaration", "constructor_declaration", "compact_constructor_declaration"}
_COMMENTS = {"line_comment", "block_comment"}
_GENERICS = re.compile(r"<[^<>]*>")
_ANNOTATION = re.compile(r"@[\w.]+(\([^()]*\))?\s*")


def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _erase(type_text: str) -> str:
    """'final @NonNull Map<String, List<Integer>>' -> 'Map'."""
    t = _ANNOTATION.sub("", type_text)
    prev = None
    while prev != t:
        prev, t = t, _GENERICS.sub("", t)
    return re.sub(r"\s+", "", t)


def _param_types(src: bytes, params_node) -> str:
    if params_node is None:
        return ""
    types = []
    for p in params_node.named_children:
        if p.type == "formal_parameter":
            t = _erase(_text(src, p.child_by_field_name("type")))
            dims = p.child_by_field_name("dimensions")
            types.append(t + (_text(src, dims).replace(" ", "") if dims is not None else ""))
        elif p.type == "spread_parameter":   # varargs: String... xs
            typ = next((c for c in p.named_children if c.type not in ("modifiers", "variable_declarator")), None)
            types.append(_erase(_text(src, typ)) + "..." if typ is not None else "...")
        # receiver_parameter (Foo this) is not part of the JVM signature
    return ",".join(types)


def _fingerprint(node, src: bytes, skip: frozenset = frozenset()) -> str:
    """Structure + tokens of a node, without comments and whitespace.
    Children whose type is in `skip` (only directly inside type bodies) are left out."""
    out: list[str] = []

    def walk(n, in_body: bool) -> None:
        if n.type in _COMMENTS:
            return
        if in_body and n.type in skip:
            return
        if n.child_count == 0:
            out.append(_text(src, n))
            return
        out.append("(" + n.type)
        child_in_body = n.type in _BODIES
        for c in n.children:
            walk(c, child_in_body)
        out.append(")")

    walk(node, False)
    return " ".join(out)


class JavaSupport:
    name = "java"
    extensions = (".java",)

    def __init__(self) -> None:
        self._parser = Parser(_LANGUAGE) if _LANGUAGE is not None else None

    def parse(self, source: bytes) -> Definitions:
        if self._parser is None:
            raise ValueError("Java support needs: pip install tree-sitter tree-sitter-java")
        tree = self._parser.parse(source)
        root = tree.root_node
        if root.has_error:
            raise SyntaxError("Java source has syntax errors")

        entries: dict[str, _Entry] = {}
        classes: set[str] = set()
        package = ""
        header_nodes = []
        for child in root.named_children:
            if child.type == "package_declaration":
                name = next((c for c in child.named_children if c.type in ("scoped_identifier", "identifier")), None)
                package = _text(source, name) if name is not None else ""
                header_nodes.append(child)
            elif child.type == "import_declaration":
                header_nodes.append(child)

        skip_in_residual = frozenset(_MEMBER_CALLABLES | _TYPE_DECLS | {"static_initializer"})

        def visit_type(node, qualified: str) -> None:
            classes.add(qualified)
            # class-level residual: everything except methods, ctors, nested types, static blocks
            entries[qualified] = _Entry(_text(source, node), ("residual", node))
            body = node.child_by_field_name("body")
            if body is None:
                return
            members = list(body.named_children)
            for m in list(members):   # enum methods live one level deeper
                if m.type == "enum_body_declarations":
                    members.extend(m.named_children)
            static_blocks = []
            for m in members:
                if m.type in _TYPE_DECLS:
                    inner = m.child_by_field_name("name")
                    visit_type(m, f"{qualified}${_text(source, inner)}")
                elif m.type == "method_declaration":
                    name = _text(source, m.child_by_field_name("name"))
                    sig = _param_types(source, m.child_by_field_name("parameters"))
                    entries[f"{qualified}.{name}({sig})"] = _Entry(_text(source, m), m)
                elif m.type == "constructor_declaration":
                    sig = _param_types(source, m.child_by_field_name("parameters"))
                    entries[f"{qualified}.<init>({sig})"] = _Entry(_text(source, m), m)
                elif m.type == "compact_constructor_declaration":
                    sig = _param_types(source, node.child_by_field_name("parameters"))
                    entries[f"{qualified}.<init>({sig})"] = _Entry(_text(source, m), m)
                elif m.type == "static_initializer":
                    static_blocks.append(m)
            if static_blocks:
                entries[f"{qualified}.<clinit>"] = _Entry(
                    "\x00".join(_text(source, b) for b in static_blocks), ("many", tuple(static_blocks))
                )

        prefix = package + "." if package else ""
        for child in root.named_children:
            if child.type in _TYPE_DECLS:
                visit_type(child, prefix + _text(source, child.child_by_field_name("name")))

        entries["<module>"] = _Entry("\x00".join(_text(source, n) for n in header_nodes),
                                     ("many", tuple(header_nodes)))

        def fingerprint(item) -> str:
            if isinstance(item, tuple) and item[0] == "residual":
                return _fingerprint(item[1], source, skip_in_residual)
            if isinstance(item, tuple) and item[0] == "many":
                return "|".join(_fingerprint(n, source) for n in item[1])
            return _fingerprint(item, source)

        return Definitions(entries, frozenset(classes), fingerprint, namespace=package)

    def qualifier(self, path: str, old: Definitions, new: Definitions) -> Callable[[str], str]:
        package = new.namespace or old.namespace
        stem = PurePosixPath(path.replace("\\", "/")).stem

        def q(key: str) -> str:
            if key == "<module>":
                return f"{package + '.' if package else ''}{stem}.<imports>"
            return key

        return q
