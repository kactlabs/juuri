"""Turtle / OWL export.

The output is Turtle using the OWL 2 vocabulary, which Protégé, rdflib and
triple stores can load. Every name becomes an IRI under a base namespace:

  <base>entity/<name>   entities (and merged aliases)
  <base>class/<name>    schema classes
  <base>rel/<name>      relations
  <base>vocab/...       juuri's own terms (source, chunk, confidence, ...)

Names are percent-encoded, so any name gives a valid, distinct IRI.
"""

import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote

from .temporal import DateLike

_BASE = re.compile(r'^https?://[^\s<>"{}|\\^`#?]+/$')

_PREFIXES = (
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
)

_PROVENANCE_PROPERTIES = ("source", "chunk", "confidence", "validFrom", "validTo", "recordedAt")


def _literal(text: str) -> str:
    """A Turtle string literal with the characters Turtle requires escaped."""
    out = []
    for ch in str(text):
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _datetime_literal(stamp: str) -> str:
    """xsd:dateTime if `stamp` is a full ISO timestamp, otherwise a plain string.

    The store accepts any string as a timestamp, so unreadable ones are kept
    as text and never raise.
    """
    try:
        parsed = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return _literal(stamp)
    if len(stamp) <= 10:  # a bare date is not an xsd:dateTime
        return _literal(stamp)
    return f'"{parsed.isoformat()}"^^xsd:dateTime'


def _block(subject: str, pairs: List[Tuple[str, str]]) -> str:
    lines = [f"{subject} {pairs[0][0]} {pairs[0][1]}"]
    lines += [f"    {pred} {obj}" for pred, obj in pairs[1:]]
    return " ;\n".join(lines) + " ."


def to_turtle(
    store,
    path: Optional[str] = None,
    base: str = "http://example.org/juuri/",
    include_provenance: bool = True,
    as_of: DateLike = None,
    include_undated: bool = True,
) -> str:
    """Export the store as Turtle. See TripleStore.to_turtle()."""
    if not isinstance(base, str) or not _BASE.match(base):
        raise ValueError("base must be an http(s) IRI that ends with '/', with no '#' or '?', e.g. http://example.org/juuri/")

    def iri(kind: str, name: str) -> str:
        return f"<{base}{kind}/{quote(name, safe='')}>"

    schema = store.schema
    triples = sorted(store.find(as_of=as_of, include_undated=include_undated))
    entities = sorted({t[0] for t in triples} | {t[2] for t in triples})
    used_relations = sorted({t[1] for t in triples})

    blocks: List[str] = [
        "\n".join(f"@prefix {name}: <{uri}> ." for name, uri in _PREFIXES) + f"\n@prefix juuri: <{base}vocab/> .",
        _block(f"<{base}>", [("a", "owl:Ontology"), ("rdfs:comment", _literal("Exported from juuri."))]),
    ]

    declared = set()
    if schema is not None:
        for cls in sorted(schema.classes):
            pairs = [("a", "owl:Class"), ("rdfs:label", _literal(cls))]
            parent = schema.subclasses.get(cls)
            if parent:
                pairs.append(("rdfs:subClassOf", iri("class", parent)))
            blocks.append(_block(iri("class", cls), pairs))
        for rel in sorted(schema.relations):
            declared.add(rel)
            domain, rng = schema.relations[rel]
            types = ["owl:ObjectProperty"]
            extra: List[Tuple[str, str]] = []
            kind, _, arg = schema.rules.get(rel, "").partition(":")
            if kind == "transitive":
                types.append("owl:TransitiveProperty")
            elif kind == "symmetric":
                types.append("owl:SymmetricProperty")
            elif kind == "inverse":
                extra.append(("owl:inverseOf", iri("rel", arg)))
            elif kind == "implies":
                extra.append(("rdfs:subPropertyOf", iri("rel", arg)))
            for constraint in schema.constraints.get(rel, ()):
                if constraint == "functional":
                    types.append("owl:FunctionalProperty")
                elif constraint == "inverse_functional":
                    types.append("owl:InverseFunctionalProperty")
                elif constraint.startswith("exclusive:"):
                    extra.append(("owl:propertyDisjointWith", iri("rel", constraint.split(":", 1)[1])))
            pairs = [("a", ", ".join(types)), ("rdfs:label", _literal(rel)),
                     ("rdfs:domain", iri("class", domain)), ("rdfs:range", iri("class", rng))] + extra
            blocks.append(_block(iri("rel", rel), pairs))
    for rel in used_relations:
        if rel not in declared:
            blocks.append(_block(iri("rel", rel), [("a", "owl:ObjectProperty"), ("rdfs:label", _literal(rel))]))

    if include_provenance and triples:
        for name in _PROVENANCE_PROPERTIES:
            blocks.append(_block(f"juuri:{name}", [("a", "owl:DatatypeProperty")]))

    for name in entities:
        cls = store.type_of(name)
        types = ["owl:NamedIndividual"] + ([iri("class", cls)] if cls else [])
        blocks.append(_block(iri("entity", name), [("a", ", ".join(types)), ("rdfs:label", _literal(name))]))

    exported = set(entities)
    for alias, canonical in sorted(set(store._aliases.values())):
        if canonical in exported and alias != canonical and alias not in exported:
            blocks.append(_block(iri("entity", alias), [
                ("a", "owl:NamedIndividual"), ("rdfs:label", _literal(alias)), ("owl:sameAs", iri("entity", canonical)),
            ]))

    if triples:
        blocks.append("\n".join(f"{iri('entity', s)} {iri('rel', p)} {iri('entity', o)} ." for s, p, o in triples))

    if include_provenance:
        for s, p, o in triples:
            for rec in store.provenance(s, p, o):
                pairs = [("a", "rdf:Statement"), ("rdf:subject", iri("entity", s)),
                         ("rdf:predicate", iri("rel", p)), ("rdf:object", iri("entity", o))]
                if rec.source:
                    pairs.append(("juuri:source", _literal(rec.source)))
                if rec.chunk_id:
                    pairs.append(("juuri:chunk", _literal(rec.chunk_id)))
                if rec.confidence is not None:
                    pairs.append(("juuri:confidence", f'"{format(Decimal(repr(float(rec.confidence))), "f")}"^^xsd:decimal'))
                if rec.valid_from:
                    pairs.append(("juuri:validFrom", f'"{rec.valid_from}"^^xsd:date'))
                if rec.valid_to:
                    pairs.append(("juuri:validTo", f'"{rec.valid_to}"^^xsd:date'))
                pairs.append(("juuri:recordedAt", _datetime_literal(rec.timestamp)))
                blocks.append(_block("[]", pairs))

    text = "\n\n".join(blocks) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def to_networkx(store, as_of: DateLike = None, include_undated: bool = True):
    """Export the store as a networkx.MultiDiGraph. See TripleStore.to_networkx()."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("to_networkx() needs networkx: pip install networkx") from exc

    graph = nx.MultiDiGraph()
    for s, p, o in sorted(store.find(as_of=as_of, include_undated=include_undated)):
        records = store.provenance(s, p, o)
        confidences = [r.confidence for r in records if r.confidence is not None]
        for name in (s, o):
            if name not in graph:
                graph.add_node(name, type=store.type_of(name), aliases=store.aliases(name))
        graph.add_edge(
            s, o, key=p,
            predicate=p,
            sources=sorted({r.source for r in records if r.source}),
            chunks=sorted({r.chunk_id for r in records if r.chunk_id}),
            confidence=max(confidences) if confidences else None,
            periods=sorted(
                {(r.valid_from, r.valid_to) for r in records if r.valid_from or r.valid_to},
                key=lambda period: (period[0] or "", period[1] or "9999-12-31"),
            ),
            provenance=tuple(records),
        )
    return graph
