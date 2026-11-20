"""Structured queries, and text-to-query: the LLM writes JSON, juuri validates and runs it.

Query format (a JSON object):

  {"op": "match", "patterns": [{"subject": "?p", "predicate": "CAPTAINED", "object": "CSK"}],
   "as_of": "2010-05-01", "limit": 50}
  {"op": "root_cause", "entity": "Login timeout", "relation": "CAUSED_BY", "max_hops": 10}
  {"op": "blast_radius", "entity": "Disk full", "relation": "CAUSED_BY"}
  {"op": "paths", "source": "A", "target": "D", "predicates": ["CAUSED_BY"], "max_hops": 4, "direction": "out"}

A term starting with "?" in a match pattern is a variable. "as_of" is optional on every op.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .extract import load_prompt
from .paths import Chain
from .provenance import Provenance
from .resolve import name_score, tokens
from .retrieve import Hit, _refs, _validity_note, estimate_tokens
from .schema import Schema
from .temporal import parse_as_of

Triple = Tuple[str, str, str]

MAX_PATTERNS = 5
MAX_LIMIT = 1000
MAX_INTERMEDIATE = 10000

_VAR = re.compile(r"^\?[A-Za-z_]\w*$")
_DIRECTIONS = ("out", "in", "both")
_KEYS = {
    "match": {"op", "patterns", "as_of", "limit"},
    "root_cause": {"op", "entity", "relation", "max_hops", "as_of"},
    "blast_radius": {"op", "entity", "relation", "max_hops", "as_of"},
    "paths": {"op", "source", "target", "predicates", "max_hops", "direction", "as_of"},
}


class QueryError(ValueError):
    """Raised when a query is malformed, breaks the schema, or names an unknown entity."""


@dataclass(frozen=True)
class Row:
    """One answer to a match query.

    bindings: variable name (with "?") -> entity.
    triples / provenance: the facts that produced the row, one per pattern.
    """

    bindings: Dict[str, str]
    triples: Tuple[Triple, ...]
    provenance: Tuple[Tuple[Provenance, ...], ...]


@dataclass
class QueryResult:
    """Result of a query.

    query: the validated query with entity names replaced by the graph's names.
    rows: answers to a match query.
    chains: results of root_cause, blast_radius and paths.
    truncated: a match had more rows than `limit`.
    """

    query: Dict[str, Any]
    rows: List[Row] = field(default_factory=list)
    chains: List[Chain] = field(default_factory=list)
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.rows) + len(self.chains)

    def to_context(self, tokens: int = 2000, count_tokens: Optional[Callable[[str], int]] = None) -> str:
        """Cited text for a prompt, within a token budget. Empty string if there are no results.

        Match rows give their facts (each once, in row order). Chains are
        added whole after the facts. Anything that does not fit is dropped.
        """
        if tokens < 1:
            raise ValueError("tokens must be positive")
        count = count_tokens or estimate_tokens
        used = 0
        parts: List[str] = []
        seen = set()
        facts: List[str] = []
        for row in self.rows:
            for triple, prov in zip(row.triples, row.provenance):
                if triple in seen:
                    continue
                seen.add(triple)
                line = f"- {triple[0]} {triple[1]} {triple[2]}{_validity_note(prov)} [{', '.join(_refs(Hit(triple, 0, 0.0, prov)))}]"
                if used + count(line) <= tokens:
                    used += count(line)
                    facts.append(line)
        if facts:
            parts.append("\n".join(["QUERY RESULT:", *facts]))
        for i, chain in enumerate(self.chains, 1):
            text = chain.to_context(f"CHAIN {i}")
            if used + count(text) <= tokens:
                used += count(text)
                parts.append(text)
        return "\n\n".join(parts)


def is_var(term: str) -> bool:
    return term.startswith("?")


def build_prompt(question: str, schema: Schema, today: str) -> str:
    relations = "\n".join(
        f"- {name}: {dom} -> {rng}" for name, (dom, rng) in sorted(schema.relations.items())
    )
    return load_prompt("query").format(relations=relations, today=today, question=question)


def parse_query(raw: str) -> Dict[str, Any]:
    """Parse the LLM's JSON text into a query dict. Raises QueryError if it is not a JSON object."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QueryError(f"LLM output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise QueryError("query must be a JSON object")
    if "error" in data and "op" not in data:
        raise QueryError(f"LLM could not write a query: {data['error']}")
    return data


def _text(spec: Dict[str, Any], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise QueryError(f"'{key}' must be a non-empty string")
    return value.strip()


def _int(spec: Dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = spec.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise QueryError(f"'{key}' must be an integer from {low} to {high}")
    return value


def _relation(name: Any, schema: Optional[Schema]) -> str:
    if not isinstance(name, str) or not name.strip():
        raise QueryError("relation names must be non-empty strings")
    name = name.strip()
    if is_var(name):
        raise QueryError(f"relation variables are not supported: '{name}'")
    if schema is not None and name not in schema.relations:
        raise QueryError(f"undeclared relation '{name}'")
    return name


def validate_query(spec: Dict[str, Any], schema: Optional[Schema]) -> Dict[str, Any]:
    """Check a query against the format and schema, and return a cleaned copy.

    Unknown keys are rejected, so a misspelt option is an error and not
    silently ignored. Entity names are not linked here (see run_query).
    """
    if not isinstance(spec, dict):
        raise QueryError("query must be a JSON object")
    op = spec.get("op")
    if op not in _KEYS:
        raise QueryError(f"'op' must be one of {sorted(_KEYS)}, got {op!r}")
    extra = set(spec) - _KEYS[op]
    if extra:
        raise QueryError(f"unknown keys for '{op}': {sorted(extra)}")
    out: Dict[str, Any] = {"op": op}
    if spec.get("as_of") is not None:
        try:
            out["as_of"] = parse_as_of(spec["as_of"])
        except (ValueError, TypeError) as exc:
            raise QueryError(f"invalid as_of: {exc}") from exc
    if op == "match":
        patterns = spec.get("patterns")
        if not isinstance(patterns, list) or not 1 <= len(patterns) <= MAX_PATTERNS:
            raise QueryError(f"'patterns' must be a list of 1 to {MAX_PATTERNS} patterns")
        clean = []
        for pat in patterns:
            if not isinstance(pat, dict) or set(pat) != {"subject", "predicate", "object"}:
                raise QueryError("each pattern needs exactly: subject, predicate, object")
            for key in ("subject", "object"):
                term = _text(pat, key)
                if is_var(term) and not _VAR.match(term):
                    raise QueryError(f"invalid variable '{term}'")
                pat = {**pat, key: term}
            clean.append({"subject": pat["subject"], "predicate": _relation(pat["predicate"], schema), "object": pat["object"]})
        out["patterns"] = clean
        out["limit"] = _int(spec, "limit", 50, 1, MAX_LIMIT)
    elif op in ("root_cause", "blast_radius"):
        out["entity"] = _text(spec, "entity")
        out["relation"] = _relation(spec.get("relation", "CAUSED_BY"), schema)
        out["max_hops"] = _int(spec, "max_hops", 10, 1, 100)
    else:
        out["source"] = _text(spec, "source")
        out["target"] = _text(spec, "target")
        preds = spec.get("predicates")
        if preds is not None:
            if not isinstance(preds, list) or not preds:
                raise QueryError("'predicates' must be a non-empty list")
            out["predicates"] = [_relation(p, schema) for p in preds]
        out["max_hops"] = _int(spec, "max_hops", 4, 1, 100)
        direction = spec.get("direction", "out")
        if direction not in _DIRECTIONS:
            raise QueryError(f"'direction' must be one of {list(_DIRECTIONS)}")
        out["direction"] = direction
    return out


def link_entity(store, name: str) -> str:
    """Map a name from the question to an entity in the graph.

    Tries, in order: the exact name or a merged alias, the same words in
    any case or punctuation, then names containing all its words (such as
    "Dhoni" for "MS Dhoni") or spelled almost the same. Raises QueryError
    if nothing matches or several entities match, and never guesses.
    """
    entities = store.entities()
    known = set(entities)
    canon = store.canonical(name)
    if canon in known:
        return canon
    wanted = tokens(name)
    same = [e for e in entities if tokens(e) == wanted]
    if len(same) == 1:
        return same[0]
    candidates = same or [e for e in entities if name_score(name, e, 0.9) is not None]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise QueryError(f"unknown entity '{name}'")
    raise QueryError(f"ambiguous entity '{name}': could be {sorted(candidates)[:5]}")


def run_query(store, spec: Any) -> QueryResult:
    """Validate and run a query (a dict or a JSON string) against `store`."""
    if isinstance(spec, str):
        spec = parse_query(spec)
    query = validate_query(spec, store.schema)
    as_of = query.get("as_of")
    result = QueryResult(query=query)
    op = query["op"]
    if op == "match":
        _run_match(store, query, result)
    elif op in ("root_cause", "blast_radius"):
        query["entity"] = link_entity(store, query["entity"])
        fn = store.root_cause if op == "root_cause" else store.blast_radius
        result.chains = fn(query["entity"], query["relation"], query["max_hops"], as_of=as_of)
    else:
        query["source"] = link_entity(store, query["source"])
        query["target"] = link_entity(store, query["target"])
        if query["source"] == query["target"]:
            raise QueryError("source and target are the same entity")
        result.chains = store.paths(
            query["source"], query["target"], query["max_hops"],
            query.get("predicates"), query["direction"], as_of=as_of,
        )
    return result


def _run_match(store, query: Dict[str, Any], result: QueryResult) -> None:
    as_of = query.get("as_of")
    patterns: List[Tuple[str, str, str]] = []
    for pat in query["patterns"]:
        s, o = pat["subject"], pat["object"]
        s = s if is_var(s) else link_entity(store, s)
        o = o if is_var(o) else link_entity(store, o)
        patterns.append((s, pat["predicate"], o))
    query["patterns"] = [{"subject": s, "predicate": p, "object": o} for s, p, o in patterns]

    rows: List[Tuple[Dict[str, str], Tuple[Triple, ...]]] = [({}, ())]
    bound: set = set()
    remaining = list(patterns)
    while remaining:
        def score(pat: Tuple[str, str, str]) -> int:
            return sum(1 for t in (pat[0], pat[2]) if not is_var(t) or t in bound)

        pat = max(remaining, key=score)  # max() keeps the first of equal scores
        remaining.remove(pat)
        s_term, pred, o_term = pat
        nxt: List[Tuple[Dict[str, str], Tuple[Triple, ...]]] = []
        for bindings, triples in rows:
            s = bindings.get(s_term) if is_var(s_term) else s_term
            o = bindings.get(o_term) if is_var(o_term) else o_term
            for triple in store.find(s, pred, o, as_of=as_of):
                new = dict(bindings)
                if is_var(s_term) and new.setdefault(s_term, triple[0]) != triple[0]:
                    continue
                if is_var(o_term) and new.setdefault(o_term, triple[2]) != triple[2]:
                    continue
                nxt.append((new, triples + (triple,)))
                if len(nxt) > MAX_INTERMEDIATE:
                    raise QueryError("query is too broad: more than 10000 intermediate rows")
        rows = nxt
        bound.update(t for t in (s_term, o_term) if is_var(t))
        if not rows:
            break
    rows.sort(key=lambda r: (sorted(r[0].items()), r[1]))
    limit = query["limit"]
    result.truncated = len(rows) > limit
    for bindings, triples in rows[:limit]:
        prov = tuple(tuple(store.provenance(*t)) for t in triples)
        result.rows.append(Row(bindings, triples, prov))
