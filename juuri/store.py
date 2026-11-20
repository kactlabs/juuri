"""Compact triple store: interned strings, columnar triples, S/P/O indexes."""

from array import array
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple

from .provenance import Provenance
from .schema import Schema, SchemaError

Triple = Tuple[str, str, str]


class Interner:
    """Maps strings to dense integer IDs and back."""

    def __init__(self) -> None:
        self._to_id: Dict[str, int] = {}
        self._to_str: list = []

    def intern(self, value: str) -> int:
        idx = self._to_id.get(value)
        if idx is None:
            idx = len(self._to_str)
            self._to_id[value] = idx
            self._to_str.append(value)
        return idx

    def lookup(self, value: str) -> Optional[int]:
        """Return the ID for value, or None if it was never interned."""
        return self._to_id.get(value)

    def string(self, idx: int) -> str:
        return self._to_str[idx]

    def __len__(self) -> int:
        return len(self._to_str)


class TripleStore:
    """Stores (subject, predicate, object) triples.

    Triples are three parallel integer columns. Each of the subject,
    predicate and object columns has an index from term ID to the IDs of
    the triples that use it, so lookups are fast in any direction.
    """

    def __init__(self, schema: Optional[Schema] = None) -> None:
        self.schema = schema
        self._types: Dict[str, str] = {}
        self._terms = Interner()
        self._s = array("I")
        self._p = array("I")
        self._o = array("I")
        self._seen: set = set()
        self._by_s: Dict[int, array] = {}
        self._by_p: Dict[int, array] = {}
        self._by_o: Dict[int, array] = {}
        self._prov: Dict[int, List[Provenance]] = {}
        self._chunks: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._s)

    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        subject_type: Optional[str] = None,
        object_type: Optional[str] = None,
        source: Optional[str] = None,
        chunk_id: Optional[str] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Add a triple and return its ID. Duplicates return the existing ID.

        With a schema, the relation and entity types are checked first.
        An entity's type is remembered after its first use.
        Every call records a Provenance entry for the triple (timestamp
        defaults to now, UTC). Repeating a fact from a new source adds
        another entry. An identical entry is not stored twice.
        """
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        for name, value in (("subject", subject), ("predicate", predicate), ("object", obj)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema is not None:
            st = self._resolve_type(subject, subject_type)
            ot = self._resolve_type(obj, object_type)
            self.schema.check(st, predicate, ot)
            self._types[subject] = st
            self._types[obj] = ot
        s = self._terms.intern(subject)
        p = self._terms.intern(predicate)
        o = self._terms.intern(obj)
        key = (s, p, o)
        if key in self._seen:
            tid = self._find_id(s, p, o)
        else:
            tid = len(self._s)
            self._seen.add(key)
            self._s.append(s)
            self._p.append(p)
            self._o.append(o)
            for index, term in ((self._by_s, s), (self._by_p, p), (self._by_o, o)):
                index.setdefault(term, array("I")).append(tid)
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        record = Provenance(source, chunk_id, confidence, ts)
        records = self._prov.setdefault(tid, [])
        if not any(r.source == source and r.chunk_id == chunk_id and r.confidence == confidence for r in records):
            records.append(record)
        return tid

    def provenance(self, subject: str, predicate: str, obj: str) -> List[Provenance]:
        """Return all provenance records for a triple. Empty if the triple is unknown."""
        ids = list(self._candidates(subject, predicate, obj))
        return list(self._prov.get(ids[0], [])) if ids else []

    def chunk_text(self, chunk_id: str) -> Optional[str]:
        """Return the text of an ingested chunk, or None if unknown."""
        return self._chunks.get(chunk_id)

    def ingest(self, docs, llm, max_chars: int = 2000, min_confidence: float = 0.0, source: Optional[str] = None) -> int:
        """Extract triples from text with an LLM and add them. Returns the count of new triples.

        docs is a string, a list of strings, or a dict {source_name: text}.
        For a string or list, `source` names the document (None if not given).
        Each fact records its source, chunk ID ("<source>#<n>"), confidence
        and timestamp. Requires a schema. Triples that fail the schema or
        fall below min_confidence are skipped.
        """
        from .extract import extract_triples

        if self.schema is None:
            raise SchemaError("ingest requires a schema")
        if isinstance(docs, str):
            docs = [(source, docs)]
        elif isinstance(docs, dict):
            docs = list(docs.items())
        else:
            docs = [(source, d) for d in docs]
        before = len(self)
        for name, doc in docs:
            for t in extract_triples(doc, self.schema, llm, max_chars):
                if t["confidence"] < min_confidence:
                    continue
                chunk_id = f"{name}#{t['chunk_index']}" if name else f"#{t['chunk_index']}"
                try:
                    self.add(
                        t["subject"], t["predicate"], t["object"],
                        t["subject_type"], t["object_type"],
                        source=name, chunk_id=chunk_id, confidence=t["confidence"],
                    )
                except SchemaError:
                    continue  # e.g. entity already typed differently
                self._chunks.setdefault(chunk_id, t["chunk"])
        return len(self) - before

    def _resolve_type(self, entity: str, given: Optional[str]) -> Optional[str]:
        known = self._types.get(entity)
        if given is not None and known is not None and given != known:
            raise SchemaError(f"'{entity}' is already typed '{known}', got '{given}'")
        return given if given is not None else known

    def type_of(self, entity: str) -> Optional[str]:
        return self._types.get(entity)

    def _find_id(self, s: int, p: int, o: int) -> int:
        for tid in self._by_s[s]:
            if self._p[tid] == p and self._o[tid] == o:
                return tid
        raise KeyError((s, p, o))

    def get(self, tid: int) -> Triple:
        t = self._terms.string
        return t(self._s[tid]), t(self._p[tid]), t(self._o[tid])

    def find(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> Iterator[Triple]:
        """Yield triples matching the given terms. None acts as a wildcard."""
        for tid in self._candidates(subject, predicate, obj):
            yield self.get(tid)

    def _candidates(self, subject, predicate, obj) -> Iterator[int]:
        constraints = []
        for value, index, column in (
            (subject, self._by_s, self._s),
            (predicate, self._by_p, self._p),
            (obj, self._by_o, self._o),
        ):
            if value is None:
                continue
            term = self._terms.lookup(value)
            if term is None or term not in index:
                return
            constraints.append((index[term], column, term))
        if not constraints:
            yield from range(len(self._s))
            return
        # Scan the shortest index and check the remaining constraints.
        constraints.sort(key=lambda c: len(c[0]))
        smallest, rest = constraints[0][0], constraints[1:]
        for tid in smallest:
            if all(column[tid] == term for _, column, term in rest):
                yield tid
