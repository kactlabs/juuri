"""Compact triple store: interned strings, columnar triples, S/P/O indexes."""

from array import array
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple

from typing import Sequence

from .provenance import Provenance
from .retrieve import Hit, Hits, top_seeds
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
        self._emb: Dict[str, List[float]] = {}

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

    def entities(self) -> List[str]:
        """All entity names (every term used as a subject or object)."""
        ids = set(self._by_s) | set(self._by_o)
        return sorted(self._terms.string(i) for i in ids)

    def embed(self, embedder, batch_size: int = 64) -> int:
        """Embed every entity that has no vector yet. Returns the number embedded."""
        todo = [e for e in self.entities() if e not in self._emb]
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            for name, vec in zip(batch, embedder.embed(batch)):
                self._emb[name] = list(vec)
        return len(todo)

    def retrieve(
        self,
        query: str,
        embedder,
        seeds: int = 3,
        hops: int = 1,
        expand: Optional[Sequence[str]] = None,
        direction: str = "both",
        min_score: float = 0.0,
    ) -> Hits:
        """Find seed entities by vector similarity, then expand `hops` steps.

        expand: only follow these relation names (default: all).
        direction: "out" follows subject -> object, "in" the reverse, "both" either way.
        Seeds scoring below min_score are ignored. Call embed() first.
        Triples are ordered by hop, then by seed score.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be 'out', 'in' or 'both'")
        if hops < 0 or seeds < 1:
            raise ValueError("hops must be >= 0 and seeds >= 1")
        if not self._emb:
            raise ValueError("no embeddings: call embed() first")
        qvec = embedder.embed([query])[0]
        seed_list = [s for s in top_seeds(qvec, self._emb, seeds) if s[1] >= min_score]
        allowed = None if expand is None else set(expand)

        best: Dict[int, Tuple[int, float]] = {}  # triple id -> (hop, seed score)
        for seed, score in seed_list:
            frontier = {seed}
            visited = {seed}
            for hop in range(hops + 1):
                nxt = set()
                for ent in frontier:
                    for tid in self._touching(ent, direction):
                        if allowed is not None and self._terms.string(self._p[tid]) not in allowed:
                            continue
                        prev = best.get(tid)
                        if prev is None or (hop, -score) < (prev[0], -prev[1]):
                            best[tid] = (hop, score)
                        for term in (self._s[tid], self._o[tid]):
                            name = self._terms.string(term)
                            if name not in visited:
                                visited.add(name)
                                nxt.add(name)
                frontier = nxt
                if not frontier:
                    break
        ordered = sorted(best.items(), key=lambda kv: (kv[1][0], -kv[1][1], kv[0]))
        hits = [
            Hit(self.get(tid), hop, score, tuple(self._prov.get(tid, [])))
            for tid, (hop, score) in ordered
        ]
        chunks = {
            p.chunk_id: self._chunks[p.chunk_id]
            for h in hits
            for p in h.provenance
            if p.chunk_id in self._chunks
        }
        return Hits(query=query, seeds=seed_list, hits=hits, chunks=chunks)

    def _touching(self, entity: str, direction: str) -> List[int]:
        term = self._terms.lookup(entity)
        if term is None:
            return []
        out: List[int] = []
        if direction in ("out", "both"):
            out.extend(self._by_s.get(term, ()))
        if direction in ("in", "both"):
            out.extend(self._by_o.get(term, ()))
        return out

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
