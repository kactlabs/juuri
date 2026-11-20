"""Compact triple store: interned strings, columnar triples, S/P/O indexes."""

from array import array
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .paths import Chain
from .provenance import Provenance
from .resolve import MergeCandidate, name_score
from .retrieve import Hit, Hits, cosine, top_seeds
from .schema import Schema, SchemaError
from .verify import Verification
from .verify import verify as verify_answer

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


def _check_walk(direction: str, max_hops: int) -> None:
    if direction not in ("out", "in", "both"):
        raise ValueError("direction must be 'out', 'in' or 'both'")
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")


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
        self._chunk_source: Dict[str, str] = {}
        self._emb: Dict[str, List[float]] = {}
        # normalised alias -> (alias as given, canonical name)
        self._aliases: Dict[str, Tuple[str, str]] = {}

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
        subject, obj = self.canonical(subject), self.canonical(obj)
        if self.schema is not None:
            st = self._resolve_type(subject, subject_type)
            ot = self._resolve_type(obj, object_type)
            self.schema.check(st, predicate, ot)
            self._types[subject] = st
            self._types[obj] = ot
        tid = self._insert(subject, predicate, obj)
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        record = Provenance(source, chunk_id, confidence, ts)
        records = self._prov.setdefault(tid, [])
        if not any(r.source == source and r.chunk_id == chunk_id and r.confidence == confidence for r in records):
            records.append(record)
        return tid

    def _steps(self, entity: str, predicates, direction: str) -> List[Tuple[int, str]]:
        """Edges leaving `entity` as (triple id, entity reached), in insertion order."""
        out: List[Tuple[int, str]] = []
        for tid in self._touching(entity, "out" if direction == "both" else direction):
            if predicates is None or self._terms.string(self._p[tid]) in predicates:
                end = self._o[tid] if direction in ("out", "both") else self._s[tid]
                out.append((tid, self._terms.string(end)))
        if direction == "both":
            for tid in self._touching(entity, "in"):
                if predicates is None or self._terms.string(self._p[tid]) in predicates:
                    out.append((tid, self._terms.string(self._s[tid])))
        return out

    def _chain(self, start: str, end: str, tids: List[int], cyclic: bool = False, truncated: bool = False) -> Chain:
        return Chain(
            start=start,
            end=end,
            triples=tuple(self.get(t) for t in tids),
            provenance=tuple(tuple(self._prov.get(t, ())) for t in tids),
            cyclic=cyclic,
            truncated=truncated,
        )

    def paths(
        self,
        source: str,
        target: str,
        max_hops: int = 4,
        predicates: Optional[Sequence[str]] = None,
        direction: str = "out",
        limit: int = 10,
    ) -> List[Chain]:
        """Find simple paths (no repeated entity) from source to target, shortest first.

        predicates: only follow these relation names (default: all).
        direction: "out" follows subject -> object, "in" the reverse, "both" either way.
        Returns at most `limit` chains. Empty if there is no path.
        """
        _check_walk(direction, max_hops)
        if limit < 1:
            raise ValueError("limit must be >= 1")
        source, target = self.canonical(source), self.canonical(target)
        if source == target:
            raise ValueError("source and target must differ")
        allowed = None if predicates is None else set(predicates)
        results: List[Chain] = []
        queue = deque([(source, [], {source})])
        while queue:
            node, tids, visited = queue.popleft()
            if len(tids) >= max_hops:
                continue
            for tid, other in self._steps(node, allowed, direction):
                if other in visited:
                    continue
                if other == target:
                    results.append(self._chain(source, target, tids + [tid]))
                    if len(results) >= limit:
                        return results
                else:
                    queue.append((other, tids + [tid], visited | {other}))
        return results

    def chains(self, entity: str, relation: str, direction: str = "out", max_hops: int = 10) -> List[Chain]:
        """Follow `relation` from `entity` until each branch ends.

        Returns one Chain per branch. A branch ends when no edge continues
        (the end entity), when it would revisit an entity (cyclic=True) or
        when max_hops is reached (truncated=True). A known entity with no
        matching edge gives one empty chain. An unknown entity gives [].
        """
        _check_walk(direction, max_hops)
        entity = self.canonical(entity)
        term = self._terms.lookup(entity)
        if term is None or (term not in self._by_s and term not in self._by_o):
            return []
        results: List[Chain] = []
        stack = [(entity, [], {entity})]
        while stack:
            node, tids, visited = stack.pop()
            nxt = self._steps(node, {relation}, direction)
            if not nxt:
                results.append(self._chain(entity, node, tids))
            elif len(tids) >= max_hops:
                results.append(self._chain(entity, node, tids, truncated=True))
            else:
                fresh = []
                for tid, other in nxt:
                    if other in visited:
                        results.append(self._chain(entity, other, tids + [tid], cyclic=True))
                    else:
                        fresh.append((other, tids + [tid], visited | {other}))
                stack.extend(reversed(fresh))
        return results

    def root_cause(self, entity: str, relation: str = "CAUSED_BY", max_hops: int = 10) -> List[Chain]:
        """Follow `relation` (subject -> object) from `entity` to the end of each branch.

        With CAUSED_BY, "A CAUSED_BY B" means B is a cause of A, so each
        chain ends at a root cause.
        """
        return self.chains(entity, relation, "out", max_hops)

    def blast_radius(self, entity: str, relation: str = "CAUSED_BY", max_hops: int = 10) -> List[Chain]:
        """Follow `relation` backwards from `entity`: what it led to, branch by branch."""
        return self.chains(entity, relation, "in", max_hops)

    def verify(self, answer: str, hits: Hits, llm) -> Verification:
        """Check an LLM answer against retrieved hits. Requires a schema.

        The LLM lists the answer's claims. Each claim is supported if a hit
        backs it (directly or by a symmetric, inverse or implies rule),
        unsupported if it uses a declared relation but nothing backs it,
        and unchecked if it uses an undeclared relation.
        """
        if self.schema is None:
            raise SchemaError("verify requires a schema")
        return verify_answer(answer, hits, llm, self.schema, canon=self.canonical)

    def _insert(self, subject: str, predicate: str, obj: str) -> int:
        """Add the triple's columns and index entries, or return the existing ID."""
        s = self._terms.intern(subject)
        p = self._terms.intern(predicate)
        o = self._terms.intern(obj)
        if (s, p, o) in self._seen:
            return self._find_id(s, p, o)
        tid = len(self._s)
        self._seen.add((s, p, o))
        self._s.append(s)
        self._p.append(p)
        self._o.append(o)
        for index, term in ((self._by_s, s), (self._by_p, p), (self._by_o, o)):
            index.setdefault(term, array("I")).append(tid)
        return tid

    def canonical(self, name: str) -> str:
        """Return the entity name that `name` was merged into, or `name` itself.

        Alias lookup ignores case and extra spacing.
        """
        record = self._aliases.get(" ".join(name.split()).casefold())
        return record[1] if record else name

    def aliases(self, entity: str) -> List[str]:
        """Names that were merged into `entity`."""
        entity = self.canonical(entity)
        return sorted(a for a, target in self._aliases.values() if target == entity)

    def merge(self, alias: str, canonical: str) -> int:
        """Merge entity `alias` into `canonical` (a sameAs link). Returns triples that collapsed.

        Triples about `alias` are rewritten to use `canonical`. Duplicates
        merge and keep every provenance record. Later adds, lookups and
        traversals using the alias name resolve to `canonical`. Raises
        SchemaError if the two entities have different types.
        """
        for label, value in (("alias", alias), ("canonical", canonical)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        old = self.canonical(alias)
        new = self.canonical(canonical)
        if old == new:
            return 0
        t_old, t_new = self._types.get(old), self._types.get(new)
        if t_old and t_new and t_old != t_new:
            raise SchemaError(f"cannot merge '{old}' ({t_old}) into '{new}' ({t_new})")
        for key, (orig, target) in list(self._aliases.items()):
            if target == old:
                self._aliases[key] = (orig, new)
        for name in {alias, old}:
            self._aliases[" ".join(name.split()).casefold()] = (name, new)
        if t_old and not t_new:
            self._types[new] = t_old
        self._types.pop(old, None)
        self._emb.pop(old, None)
        before = len(self)
        self._rebuild({old: new})
        return before - len(self)

    def _rebuild(self, mapping: Dict[str, str], drop_source: Optional[str] = None) -> None:
        """Re-insert every triple with entity names replaced per `mapping`.

        If drop_source is set, that source's provenance is removed and triples
        left with no provenance are dropped.
        """
        old = []
        for t in range(len(self._s)):
            records = self._prov.get(t, [])
            if drop_source is not None:
                remaining = [r for r in records if r.source != drop_source]
                if records and not remaining:
                    continue
                records = remaining
            old.append((self.get(t), records))
        self._terms = Interner()
        self._s, self._p, self._o = array("I"), array("I"), array("I")
        self._seen = set()
        self._by_s, self._by_p, self._by_o = {}, {}, {}
        self._prov = {}
        for (s, p, o), records in old:
            tid = self._insert(mapping.get(s, s), p, mapping.get(o, o))
            kept = self._prov.setdefault(tid, [])
            for r in records:
                if not any(k.source == r.source and k.chunk_id == r.chunk_id and k.confidence == r.confidence for k in kept):
                    kept.append(r)

    def suggest_merges(self, threshold: float = 0.85, min_embedding: Optional[float] = None) -> List[MergeCandidate]:
        """Suggest entity pairs that may be the same thing. Nothing is merged.

        Only entities of the same type are compared. Pairs are suggested if
        their names match after normalising, one name's words are all in the
        other (e.g. "Dhoni" and "MS Dhoni"), or their spelling similarity is
        at least `threshold`. If `min_embedding` is set, pairs whose stored
        vectors have cosine similarity at or above it are suggested too.
        Embedding-only matches are unreliable for short names, so review them.
        Compares every pair within a type, so it is slow on very large graphs.
        Best matches come first.
        """
        by_type: Dict[Optional[str], List[str]] = {}
        for name in self.entities():
            by_type.setdefault(self._types.get(name), []).append(name)
        found: List[MergeCandidate] = []
        for names in by_type.values():
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    result = name_score(a, b, threshold)
                    if min_embedding is not None and a in self._emb and b in self._emb:
                        sim = cosine(self._emb[a], self._emb[b])
                        if sim >= min_embedding and (result is None or sim > result[0]):
                            result = (sim, "similar embedding")
                    if result is None:
                        continue
                    keep, other = self._pick_keep(a, b)
                    found.append(MergeCandidate(keep, other, result[0], result[1]))
        found.sort(key=lambda c: (-c.score, c.keep, c.merge))
        return found

    def _pick_keep(self, a: str, b: str) -> Tuple[str, str]:
        def rank(name: str):
            return (-len(self._touching(name, "both")), -len(name), name)

        keep, other = sorted((a, b), key=rank)
        return keep, other

    def resolve_entities(self, min_score: float = 1.0, **suggest_args) -> List[MergeCandidate]:
        """Merge every suggested pair scoring at least `min_score`. Returns the merges made.

        The default of 1.0 only merges names that are identical after
        normalising case, spacing and punctuation. Lower it with care:
        "Dhoni" and "MS Dhoni" score 0.9, but short names can match several
        different entities. Use suggest_merges() to review first.
        Pairs that would merge entities of different types are skipped.
        """
        done: List[MergeCandidate] = []
        for cand in self.suggest_merges(**suggest_args):
            if cand.score < min_score:
                break
            try:
                if self.canonical(cand.merge) != self.canonical(cand.keep):
                    self.merge(cand.merge, cand.keep)
                    done.append(cand)
            except SchemaError:
                continue
        return done

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

    def ingest(
        self,
        docs,
        llm,
        max_chars: int = 2000,
        min_confidence: float = 0.0,
        source: Optional[str] = None,
        replace: bool = True,
    ) -> int:
        """Extract triples from text with an LLM and store them. Returns how many distinct triples were stored.

        docs is a string, a list of strings, or a dict {source_name: text}.
        For a string or list, `source` names the document (None if not given).
        Each fact records its source, chunk ID ("<source>#<n>"), confidence
        and timestamp. Requires a schema. Triples that fail the schema or
        fall below min_confidence are skipped.

        With replace=True (default), re-ingesting a named source first removes
        that source's earlier facts, so edited documents do not leave stale
        facts behind. Facts from other sources, and facts shared with them,
        stay. All documents are extracted before anything is replaced, so a
        failed LLM call changes nothing.
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
        extracted = [
            (name, [t for t in extract_triples(doc, self.schema, llm, max_chars) if t["confidence"] >= min_confidence])
            for name, doc in docs
        ]
        stored = set()
        for name, triples in extracted:
            if replace and name:
                self.remove_source(name)
            for t in triples:
                chunk_id = f"{name}#{t['chunk_index']}" if name else f"#{t['chunk_index']}"
                try:
                    tid = self.add(
                        t["subject"], t["predicate"], t["object"],
                        t["subject_type"], t["object_type"],
                        source=name, chunk_id=chunk_id, confidence=t["confidence"],
                    )
                except SchemaError:
                    continue  # e.g. entity already typed differently
                stored.add(tid)
                self._chunks.setdefault(chunk_id, t["chunk"])
                if name:
                    self._chunk_source[chunk_id] = name
        return len(stored)

    def sources(self) -> List[str]:
        """Names of all sources that have facts in the store."""
        return sorted({r.source for records in self._prov.values() for r in records if r.source})

    def remove_source(self, source: str) -> int:
        """Remove everything that came from `source`. Returns the number of triples deleted.

        A triple is deleted only if no other source also backs it. If other
        sources do, the triple stays and only this source's provenance is
        dropped. Entities left without any triple lose their type and
        vector. Aliases from merge() are kept. Facts added without a source
        are never removed.
        """
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        before = len(self)
        self._rebuild({}, drop_source=source)
        for chunk_id in [c for c, s in self._chunk_source.items() if s == source]:
            self._chunks.pop(chunk_id, None)
            del self._chunk_source[chunk_id]
        present = set(self.entities())
        for name in [n for n in self._types if n not in present]:
            del self._types[name]
        for name in [n for n in self._emb if n not in present]:
            del self._emb[name]
        return before - len(self)

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
        subject = None if subject is None else self.canonical(subject)
        obj = None if obj is None else self.canonical(obj)
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
