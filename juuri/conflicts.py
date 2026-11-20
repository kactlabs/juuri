"""Conflict detection: facts that contradict each other under the schema's constraints.

Two facts conflict only if their validity periods can overlap, so "captained
CSK until 2015" and "captained Pune from 2016" do not conflict.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .retrieve import Fact, Hit, _refs, estimate_tokens
from .schema import Schema, SchemaError
from .temporal import DateLike, covers, parse_as_of

Period = Tuple[Optional[str], Optional[str]]

_OPEN_END = "9999-12-31"


@dataclass(frozen=True)
class Conflict:
    """Two facts that cannot both be true.

    kind: "functional", "inverse_functional" or "exclusive".
    a, b: the two facts, ordered by triple. Each has its provenance, so you
        can see which sources and chunks back each side.
    reason: a one-sentence explanation.
    overlap: the first period (from, to) in which both facts are said to hold,
        None for an open end. None when undated is True.
    undated: at least one fact has no dated evidence, so overlap cannot be
        ruled out. This is a possible conflict, not a proven one.
    """

    kind: str
    a: Fact
    b: Fact
    reason: str
    overlap: Optional[Period] = None
    undated: bool = False

    @property
    def a_sources(self) -> Tuple[str, ...]:
        return tuple(sorted({p.source for p in self.a.provenance if p.source}))

    @property
    def b_sources(self) -> Tuple[str, ...]:
        return tuple(sorted({p.source for p in self.b.provenance if p.source}))

    @property
    def cross_source(self) -> bool:
        """True if both sides name sources and share none of them."""
        a, b = set(self.a_sources), set(self.b_sources)
        return bool(a) and bool(b) and a.isdisjoint(b)

    def to_context(self) -> str:
        lines = [f"CONFLICT ({self.kind}): {self.reason}"]
        for fact in (self.a, self.b):
            refs = ", ".join(_refs(Hit(fact.triple, 0, 0.0, fact.provenance)))
            lines.append(f"- {fact.triple[0]} {fact.triple[1]} {fact.triple[2]} [{refs}]")
        if self.undated:
            lines.append("(dates unknown: they may not overlap)")
        elif self.overlap is not None:
            start, end = self.overlap
            lines.append(f"(both hold {start or 'from the start'} to {end or 'now'})")
        return "\n".join(lines)


@dataclass
class ConflictReport:
    """Conflicts found, proven ones (dated overlap) first.

    truncated: more conflicts exist than `limit`. The list holds the first
    ones found, so a proven conflict can be missing from it.
    """

    conflicts: List[Conflict] = field(default_factory=list)
    truncated: bool = False

    def __iter__(self):
        return iter(self.conflicts)

    def __len__(self) -> int:
        return len(self.conflicts)

    def to_context(self, tokens: int = 2000, count_tokens: Optional[Callable[[str], int]] = None) -> str:
        """Conflicts as text for a prompt, within a token budget. Empty if there are none."""
        if tokens < 1:
            raise ValueError("tokens must be positive")
        count = count_tokens or estimate_tokens
        used, parts = 0, []
        for conflict in self.conflicts:
            text = conflict.to_context()
            if used + count(text) <= tokens:
                used += count(text)
                parts.append(text)
        return "\n\n".join(parts)


def _overlap(pa: Sequence[Period], pb: Sequence[Period]) -> Optional[Period]:
    """The earliest overlap between any period of `pa` and any period of `pb`, or None."""
    best: Optional[Period] = None
    for from_a, to_a in pa:
        for from_b, to_b in pb:
            start = max(from_a or "", from_b or "")
            end = min(to_a or _OPEN_END, to_b or _OPEN_END)
            if start <= end:
                found = (start or None, None if end == _OPEN_END else end)
                if best is None or (found[0] or "") < (best[0] or ""):
                    best = found
    return best


def find_conflicts(
    store,
    as_of: DateLike = None,
    include_undated: bool = True,
    relations: Optional[Sequence[str]] = None,
    cross_source_only: bool = False,
    limit: int = 1000,
) -> ConflictReport:
    """Find facts that break the schema's constraints. See TripleStore.conflicts()."""
    schema: Optional[Schema] = store.schema
    if schema is None:
        raise SchemaError("conflicts require a schema")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    day = None if as_of is None else parse_as_of(as_of)
    wanted = None if relations is None else set(relations)
    if wanted is not None:
        unknown = wanted - set(schema.relations)
        if unknown:
            raise ValueError(f"undeclared relations: {sorted(unknown)}")

    facts: List[Fact] = []
    for fact in store.history():
        if not fact.periods:
            if include_undated:
                facts.append(fact)
        elif day is None or covers(fact.provenance, day):
            facts.append(fact)

    def periods(fact: Fact) -> List[Period]:
        if day is None:
            return list(fact.periods)
        return [p for p in fact.periods if (p[0] is None or p[0] <= day) and (p[1] is None or day <= p[1])]

    groups: Dict[Tuple[str, Tuple[str, str]], List[Fact]] = {}
    by_pair: Dict[Tuple[str, str], Dict[str, Fact]] = {}
    for fact in facts:
        s, p, o = fact.triple
        by_pair.setdefault((s, o), {})[p] = fact
        for constraint in schema.constraints.get(p, ()):
            if constraint == "functional":
                groups.setdefault(("functional", (s, p)), []).append(fact)
            elif constraint == "inverse_functional":
                groups.setdefault(("inverse_functional", (p, o)), []).append(fact)

    candidates: List[Tuple[str, Fact, Fact, str]] = []
    for (kind, key), members in groups.items():
        for x, y in combinations(members, 2):
            if kind == "functional":
                reason = f"{key[1]} allows one object per subject, but '{key[0]}' has '{x.triple[2]}' and '{y.triple[2]}'"
            else:
                reason = f"{key[0]} allows one subject per object, but '{key[1]}' has '{x.triple[0]}' and '{y.triple[0]}'"
            candidates.append((kind, x, y, reason))
    for fact in facts:
        s, p, o = fact.triple
        for constraint in schema.constraints.get(p, ()):
            if constraint.startswith("exclusive:"):
                other = by_pair.get((s, o), {}).get(constraint.split(":", 1)[1])
                if other is not None:
                    candidates.append(("exclusive", fact, other, f"{p} and {other.triple[1]} cannot both hold between '{s}' and '{o}'"))

    report = ConflictReport()
    seen = set()
    for kind, x, y, reason in candidates:
        a, b = sorted((x, y), key=lambda f: f.triple)
        if (kind, a.triple, b.triple) in seen:
            continue
        seen.add((kind, a.triple, b.triple))
        if wanted is not None and not ({a.triple[1], b.triple[1]} & wanted):
            continue
        pa, pb = periods(a), periods(b)
        if pa and pb:
            overlap = _overlap(pa, pb)
            if overlap is None:
                continue
            conflict = Conflict(kind, a, b, reason, overlap=overlap)
        else:
            conflict = Conflict(kind, a, b, reason, undated=True)
        if cross_source_only and not conflict.cross_source:
            continue
        if len(report.conflicts) >= limit:
            report.truncated = True
            break
        report.conflicts.append(conflict)
    report.conflicts.sort(key=lambda c: (c.undated, c.kind, c.a.triple, c.b.triple))
    return report
