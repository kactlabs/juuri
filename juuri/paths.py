"""Chains of triples returned by path, root-cause and blast-radius queries."""

from dataclasses import dataclass
from typing import Tuple

from .provenance import Provenance
from .retrieve import _refs


@dataclass(frozen=True)
class Chain:
    """An ordered walk through the graph.

    start / end: first and last entity of the walk.
    triples: the edges in walk order, as stored (subject, predicate, object).
    provenance: one tuple of Provenance records per triple.
    cyclic: the walk stopped because the next entity was already on the chain,
        so there is no single end point.
    truncated: the walk stopped at max_hops while more edges existed.
    """

    start: str
    end: str
    triples: Tuple[Tuple[str, str, str], ...]
    provenance: Tuple[Tuple[Provenance, ...], ...]
    cyclic: bool = False
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.triples)

    def to_context(self, title: str = "CHAIN") -> str:
        """Render the chain as numbered, cited lines."""
        lines = [f"{title}:"]
        for i, (triple, prov) in enumerate(zip(self.triples, self.provenance), 1):
            fake_hit = _HitLike(prov)
            lines.append(f"{i}. {triple[0]} {triple[1]} {triple[2]} [{', '.join(_refs(fake_hit))}]")
        if not self.triples:
            lines.append(f"{self.start} (no further links)")
        if self.cyclic:
            lines.append("(cycle: no single end point)")
        if self.truncated:
            lines.append("(stopped at max_hops: more links exist)")
        return "\n".join(lines)


@dataclass(frozen=True)
class _HitLike:
    provenance: Tuple[Provenance, ...]
