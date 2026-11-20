"""Hybrid retrieval: vector search finds seed entities, then k-hop expansion."""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .provenance import Provenance


@dataclass(frozen=True)
class Hit:
    """One retrieved triple.

    hop: 0 if the triple touches a seed entity, 1 if it is one step further, and so on.
    score: similarity of the seed entity that led to this triple.
    """

    triple: Tuple[str, str, str]
    hop: int
    score: float
    provenance: Tuple[Provenance, ...]


@dataclass
class Hits:
    query: str
    seeds: List[Tuple[str, float]]
    hits: List[Hit] = field(default_factory=list)
    chunks: Dict[str, str] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def to_context(
        self,
        tokens: int = 2000,
        passages: bool = True,
        count_tokens: Optional[Callable[[str], int]] = None,
    ) -> str:
        """Build a cited prompt section that fits within `tokens`.

        Facts are added in rank order, each followed by the chunk IDs (or
        source names) that support it. Facts that do not fit are dropped.
        If passages is True, the text of cited chunks is added after the
        facts while budget remains.
        count_tokens defaults to a rough estimate of 1 token per 4 characters.
        Use a real tokenizer's counter for exact budgets.
        """
        if tokens < 1:
            raise ValueError("tokens must be positive")
        count = count_tokens or estimate_tokens
        used = 0
        fact_lines: List[str] = []
        cited: List[str] = []
        header = "FACTS:"
        used += count(header)
        for hit in self.hits:
            refs = _refs(hit)
            line = f"- {hit.triple[0]} {hit.triple[1]} {hit.triple[2]} [{', '.join(refs)}]"
            cost = count(line)
            if used + cost > tokens:
                continue
            used += cost
            fact_lines.append(line)
            for ref in refs:
                if ref not in cited:
                    cited.append(ref)
        if not fact_lines:
            return ""
        parts = [header, *fact_lines]
        if passages:
            extra: List[str] = []
            sources_header = "SOURCES:"
            cost = count(sources_header)
            for ref in cited:
                text = self.chunks.get(ref)
                if text is None:
                    continue
                line = f"[{ref}] {text}"
                line_cost = count(line)
                if used + cost + line_cost > tokens:
                    continue
                cost += line_cost
                extra.append(line)
            if extra:
                parts += ["", sources_header, *extra]
        return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: about 1 token per 4 characters."""
    return max(1, (len(text) + 3) // 4)


def _refs(hit: "Hit") -> List[str]:
    refs: List[str] = []
    for p in hit.provenance:
        ref = p.chunk_id or p.source
        if ref and ref not in refs:
            refs.append(ref)
    return refs or ["no source"]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector sizes differ: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def top_seeds(query_vec: Sequence[float], vectors: Dict[str, Sequence[float]], k: int) -> List[Tuple[str, float]]:
    scored = [(name, cosine(query_vec, vec)) for name, vec in vectors.items()]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]
