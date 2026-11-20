"""Hybrid retrieval: vector search finds seed entities, then k-hop expansion."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)


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
