"""Provenance records: where a fact came from and how sure we are."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Provenance:
    """One piece of evidence for a triple.

    source: document name or path, or None if not given.
    chunk_id: ID of the source chunk, e.g. "app.log#3", or None if not from a chunk.
    confidence: extraction confidence between 0 and 1, or None if not scored.
    timestamp: UTC ISO 8601 time the fact was recorded.
    """

    source: Optional[str]
    chunk_id: Optional[str]
    confidence: Optional[float]
    timestamp: str
