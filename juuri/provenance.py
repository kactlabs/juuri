"""Provenance records: where a fact came from and how sure we are."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Provenance:
    """One piece of evidence for a triple.

    source: document name or path, or None if not given.
    chunk_id: ID of the source chunk, e.g. "app.log#3", or None if not from a chunk.
    confidence: extraction confidence between 0 and 1, or None if not scored.
    timestamp: UTC ISO 8601 time the fact was recorded.
    valid_from / valid_to: the period this evidence says the fact held, as
        "YYYY-MM-DD" with both ends inclusive. None means open on that side.
        Both None means the evidence gives no dates.
    """

    source: Optional[str]
    chunk_id: Optional[str]
    confidence: Optional[float]
    timestamp: str
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None

    @property
    def key(self) -> Tuple:
        """Identity used to avoid storing the same evidence twice (ignores timestamp)."""
        return (self.source, self.chunk_id, self.confidence, self.valid_from, self.valid_to)
