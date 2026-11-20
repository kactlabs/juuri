"""Entity resolution helpers: name similarity and merge candidates."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class MergeCandidate:
    """A suggested merge of two entities.

    keep: the name to keep (the entity with more triples, then the longer name).
    merge: the name that would become an alias of `keep`.
    score: 0 to 1. 1.0 means identical after normalising case, spacing and punctuation.
    reason: why the pair was suggested.
    """

    keep: str
    merge: str
    score: float
    reason: str


def tokens(name: str) -> List[str]:
    return re.findall(r"\w+", name.casefold())


def name_score(a: str, b: str, threshold: float) -> Optional[Tuple[float, str]]:
    """Score how likely two names refer to one entity, or None if they do not look alike."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return None
    if ta == tb:
        return 1.0, "same name after normalising case, spacing and punctuation"
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) < len(long_) and set(short) <= set(long_):
        return 0.9, "one name's words are all in the other"
    ratio = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    if ratio >= threshold:
        return ratio, "similar spelling"
    return None
