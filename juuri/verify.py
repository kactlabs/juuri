"""Grounding check: are the claims in an LLM answer backed by retrieved triples?"""

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .extract import ExtractionError, load_prompt
from .provenance import Provenance
from .retrieve import Hits
from .schema import Schema

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
UNCHECKED = "unchecked"

_KEYS = ("subject", "predicate", "object")


@dataclass(frozen=True)
class Claim:
    """One claim from the answer and what the check found.

    status:
      supported   a retrieved triple backs the claim (directly or via a schema rule).
      unsupported the claim uses a known relation but no retrieved triple backs it.
      unchecked   the claim uses a relation the schema does not declare, so it cannot be checked.
    support: the retrieved triple that backs the claim, if any.
    via: how it matches: "exact", "symmetric", "inverse:REL" or "implies:REL".
    """

    subject: str
    predicate: str
    object: str
    status: str
    support: Optional[Tuple[str, str, str]] = None
    via: Optional[str] = None
    provenance: Tuple[Provenance, ...] = ()


@dataclass
class Verification:
    claims: List[Claim]

    @property
    def supported(self) -> List[Claim]:
        return [c for c in self.claims if c.status == SUPPORTED]

    @property
    def unsupported(self) -> List[Claim]:
        return [c for c in self.claims if c.status == UNSUPPORTED]

    @property
    def unchecked(self) -> List[Claim]:
        return [c for c in self.claims if c.status == UNCHECKED]

    @property
    def grounded(self) -> bool:
        """True if there is at least one claim and none is unsupported or unchecked."""
        return bool(self.claims) and all(c.status == SUPPORTED for c in self.claims)

    @property
    def score(self) -> Optional[float]:
        """Share of claims that are supported, or None if the answer made no claims."""
        if not self.claims:
            return None
        return len(self.supported) / len(self.claims)


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def parse_claims(raw: str) -> List[Tuple[str, str, str]]:
    """Parse the LLM's claim list. Raises ExtractionError on malformed output."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"LLM output is not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        if all(k in data for k in _KEYS):
            data = [data]
        elif isinstance(data.get("claims"), list):
            data = data["claims"]
        else:
            raise ExtractionError("LLM output has no 'claims' list")
    if not isinstance(data, list):
        raise ExtractionError("LLM output is not a list")
    claims = []
    for item in data:
        if isinstance(item, dict) and all(isinstance(item.get(k), str) and item[k].strip() for k in _KEYS):
            claims.append(tuple(item[k].strip() for k in _KEYS))
    return claims


def _match(claim, hit_triple, schema: Schema) -> Optional[str]:
    """Return how hit_triple backs claim, or None."""
    s, p, o = (_norm(claim[0]), claim[1], _norm(claim[2]))
    hs, hp, ho = _norm(hit_triple[0]), hit_triple[1], _norm(hit_triple[2])
    same = (hs, ho) == (s, o)
    swapped = (hs, ho) == (o, s)
    if hp == p and same:
        return "exact"
    if hp == p and swapped and schema.rules.get(p) == "symmetric":
        return "symmetric"
    if swapped and (schema.rules.get(p) == f"inverse:{hp}" or schema.rules.get(hp) == f"inverse:{p}"):
        return f"inverse:{hp}"
    if same and schema.rules.get(hp) == f"implies:{p}":
        return f"implies:{hp}"
    return None


def verify(answer: str, hits: Hits, llm, schema: Schema) -> Verification:
    """Extract claims from `answer` with `llm` and check each against `hits`.

    Raises ExtractionError if the LLM output cannot be parsed, so a failed
    check is never reported as "no claims".
    """
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    relations = "\n".join(
        f"- {name}: {dom} -> {rng}" for name, (dom, rng) in sorted(schema.relations.items())
    )
    prompt = load_prompt("claims").format(relations=relations, answer=answer)
    results: List[Claim] = []
    for s, p, o in parse_claims(llm.complete(prompt)):
        if p not in schema.relations:
            results.append(Claim(s, p, o, UNCHECKED))
            continue
        found = None
        for hit in hits:
            via = _match((s, p, o), hit.triple, schema)
            if via is not None:
                found = (hit, via)
                if via == "exact":
                    break
        if found:
            hit, via = found
            results.append(Claim(s, p, o, SUPPORTED, hit.triple, via, hit.provenance))
        else:
            results.append(Claim(s, p, o, UNSUPPORTED))
    return Verification(results)
