"""Retrieval evaluation: run questions with known answers and measure what comes back.

A gold case is a question plus the facts (triples) a good retrieval should return:

  {"question": "Who captained CSK?",
   "triples": [["MS Dhoni", "CAPTAINED", "Chennai Super Kings"]],
   "as_of": "2010-05-01"}          # optional

Entity names are compared ignoring case and spacing, and merged aliases count as the same entity.
"""

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .retrieve import estimate_tokens

Triple = Tuple[str, str, str]

_RETRIEVE_ARGS = {"seeds", "hops", "expand", "direction", "min_score"}
MAX_GRID = 100


@dataclass(frozen=True)
class EvalCase:
    """A question and the triples that should be retrieved for it."""

    question: str
    triples: Tuple[Triple, ...]
    as_of: Optional[str] = None


@dataclass(frozen=True)
class CaseResult:
    """Metrics for one case.

    hit: at least one gold triple was retrieved.
    coverage: share of gold triples retrieved (recall).
    precision: retrieved gold triples divided by all retrieved triples. Retrieval
        also returns neighbouring facts on purpose, so expect this to be low.
    reciprocal_rank: 1 / position of the first gold triple in the ranked hits, 0 if none.
    missing: gold triples that are not in the graph at all (for the case's as_of),
        so no retrieval setting could return them. They still count against coverage.
    context_coverage: share of gold triples that appear in the token-budgeted context.
    context_tokens: size of that context.
    """

    question: str
    gold: Tuple[Triple, ...]
    found: Tuple[Triple, ...]
    missing: Tuple[Triple, ...]
    n_hits: int
    hit: bool
    coverage: float
    precision: float
    reciprocal_rank: float
    context_coverage: float
    context_tokens: int


@dataclass
class Evaluation:
    """Results for a set of cases under one retrieval setting."""

    params: Dict[str, Any]
    results: List[CaseResult] = field(default_factory=list)

    def _mean(self, name: str) -> float:
        return sum(getattr(r, name) for r in self.results) / len(self.results)

    @property
    def hit_rate(self) -> float:
        return sum(1 for r in self.results if r.hit) / len(self.results)

    @property
    def coverage(self) -> float:
        return self._mean("coverage")

    @property
    def precision(self) -> float:
        return self._mean("precision")

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank."""
        return self._mean("reciprocal_rank")

    @property
    def context_coverage(self) -> float:
        return self._mean("context_coverage")

    @property
    def avg_hits(self) -> float:
        return self._mean("n_hits")

    @property
    def avg_context_tokens(self) -> float:
        return self._mean("context_tokens")

    @property
    def missing(self) -> int:
        """Number of gold triples that are not in the graph."""
        return sum(len(r.missing) for r in self.results)

    def summary(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items())) or "defaults"
        lines = [
            f"Retrieval evaluation ({len(self.results)} cases; {p})",
            f"  hit rate         {self.hit_rate:.2f}",
            f"  coverage         {self.coverage:.2f}",
            f"  precision        {self.precision:.2f}",
            f"  MRR              {self.mrr:.2f}",
            f"  context coverage {self.context_coverage:.2f}",
            f"  avg hits         {self.avg_hits:.1f}",
            f"  avg context      {self.avg_context_tokens:.0f} tokens",
        ]
        if self.missing:
            lines.append(f"  not in graph     {self.missing} gold triple(s): ingestion missed them, not retrieval")
        return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def parse_cases(cases: Iterable[Any]) -> List[EvalCase]:
    """Accept EvalCase objects or dicts, and return validated EvalCases."""
    out: List[EvalCase] = []
    for i, case in enumerate(cases):
        if isinstance(case, EvalCase):
            question, triples, as_of = case.question, case.triples, case.as_of
        elif isinstance(case, Mapping):
            extra = set(case) - {"question", "triples", "as_of"}
            if extra:
                raise ValueError(f"case {i}: unknown keys {sorted(extra)}")
            question, triples, as_of = case.get("question"), case.get("triples"), case.get("as_of")
        else:
            raise ValueError(f"case {i}: must be an EvalCase or a dict")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"case {i}: 'question' must be a non-empty string")
        if not isinstance(triples, (list, tuple)) or not triples:
            raise ValueError(f"case {i}: 'triples' must be a non-empty list")
        clean = []
        for t in triples:
            if not isinstance(t, (list, tuple)) or len(t) != 3 or not all(isinstance(x, str) and x.strip() for x in t):
                raise ValueError(f"case {i}: each triple must be [subject, predicate, object] of non-empty strings")
            clean.append((t[0].strip(), t[1].strip(), t[2].strip()))
        out.append(EvalCase(question.strip(), tuple(clean), as_of))
    if not out:
        raise ValueError("no cases to evaluate")
    return out


def evaluate_retrieval(
    store,
    embedder,
    cases: Iterable[Any],
    tokens: int = 2000,
    count_tokens: Optional[Callable[[str], int]] = None,
    **retrieve_args: Any,
) -> Evaluation:
    """Run each case through store.retrieve() and score it against its gold triples.

    retrieve_args (seeds, hops, expand, direction, min_score) go to retrieve().
    tokens / count_tokens set the budget for the context check. Call
    store.embed(embedder) first. An error from retrieve() is raised, not
    scored as a miss.
    """
    bad = set(retrieve_args) - _RETRIEVE_ARGS
    if bad:
        raise ValueError(f"unknown retrieve arguments: {sorted(bad)}")
    count = count_tokens or estimate_tokens
    evaluation = Evaluation(params=dict(retrieve_args))

    def key(t: Triple) -> Tuple[str, str, str]:
        return (_norm(store.canonical(t[0])), t[1], _norm(store.canonical(t[2])))

    graph_keys: Dict[Optional[str], set] = {}
    for case in parse_cases(cases):
        gold = list(dict.fromkeys(case.triples))
        gold_keys = {key(t): t for t in gold}
        if case.as_of not in graph_keys:
            graph_keys[case.as_of] = {key(t) for t in store.find(as_of=case.as_of)}
        missing = tuple(t for t in gold if key(t) not in graph_keys[case.as_of])
        hits = store.retrieve(case.question, embedder, as_of=case.as_of, **retrieve_args)
        hit_keys = [key(h.triple) for h in hits]
        found_keys = [k for k in dict.fromkeys(hit_keys) if k in gold_keys]
        rank = next((i for i, k in enumerate(hit_keys, 1) if k in gold_keys), None)
        context = hits.to_context(tokens=tokens, passages=False, count_tokens=count)
        in_context = sum(1 for t in gold if _in_context(context, t, store))
        evaluation.results.append(
            CaseResult(
                question=case.question,
                gold=tuple(gold),
                found=tuple(gold_keys[k] for k in found_keys),
                missing=missing,
                n_hits=len(hit_keys),
                hit=bool(found_keys),
                coverage=len(found_keys) / len(gold),
                precision=len(found_keys) / len(hit_keys) if hit_keys else 0.0,
                reciprocal_rank=1.0 / rank if rank else 0.0,
                context_coverage=in_context / len(gold),
                context_tokens=count(context) if context else 0,
            )
        )
    return evaluation


def _in_context(context: str, gold: Triple, store) -> bool:
    """True if a context line states the gold triple (after alias and case normalisation)."""
    want = (_norm(store.canonical(gold[0])), _norm(store.canonical(gold[2])))
    for line in context.splitlines():
        if not line.startswith("- "):
            continue
        body = line[2:].rsplit(" [", 1)[0].split(" (valid ", 1)[0]
        subject, sep, obj = body.partition(f" {gold[1]} ")
        if sep and (_norm(subject), _norm(obj)) == want:
            return True
    return False


class _CachedEmbedder:
    """Remembers vectors so a tuning grid embeds each question once."""

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._cache: Dict[str, List[float]] = {}

    def embed(self, texts: Sequence[str]):
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            for text, vec in zip(todo, self._embedder.embed(todo)):
                self._cache[text] = vec
        return [self._cache[t] for t in texts]


def tune_retrieval(
    store,
    embedder,
    cases: Iterable[Any],
    grid: Mapping[str, Sequence[Any]],
    tokens: int = 2000,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> List[Evaluation]:
    """Evaluate every combination of the settings in `grid`, best first.

    grid: {"hops": [0, 1, 2], "seeds": [1, 3]} tries all 6 combinations.
    Best means highest coverage, then highest hit rate, then fewest context
    tokens. With few cases the differences are noisy, so check the numbers
    before trusting the top row. At most 100 combinations.
    """
    if not grid:
        raise ValueError("grid must not be empty")
    bad = set(grid) - _RETRIEVE_ARGS
    if bad:
        raise ValueError(f"unknown retrieve arguments: {sorted(bad)}")
    if any(not isinstance(v, (list, tuple)) or not v for v in grid.values()):
        raise ValueError("each grid value must be a non-empty list")
    size = 1
    for v in grid.values():
        size *= len(v)
    if size > MAX_GRID:
        raise ValueError(f"grid has {size} combinations, the limit is {MAX_GRID}")
    parsed = parse_cases(cases)
    cached = _CachedEmbedder(embedder)
    names = sorted(grid)
    runs = [
        evaluate_retrieval(store, cached, parsed, tokens, count_tokens, **dict(zip(names, combo)))
        for combo in itertools.product(*(grid[n] for n in names))
    ]
    runs.sort(key=lambda e: (-e.coverage, -e.hit_rate, e.avg_context_tokens))
    return runs
