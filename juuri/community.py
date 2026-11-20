"""Community detection (Louvain) and LLM summaries of each community."""

import json
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

from .extract import ExtractionError, load_prompt

Triple = Tuple[str, str, str]


@dataclass(frozen=True)
class Community:
    """A group of closely linked entities.

    id: "c0", "c1", ... largest first. IDs change when the graph changes.
    members: the entities in the group.
    triples: facts whose subject and object are both members.
    """

    id: str
    members: FrozenSet[str]
    triples: Tuple[Triple, ...]


@dataclass(frozen=True)
class Summary:
    """An LLM summary of one community.

    facts_used / facts_total: how many of the community's facts fit in the
    prompt. If used < total, the summary does not cover every fact.
    signature: identifies the exact facts summarised, used for caching.
    """

    community: Community
    text: str
    facts_used: int
    facts_total: int
    signature: str


def louvain(adj: Dict[str, Dict[str, float]], resolution: float = 1.0) -> List[Set[str]]:
    """Group nodes of an undirected weighted graph by modularity (Louvain).

    adj: {node: {neighbour: weight}}, symmetric, no self loops.
    resolution: above 1 gives more, smaller groups. Below 1 gives fewer, larger ones.
    Deterministic: nodes are visited in sorted order. Returns disjoint sets
    covering every node. Groups can be disconnected, so callers should
    split them (see split_components).
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    nodes = sorted(adj)
    members: Dict[str, Set[str]] = {n: {n} for n in nodes}
    graph: Dict[str, Dict[str, float]] = {n: dict(adj[n]) for n in nodes}
    loops: Dict[str, float] = {n: 0.0 for n in nodes}  # internal weight, each edge counted once
    degree: Dict[str, float] = {n: sum(adj[n].values()) for n in nodes}
    total = sum(degree.values())  # 2m
    if total == 0:
        return [{n} for n in nodes]

    while True:
        comm = {n: n for n in graph}
        tot = dict(degree)
        moved_any = False
        moved = True
        while moved:
            moved = False
            for n in sorted(graph):
                ki = degree[n]
                old = comm[n]
                tot[old] -= ki
                to: Dict[str, float] = {}
                for nb, w in graph[n].items():
                    to[comm[nb]] = to.get(comm[nb], 0.0) + w
                best, best_gain = old, to.get(old, 0.0) - resolution * tot[old] * ki / total
                for c in sorted(to):
                    gain = to[c] - resolution * tot[c] * ki / total
                    if gain > best_gain + 1e-12:
                        best, best_gain = c, gain
                tot[best] += ki
                comm[n] = best
                if best != old:
                    moved = moved_any = True
        if not moved_any:
            break
        # Collapse each group into one node and repeat.
        new_members: Dict[str, Set[str]] = {}
        new_graph: Dict[str, Dict[str, float]] = {}
        new_loops: Dict[str, float] = {}
        new_degree: Dict[str, float] = {}
        for n, c in comm.items():
            new_members.setdefault(c, set()).update(members[n])
            new_graph.setdefault(c, {})
            new_loops[c] = new_loops.get(c, 0.0) + loops[n]
            new_degree[c] = new_degree.get(c, 0.0) + degree[n]
        for n, nbrs in graph.items():
            for nb, w in nbrs.items():
                a, b = comm[n], comm[nb]
                if a == b:
                    new_loops[a] += w / 2  # each internal edge is seen from both ends
                else:
                    new_graph[a][b] = new_graph[a].get(b, 0.0) + w
        members, graph, loops, degree = new_members, new_graph, new_loops, new_degree
    return [set(m) for m in members.values()]


def split_components(groups: Sequence[Set[str]], adj: Dict[str, Dict[str, float]]) -> List[Set[str]]:
    """Split any group that is not connected inside itself into its connected parts."""
    out: List[Set[str]] = []
    for group in groups:
        left = set(group)
        while left:
            start = min(left)
            part, stack = {start}, [start]
            while stack:
                for nb in adj.get(stack.pop(), {}):
                    if nb in group and nb not in part:
                        part.add(nb)
                        stack.append(nb)
            out.append(part)
            left -= part
    return out


def build_prompt(facts: Sequence[str]) -> str:
    return load_prompt("community").format(facts="\n".join(facts))


def parse_summary(raw: str) -> str:
    """Return the summary text from the LLM's JSON. Raises ExtractionError if it is missing."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"LLM output is not valid JSON: {exc}") from exc
    text = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("LLM output has no 'summary' text")
    return text.strip()
