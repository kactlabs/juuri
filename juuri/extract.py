"""Chunking and LLM triple extraction."""

import json
from importlib import resources
from typing import Any, Dict, Iterator, List

from .schema import Schema


class ExtractionError(ValueError):
    """Raised when the LLM output cannot be parsed into triples."""


_REQUIRED = ("subject", "subject_type", "predicate", "object", "object_type")


def load_prompt(name: str) -> str:
    return resources.files("juuri").joinpath("prompts", f"{name}.txt").read_text(encoding="utf-8")


def chunk_text(text: str, max_chars: int = 2000) -> Iterator[str]:
    """Split text on blank lines into chunks of at most max_chars where possible."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    current = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        while len(para) > max_chars:
            if current:
                yield current
                current = ""
            yield para[:max_chars]
            para = para[max_chars:]
        if current and len(current) + len(para) + 2 > max_chars:
            yield current
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current:
        yield current


def build_prompt(text: str, schema: Schema) -> str:
    relations = "\n".join(
        f"- {name}: {dom} -> {rng}" for name, (dom, rng) in sorted(schema.relations.items())
    )
    return load_prompt("extract").format(
        classes=", ".join(sorted(schema.classes)), relations=relations, text=text
    )


def parse_triples(raw: str, schema: Schema) -> List[Dict[str, Any]]:
    """Parse LLM output. Triples that violate the schema are dropped.

    Raises ExtractionError if the output is not valid JSON of the expected shape.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"LLM output is not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        if all(k in data for k in _REQUIRED):
            data = [data]  # a single bare triple
        else:
            # Wrapped list, e.g. {"triples": [...]}.
            lists = [v for v in data.values() if isinstance(v, list)]
            if len(lists) != 1:
                raise ExtractionError("LLM output is an object without a single list of triples")
            data = lists[0]
    if not isinstance(data, list):
        raise ExtractionError("LLM output is not a list")
    triples = []
    for item in data:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip() for k in _REQUIRED):
            continue
        triple = {k: item[k].strip() for k in _REQUIRED}
        try:
            conf = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            continue
        triple["confidence"] = min(max(conf, 0.0), 1.0)
        try:
            schema.check(triple["subject_type"], triple["predicate"], triple["object_type"])
        except ValueError:
            continue
        triples.append(triple)
    return triples


def extract_triples(text: str, schema: Schema, llm, max_chars: int = 2000) -> List[Dict[str, Any]]:
    """Run LLM extraction over every chunk of text and return valid triples."""
    out: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunk_text(text, max_chars)):
        raw = llm.complete(build_prompt(chunk, schema))
        for triple in parse_triples(raw, schema):
            triple["chunk"] = chunk
            triple["chunk_index"] = index
            out.append(triple)
    return out
