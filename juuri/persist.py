"""Saving and loading a TripleStore as one JSON file (gzipped if the name ends in .gz).

The file holds everything the store needs to carry on where it left off:
the schema, triples, provenance and validity dates, entity types, merged
aliases, chunk text, entity embeddings and cached community summaries.
Indexes are rebuilt on load. JSON is used on purpose: loading a file can
never run code, and the file can be read and diffed by hand.
"""

import base64
import binascii
import gzip
import json
import os
import stat
import sys
import tempfile
import zlib
from array import array
from pathlib import Path
from typing import Any, Dict, List, Union

from .provenance import Provenance
from .schema import Schema, SchemaError
from .temporal import parse_date

FORMAT = "juuri-store"
VERSION = 1

_REQUIRED = (
    "schema", "terms", "triples", "provenance", "types", "aliases",
    "chunks", "chunk_sources", "embeddings", "summaries", "summary_embeddings",
)

PathLike = Union[str, "os.PathLike[str]"]


class PersistenceError(ValueError):
    """Raised when a file is not a valid juuri store file, or comes from a newer juuri."""


# ---------------------------------------------------------------- writing

def _pack(vector) -> str:
    """A vector as base64 of little-endian float64, so it round-trips exactly."""
    values = array("d", vector)
    if sys.byteorder == "big":
        values.byteswap()
    return base64.b64encode(values.tobytes()).decode("ascii")


def dump(store, include_embeddings: bool = True) -> Dict[str, Any]:
    """The store as plain data. Nothing is written."""
    from . import __version__

    n = len(store)
    return {
        "format": FORMAT,
        "version": VERSION,
        "juuri_version": __version__,
        "schema": None if store.schema is None else store.schema.to_dict(),
        "terms": [store._terms.string(i) for i in range(len(store._terms))],
        "triples": [[store._s[i], store._p[i], store._o[i]] for i in range(n)],
        "provenance": [
            [[r.source, r.chunk_id, r.confidence, r.timestamp, r.valid_from, r.valid_to] for r in store._prov.get(i, ())]
            for i in range(n)
        ],
        "types": dict(store._types),
        "aliases": [list(pair) for pair in sorted(set(store._aliases.values()))],
        "chunks": dict(store._chunks),
        "chunk_sources": dict(store._chunk_source),
        "embeddings": {k: _pack(v) for k, v in store._emb.items()} if include_embeddings else {},
        "summaries": dict(store._summaries),
        "summary_embeddings": {k: _pack(v) for k, v in store._summary_emb.items()} if include_embeddings else {},
    }


def write(path: PathLike, data: Dict[str, Any]) -> None:
    """Write `data` to `path` atomically.

    The bytes are fully prepared first, then written to a temporary file in
    the same folder and moved into place, so a failure or crash leaves any
    existing file untouched. A new file is readable by its owner only (0600).
    An existing file keeps its permissions.
    """
    path = Path(path)
    raw = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    if path.suffix == ".gz":
        raw = gzip.compress(raw, compresslevel=6, mtime=0)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------- reading

def read(path: PathLike) -> Dict[str, Any]:
    """Read and parse a store file. Raises PersistenceError if it is not JSON, FileNotFoundError if missing."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":  # gzip magic, so the file name does not matter
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError, zlib.error) as exc:
            raise PersistenceError(f"{path} is a damaged gzip file: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersistenceError(f"{path} is not a juuri store file: {exc}") from exc
    return data


def _fail(message: str) -> "PersistenceError":
    return PersistenceError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _str_map(value: object, label: str) -> Dict[str, str]:
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise _fail(f"'{label}' must map strings to strings")
    return value


def _unpack(text: object, label: str) -> List[float]:
    if not isinstance(text, str):
        raise _fail(f"{label} must be a base64 string")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _fail(f"{label} is not valid base64: {exc}") from exc
    if len(raw) % 8:
        raise _fail(f"{label} is not a whole number of float64 values")
    values = array("d")
    values.frombytes(raw)
    if sys.byteorder == "big":
        values.byteswap()
    return values.tolist()


def _vectors(value: object, label: str) -> Dict[str, List[float]]:
    if not isinstance(value, dict):
        raise _fail(f"'{label}' must be an object")
    return {k: _unpack(v, f"{label}[{k!r}]") for k, v in value.items()}


def _record(rec: object, where: str) -> Provenance:
    if not isinstance(rec, (list, tuple)) or len(rec) != 6:
        raise _fail(f"{where} must be [source, chunk_id, confidence, timestamp, valid_from, valid_to]")
    source, chunk, conf, stamp, valid_from, valid_to = rec
    for label, value in (("source", source), ("chunk_id", chunk)):
        if value is not None and not isinstance(value, str):
            raise _fail(f"{where}: {label} must be a string or null")
    if conf is not None and (isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1):
        raise _fail(f"{where}: confidence must be a number from 0 to 1 or null")
    if not isinstance(stamp, str):
        raise _fail(f"{where}: timestamp must be a string")
    for label, value in (("valid_from", valid_from), ("valid_to", valid_to)):
        if value is None:
            continue
        try:
            ok = isinstance(value, str) and parse_date(value) == value
        except ValueError:
            ok = False
        if not ok:
            raise _fail(f"{where}: {label} must be a YYYY-MM-DD date or null, got {value!r}")
    if valid_from and valid_to and valid_from > valid_to:
        raise _fail(f"{where}: valid_from {valid_from} is after valid_to {valid_to}")
    return Provenance(source, chunk, conf, stamp, valid_from, valid_to)


def restore(cls, data: Any):
    """Build a store of class `cls` from parsed file data. Checks everything before trusting it."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise _fail("not a juuri store file (missing format marker)")
    version = data.get("version")
    if not _is_int(version) or version < 1:
        raise _fail(f"invalid format version {version!r}")
    if version > VERSION:
        raise _fail(f"file format version {version} is newer than this juuri supports ({VERSION}): upgrade juuri to read it")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise _fail(f"file is missing sections: {missing}")

    schema = None
    if data["schema"] is not None:
        try:
            schema = Schema.from_dict(data["schema"])
        except SchemaError as exc:
            raise _fail(f"invalid schema: {exc}") from exc

    terms = data["terms"]
    if not isinstance(terms, list) or not all(isinstance(t, str) and t for t in terms):
        raise _fail("'terms' must be a list of non-empty strings")
    triples, provenance = data["triples"], data["provenance"]
    if not isinstance(triples, list) or not isinstance(provenance, list) or len(triples) != len(provenance):
        raise _fail("'triples' and 'provenance' must be lists of the same length")

    types = _str_map(data["types"], "types")
    if schema is None and types:
        raise _fail("file has entity types but no schema")
    if schema is not None:
        unknown = sorted({t for t in types.values() if t not in schema.classes})
        if unknown:
            raise _fail(f"entity types not in the schema: {unknown}")
    aliases = data["aliases"]
    if not isinstance(aliases, list) or not all(
        isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, str) and x.strip() for x in p) for p in aliases
    ):
        raise _fail("'aliases' must be a list of [alias, canonical] pairs")

    records: List[List[Provenance]] = []
    for i, recs in enumerate(provenance):
        if not isinstance(recs, list):
            raise _fail(f"provenance[{i}] must be a list")
        records.append([_record(r, f"provenance[{i}][{j}]") for j, r in enumerate(recs)])
    chunks = _str_map(data["chunks"], "chunks")
    chunk_sources = _str_map(data["chunk_sources"], "chunk_sources")
    summaries = _str_map(data["summaries"], "summaries")
    embeddings = _vectors(data["embeddings"], "embeddings")
    summary_embeddings = _vectors(data["summary_embeddings"], "summary_embeddings")

    store = cls(schema=schema)
    for i, triple in enumerate(triples):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3 or not all(_is_int(x) and 0 <= x < len(terms) for x in triple):
            raise _fail(f"triples[{i}] must be three term numbers from 0 to {len(terms) - 1}")
        tid = store._insert(terms[triple[0]], terms[triple[1]], terms[triple[2]])
        if tid != i:
            raise _fail(f"triples[{i}] repeats triples[{tid}]")
        store._prov[tid] = records[i]
    entities = set(store.entities())
    for entity in sorted(types):
        if entity not in entities:
            raise _fail(f"'types' names '{entity}', which is in no fact")
    for entity in sorted(embeddings):
        if entity not in entities:
            raise _fail(f"'embeddings' has a vector for '{entity}', which is in no fact")
    if schema is not None:
        for entity in sorted(entities):
            if entity not in types:
                raise _fail(f"entity '{entity}' has no type, but the file has a schema")
        for i in range(len(store)):
            s_, p_, o_ = store.get(i)
            try:
                schema.check(types[s_], p_, types[o_])
            except SchemaError as exc:
                raise _fail(f"fact {(s_, p_, o_)} breaks the schema: {exc}") from exc
    seen_alias: Dict[str, str] = {}
    for alias, canonical in aliases:
        key = " ".join(alias.split()).casefold()
        if key in seen_alias and seen_alias[key] != canonical:
            raise _fail(f"alias '{alias}' maps to both '{seen_alias[key]}' and '{canonical}'")
        seen_alias[key] = canonical
        if canonical not in entities:
            raise _fail(f"alias '{alias}' points at '{canonical}', which is in no fact")
        if alias in entities:
            raise _fail(f"alias '{alias}' is also an entity with its own facts")
    for alias, canonical in aliases:
        if " ".join(canonical.split()).casefold() in seen_alias:
            raise _fail(f"alias '{alias}' points at '{canonical}', which is itself an alias")
    for chunk_id in chunk_sources:
        if chunk_id not in chunks:
            raise _fail(f"'chunk_sources' names chunk '{chunk_id}', which has no text")
    store._types = dict(types)
    # Same key as TripleStore.canonical(): the name with spacing and case ignored.
    store._aliases = {" ".join(a.split()).casefold(): (a, c) for a, c in aliases}
    store._chunks = dict(chunks)
    store._chunk_source = dict(chunk_sources)
    store._emb = embeddings
    store._summaries = dict(summaries)
    store._summary_emb = summary_embeddings
    return store
