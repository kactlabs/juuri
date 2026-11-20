# juuri

**Juuri** (pronounced "JooRi", Finnish for *root*): Knowledge Graphs for RAG. Nothing else.

```
pip install juuri
```

## Status

v0.2: the knowledge-graph RAG loop works end to end, with root-cause tracing, grounding checks and entity resolution. The API is early and may change.

| Step | What you get |
|---|---|
| Store | Typed (subject, predicate, object) triples, integer-interned strings, S/P/O indexes |
| Schema | Allowed classes, relations with domain/range, subclasses, declared rules |
| Extract | LLM triple extraction through Ollama |
| Provenance | Source, chunk ID, confidence and timestamp on every fact |
| Re-ingest | Re-ingesting a named source replaces only that source's facts |
| Resolve | Merge names for the same entity (`sameAs`), with suggestions |
| Retrieve | Vector search for seed entities, then k-hop expansion |
| Traverse | Paths between entities, root-cause and blast-radius chains |
| Context | Token-budgeted prompt text with citations |
| Verify | Grounding check of an LLM answer against retrieved facts |

## Quick start

Needs a running [Ollama](https://ollama.com) server with a chat model and an embedding model, for example:

```
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

```python
import juuri as jr

schema = jr.Schema(
    classes=["Player", "Team"],
    relations={
        "PLAYED_FOR": ("Player", "Team"),
        "CAPTAINED":  ("Player", "Team"),
    },
)

store = jr.TripleStore(schema=schema)

# text -> triples (each fact keeps its source, chunk, confidence, timestamp)
llm = jr.OllamaLLM("qwen2.5:7b-instruct")
store.ingest({"cricket.txt": "MS Dhoni captained Chennai Super Kings."}, llm=llm)

# embed entities, then retrieve
embedder = jr.OllamaEmbedder("nomic-embed-text")
store.embed(embedder)
hits = store.retrieve("Who led CSK?", embedder, seeds=1, hops=1)

# cited prompt text for your LLM
print(hits.to_context(tokens=2000))
```

```
FACTS:
- MS Dhoni CAPTAINED Chennai Super Kings [cricket.txt#0]

SOURCES:
[cricket.txt#0] MS Dhoni captained Chennai Super Kings.
```

## Using the pieces

**Triples and lookups**

```python
store = jr.TripleStore()
store.add("Dhoni", "PLAYED_FOR", "CSK", source="stats.csv", confidence=0.9)
list(store.find(obj="CSK"))                 # None is a wildcard
store.provenance("Dhoni", "PLAYED_FOR", "CSK")
```

Adding the same fact again from another source keeps one triple and adds a second provenance record.

**Schema guard**

With a schema, `add()` rejects undeclared relations and wrong subject or object types by raising `jr.SchemaError`. Pass types the first time an entity appears: `store.add("Dhoni", "PLAYED_FOR", "CSK", "Player", "Team")`. The type is remembered afterwards. Subclasses count as their parent class (`subclasses={"Batsman": "Player"}`).

Rules (`transitive`, `symmetric`, `implies:REL`, `inverse:REL`) are validated and stored, but nothing applies them yet.

**Ingest**

`store.ingest(docs, llm, max_chars=2000, min_confidence=0.0, source=None)` accepts a string, a list of strings, or a dict `{source_name: text}`. Triples that break the schema are dropped. Output that is not valid JSON raises `jr.ExtractionError`. A failed Ollama call raises `jr.LLMError`. Any object with `complete(prompt) -> str` works as an LLM backend.

**Retrieve**

`store.retrieve(query, embedder, seeds=3, hops=1, expand=None, direction="both", min_score=0.0)`

- `expand=["CAUSED_BY"]` follows only those relations.
- `direction` is `"out"`, `"in"` or `"both"`.
- Call `store.embed(embedder)` first, and again after adding facts about new entities.
- Any object with `embed(list_of_texts) -> list_of_vectors` works as an embedder.

**Context**

`hits.to_context(tokens=2000, passages=True, count_tokens=None)` adds facts in rank order until the budget is used, then source passages with what is left. The default token count is a rough estimate of 1 token per 4 characters. Pass `count_tokens` with a real tokenizer for exact budgets.

**Re-ingest**

`ingest` with a source name replaces that source's earlier facts (`replace=True` is the default), so an edited document leaves no stale facts. A fact that another source also backs stays. All documents are extracted before anything is replaced, so a failed LLM call changes nothing. `store.sources()` lists sources and `store.remove_source(name)` deletes one. Facts added without a source are never removed, and a document ingested without a source name cannot be replaced.

**Traverse**

```python
store.root_cause("A")            # follow CAUSED_BY to the end of each branch
store.blast_radius("D")          # follow it backwards: what D led to
store.paths("A", "D", max_hops=4, predicates=["CAUSED_BY"])

for chain in store.root_cause("A"):
    print(chain.to_context("ROOT CAUSE CHAIN"))   # numbered, cited lines
```

Direction follows how triples are stored: `A CAUSED_BY B` means B is a cause of A, so `root_cause` walks subject to object. Each `Chain` has `.start`, `.end`, `.triples`, `.provenance`, `.cyclic` and `.truncated`. A cycle stops the walk with `cyclic=True` and means there is no single root cause.

**Verify**

```python
v = store.verify(answer, hits, llm)   # needs a schema
v.grounded                            # True only if every claim is supported
v.score                               # share of supported claims, or None if no claims
v.supported, v.unsupported, v.unchecked
```

The LLM lists the answer's claims and each one is matched against the retrieved facts, using the schema's `symmetric`, `inverse` and `implies` rules. `unchecked` means the claim uses a relation the schema does not declare. Invalid LLM output raises `jr.ExtractionError`. A true claim that retrieval did not return counts as unsupported.

**Entity resolution**

```python
store.suggest_merges()                 # candidates only, nothing is merged
store.merge("Dhoni", "MS Dhoni")       # alias -> canonical
store.resolve_entities()               # merges names identical after normalising
store.aliases("MS Dhoni")
```

After a merge, the alias name works in `add`, `find`, `paths`, `root_cause`, `blast_radius` and `verify`. Merging entities of different types raises `SchemaError`. `resolve_entities()` defaults to `min_score=1.0` on purpose: "Dhoni" and "MS Dhoni" score 0.9, and short names can match several different entities, so review those with `suggest_merges()` first. Pass `min_embedding` to `suggest_merges` to add suggestions from stored vectors.

## Limitations

- Everything is in memory. Nothing is saved to disk yet, including aliases.
- Vector search compares the query with every entity. There is no ANN index.
- Only entities are embedded, not edges.
- Merges and source removals rebuild the store, and `suggest_merges` compares every pair within a type, so they are slow on large graphs.
- Rules are applied one step at a time in `verify`. There is no rule chaining and `transitive` is not used.
- Ollama is the only built-in backend.

## Roadmap

See `juuri-base.md` for the design notes and the v0.1 to v0.3 roadmap.
