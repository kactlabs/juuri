# juuri

**Juuri** (pronounced "JooRi", Finnish for *root*): Knowledge Graphs for RAG. Nothing else.

```
pip install juuri
```

## Status

v0.1: the minimum knowledge-graph RAG loop works end to end. The API is early and may change.

| Step | What you get |
|---|---|
| Store | Typed (subject, predicate, object) triples, integer-interned strings, S/P/O indexes |
| Schema | Allowed classes, relations with domain/range, subclasses, declared rules |
| Extract | LLM triple extraction through Ollama |
| Provenance | Source, chunk ID, confidence and timestamp on every fact |
| Retrieve | Vector search for seed entities, then k-hop expansion |
| Context | Token-budgeted prompt text with citations |

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

## Limitations

- Everything is in memory. Nothing is saved to disk yet.
- Vector search compares the query with every entity. There is no ANN index.
- Only entities are embedded, not edges.
- No entity resolution: "MS Dhoni" and "Dhoni" are separate nodes.
- Ollama is the only built-in backend.

## Roadmap

See `juuri-base.md` for the design notes and the v0.1 to v0.3 roadmap.
