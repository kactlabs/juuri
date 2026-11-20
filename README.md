# juuri

**Juuri** (pronounced "JooRi", Finnish for *root*): Knowledge Graphs for RAG. Nothing else.

```
pip install juuri
```

## Status

v0.4: the knowledge-graph RAG loop works end to end, with temporal facts, root-cause tracing, grounding checks, entity resolution, community summaries, text-to-query, retrieval evaluation, conflict detection, exports and saving to disk. The API is early and may change.

| Step | What you get |
|---|---|
| Store | Typed (subject, predicate, object) triples, integer-interned strings, S/P/O indexes |
| Schema | Allowed classes, relations with domain/range, subclasses, rules, constraints |
| Extract | LLM triple extraction through Ollama, including dates when the text gives them |
| Provenance | Source, chunk ID, confidence and timestamp on every fact |
| Temporal | `valid_from` / `valid_to` on facts, "as of" queries, fact history |
| Re-ingest | Re-ingesting a named source replaces only that source's facts |
| Resolve | Merge names for the same entity (`sameAs`), with suggestions |
| Retrieve | Vector search for seed entities, then k-hop expansion |
| Traverse | Paths between entities, root-cause and blast-radius chains |
| Communities | Groups of linked entities with cited LLM summaries, for broad questions |
| Ask | Text-to-query: the LLM writes a structured query, juuri validates and runs it |
| Context | Token-budgeted prompt text with citations |
| Verify | Grounding check of an LLM answer against retrieved facts |
| Evaluate | Hit rate, coverage, precision and MRR against gold answers, plus settings tuning |
| Conflicts | Facts that contradict each other under schema constraints, with dates considered |
| Export | Turtle / OWL, and a one-way `networkx` graph |
| Persist | `save()` / `load()` the whole store to one JSON file, checked on load |

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

Rules (`transitive`, `symmetric`, `implies:REL`, `inverse:REL`) are validated and stored. `verify` uses `symmetric`, `inverse` and `implies`, and all of them are written to the Turtle export. `transitive` is stored and exported but not applied anywhere.

Constraints (`functional`, `inverse_functional`, `exclusive:REL`) are a separate argument, `constraints=`, used by `conflicts()` (see below). `add()` does not reject facts that break them.

**Ingest**

`store.ingest(docs, llm, max_chars=2000, min_confidence=0.0, source=None)` accepts a string, a list of strings, or a dict `{source_name: text}`. Triples that break the schema are dropped. If the text says when a fact was true, the LLM adds `valid_from` / `valid_to`. A triple with an unreadable date is skipped, not guessed. Output that is not valid JSON raises `jr.ExtractionError`. A failed Ollama call raises `jr.LLMError`. Any object with `complete(prompt) -> str` works as an LLM backend.

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

**Temporal facts**

```python
store.add("Dhoni", "CAPTAINED", "CSK", valid_from="2008", valid_to="2015")
store.add("Dhoni", "CAPTAINED", "RPS", valid_from="2016", valid_to="2017")

list(store.find("Dhoni", "CAPTAINED", as_of="2016-06-01"))   # RPS only
store.history("Dhoni")                                       # each fact with its periods
hits = store.retrieve("Who did Dhoni captain?", embedder, as_of="2016-06-01")
```

- Dates are `"YYYY"`, `"YYYY-MM"`, `"YYYY-MM-DD"`, an int year or a `date`. A partial `valid_from` means the start of the period and a partial `valid_to` means its end, so `valid_to="2015"` is 2015-12-31. Both ends are inclusive and `None` is open.
- The dates live on the provenance record, so each source can give its own period. The same fact from two sources with different periods stays one triple with both periods.
- `as_of` must be a full date. A fact with dated evidence is valid if one of its periods contains the day. A fact with no dated evidence is kept unless you pass `include_undated=False`.
- `as_of` works in `find`, `retrieve`, `paths`, `root_cause`, `blast_radius`, `communities`, `conflicts`, `query` and both exports.
- `to_context()` prints the periods, for example `(valid 2008-01-01 to 2015-12-31)`.
- This stores periods only. A newer fact does not close an earlier period automatically.

**Communities and broad questions**

```python
store.communities(min_size=2, resolution=1.0)            # groups of linked entities
store.summarize_communities(llm)                         # a cited summary for each
print(store.global_context("Tell me about the Bangalore team", embedder, llm, k=3, tokens=2000))
```

Communities come from a pure-Python Louvain algorithm, with any group that is not connected inside itself split afterwards. It is not Leiden, and there is one flat level, not a hierarchy. Direction and relation names are ignored, and several facts between the same pair add weight. Summaries are cached by the exact facts they cover, so an unchanged community is not sent to the LLM again. A community larger than `max_facts` (default 60) is summarised from its first facts only, and `global_context` says so.

**Ask (text-to-query)**

```python
result = store.ask("Which team did Dhoni captain in 2016?", llm)   # needs a schema
result.query                  # the validated query that ran, with graph entity names
result.rows                   # for "match": bindings plus the facts and provenance behind them
result.chains                 # for root_cause, blast_radius and paths
print(result.to_context(tokens=2000))
```

The LLM writes a small JSON query. juuri checks it against the schema, links entity names to real graph entities, and runs it. The LLM never writes code. `store.query(spec)` runs the same JSON directly. There are four operations:

```python
{"op": "match", "patterns": [{"subject": "Dhoni", "predicate": "CAPTAINED", "object": "?team"},
                             {"subject": "?player", "predicate": "PLAYED_FOR", "object": "?team"}],
 "as_of": "2010-05-01", "limit": 50}
{"op": "root_cause",   "entity": "Login timeout", "relation": "CAUSED_BY"}
{"op": "blast_radius", "entity": "Disk full",     "relation": "CAUSED_BY"}
{"op": "paths", "source": "Login timeout", "target": "Disk full", "predicates": ["CAUSED_BY"]}
```

A term starting with `?` is a variable. Unknown keys, undeclared relations, relation variables and partial dates raise `jr.QueryError`. A name that matches no entity raises `QueryError`, and so does a name that matches several, with the candidates listed. It never guesses. There is no OR, NOT or counting. Check the output of `ask` for your own model: a wrong query that is still valid runs and returns wrong rows.

**Evaluate retrieval**

```python
cases = [{"question": "Who led CSK?",
          "triples": [["MS Dhoni", "CAPTAINED", "Chennai Super Kings"]],
          "as_of": "2010-05-01"}]                          # as_of is optional
ev = store.evaluate(embedder, cases, seeds=2, hops=1)
print(ev.summary())
for run in store.tune(embedder, cases, {"hops": [0, 1, 2], "seeds": [1, 3]}):   # best first
    print(run.params, run.coverage, run.precision)
```

Metrics are hit rate, coverage (recall), precision, mean reciprocal rank, and context coverage (how many gold facts survive the `to_context` token budget). Gold facts that are not in the graph at all are counted separately, so an ingestion miss is not blamed on retrieval. Precision is low by design, because retrieval returns neighbouring facts. `tune` sorts by coverage, then hit rate, then fewest context tokens. With few cases many settings tie, so read the other columns before trusting the top row.

**Conflicts**

```python
schema = jr.Schema(
    classes=["Person", "Team", "City"],
    relations={"BORN_IN": ("Person", "City"), "CAPTAINED": ("Person", "Team"),
               "MANAGES": ("Person", "Team"), "COACHES": ("Person", "Team")},
    constraints={"BORN_IN": "functional",              # one object per subject
                 "CAPTAINED": "inverse_functional",    # one subject per object
                 "MANAGES": ["exclusive:COACHES"]},    # cannot both hold for the same pair
)
for conflict in store.conflicts():     # also: as_of, include_undated, relations, cross_source_only, limit
    print(conflict.to_context())
```

Two facts conflict only if their validity periods overlap, so captaining one team until 2015 and another from 2016 is fine. If either fact has no dated evidence, the conflict is reported with `undated=True`, meaning the facts may not overlap. Each side carries its provenance, and `conflict.cross_source` is true when the two sides share no source. juuri reports conflicts but does not decide which side is right. Entity names are compared as stored, so merge variants such as `Chennai` and `Chennai, India` first.

**Save and load**

```python
store.save("graph.json")                    # a name ending in .gz is gzipped
store = jr.TripleStore.load("graph.json")   # the schema comes from the file
store.save("small.json.gz", include_embeddings=False)   # vectors are most of a large file
```

The file holds the schema, facts, provenance and dates, entity types, merged aliases, chunk text, entity vectors and cached community summaries, so a loaded store answers queries the same as the saved one. Re-ingest, merges and `ask()` keep working after a load.

- It is JSON, not pickle, so loading a file cannot run code, and you can read or diff it. Output is deterministic.
- Saving is atomic: the file is written to a temporary name and moved into place, so a failed save leaves an existing file as it was. A new file is readable by its owner only (mode 0600).
- `load` checks the whole file before it builds a store. A damaged or hand-edited file raises `jr.PersistenceError`, as does a file written by a newer juuri. This includes facts that break the schema, an entity with the wrong type and aliases that point at nothing. Nothing is repaired or guessed. A missing file raises `FileNotFoundError`.
- Without vectors, call `embed()` again after loading.
- It saves the whole store each time. It does not append, and it is not safe for two processes writing the same file.
- A saved file is plain text of your data. Anyone who can read it can read your facts and source text.

**Export**

```python
store.to_turtle("graph.ttl", base="https://data.acme.com/kg/")   # OWL 2 vocabulary, loads in Protégé
g = store.to_networkx()          # networkx.MultiDiGraph, needs: pip install juuri[networkx]
```

The Turtle file contains the schema as an ontology (classes, subclasses, relations with domain and range, transitive, symmetric, inverse and implies rules, functional constraints), the entities, merged aliases as `owl:sameAs`, and the facts. With `include_provenance=True` (the default) each evidence record becomes an `rdf:Statement` with its source, chunk, confidence, dates and recording time, which you can query in SPARQL. The output is deterministic. There is no Turtle import. In the networkx graph each fact is an edge keyed by its predicate, with `sources`, `chunks`, `confidence`, `periods` and `provenance` attributes. Both exports are copies, and both accept `as_of`.

## Limitations

- The store lives in memory while you use it. `save()` and `load()` write and read the whole thing, so a big graph is slow to save and load, and there is no incremental or multi-process storage.
- Vector search compares the query with every entity. There is no ANN index.
- Only entities are embedded, not edges.
- Merges and source removals rebuild the store, and `suggest_merges` compares every pair within a type, so they are slow on large graphs.
- `verify` applies rules one step at a time. There is no rule chaining, and it ignores time, so retrieve with `as_of` first.
- Constraints are checked on demand by `conflicts()`, and exported to OWL, but never enforced by `add()`.
- Community detection is Louvain, not Leiden, and untested on large graphs.
- Ollama is the only built-in backend.

## Roadmap

See `juuri-base.md` for the design notes and the v0.1 to v0.3 roadmap.
