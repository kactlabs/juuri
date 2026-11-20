# Juuri — Base Design Notes

> **Juuri** (pronounced "JooRi") — Finnish for *root*
> **Knowledge Graphs for RAG. Nothing else.**

---

## 1. Identity

| Item | Decision |
|---|---|
| Name | **Juuri** (formerly *KenG*, then *Seitti*) |
| Pronunciation | "JooRi" (Finnish original is "YOO-ree", where Finnish *J* = English *Y*) |
| Meaning | *Root*. Also means *exactly / just* (as in *juuri niin*, "exactly so") |
| Package | `pip install juuri`. The name was free on PyPI on 2026-11-19, so reserve it soon. |
| Import | `import juuri as jr` |
| Tagline | "Knowledge Graphs for RAG. Nothing else." |
| ErrorPal line | "Juuri finds the root of the error." |

Why the name fits: Juuri traces errors to their root cause and grounds answers in their sources, and the second meaning, *exactly*, matches its precise, cited answers.

Names considered:
- **Seitti** ("cobweb"): rejected.
- **Lähde** ("source"): the runner-up.
- **Polku** ("path"), **Solmu** ("node"), **Kytkös** ("connection"), **Punos** ("weave"), **Juoni** ("plot/thread"), **Haku** ("search"), **Muisti** ("memory"), **Tangl**: all considered.
- **Tieto**, **Lanka**, **Silta**, **Vyyhti**, **Hodos**, **Nodus**, **Plexus**: taken on PyPI.

---

## 2. Positioning

- Juuri is **exclusively for Knowledge Graph RAG**. It is not a general-purpose graph library.
- **NetworkX** is too generic: it has no text, no LLMs, no provenance and no retrieval.
- **Heavy GraphRAG frameworks** are too bloated.
- Juuri is a lightweight middle ground: documents in, a trustworthy graph, and cited context out to the LLM.

**Feature filter:** does this make retrieved answers more *accurate*, more *grounded*, or *cheaper*? If not, it doesn't go in Juuri.

---

## 3. Scope

### In scope
- **Ingestion:** LLM triple extraction, a schema guard, entity resolution
- **Storage:** typed triples, provenance, embeddings, temporal facts
- **Retrieval:** entity linking, hybrid seed-and-expand, path retrieval, community summaries
- **Context:** token-budgeted, cited prompt building
- **Trust:** grounding checks, conflict detection, retrieval evaluation

### Out of scope
- General graph algorithms such as centrality, flows and isomorphism. Keep only k-hop, paths, PageRank and Leiden.
- Plotting and drawing (a simple debug view at most)
- Multigraph theory, bipartite tools and graph generators
- A full OWL reasoner or SPARQL engine
- NetworkX compatibility (a one-way `to_networkx()` export at most)

---

## 4. Features missing in NetworkX, by RAG pipeline stage

### 4.1 Ingestion (text → graph)
- **LLM triple extraction:** `web.ingest(docs, llm=...)` chunks the text, extracts (S, P, O) triples and adds them. Backends are pluggable: Ollama, llama.cpp, OpenAI.
- **Entity resolution:** merges "MS Dhoni", "Dhoni" and "Mahendra Singh Dhoni" using aliases, fuzzy matching and embeddings.
- **Schema / ontology guard:** only declared entity and relation types are allowed, so no junk predicates.
- **Chunk ↔ entity links:** every node and edge points back to its source chunks.

### 4.2 Storage
- **Typed triples** with S/P/O indexes, so lookups are fast in any direction.
- **Compact storage:** strings mapped to integer IDs, CSR/columnar adjacency, a target of 5–10× less RAM than NetworkX, and memory-mapped loading.
- **Provenance and confidence on every edge:** source document, chunk ID, extraction confidence, timestamp.
- **Vector embeddings** on nodes and edges, with a light built-in ANN index (HNSW).
- **Temporal facts:** `valid_from` / `valid_to`, plus "as of" queries.
- **Incremental updates:** re-ingesting a document replaces only that document's facts.

### 4.3 Retrieval (the core gap)
- **Query → entity linking:** finds the graph entities mentioned in the question.
- **Hybrid seed and expand:** vector search finds seed nodes, then a k-hop expansion filtered by relation type.
- **Relevance-ranked subgraphs:** personalised PageRank, seeded by query similarity, prunes the expansion.
- **Path retrieval:** multi-hop reasoning chains between the entities in a query.
- **Community summaries** (GraphRAG style): Leiden clustering plus an LLM summary per cluster, for global questions.
- **Text-to-query:** the LLM writes a structured Juuri query from natural language.
- **Query styles:** pattern matching (Cypher-like) and fluent traversal (Gremlin-like).

### 4.4 Context building (graph → prompt)
- **Token-budgeted serialisation:** packs the most relevant triples into N tokens.
- **Citations:** every fact in the prompt carries its source chunk ID.
- **Graph and chunk context together:** the facts plus the original passages they came from.

### 4.5 Evaluation and trust
- **Retrieval metrics:** hit rate and coverage against gold answers, used to tune hop depth and pruning.
- **Grounding check:** confirms LLM claims are backed by retrieved triples.
- **Conflict detection:** flags contradictory facts from different sources.

---

## 5. Ontology

An **ontology** is the knowledge graph's schema: what kinds of things exist and how they may relate. The graph holds the facts; the ontology holds the rules.

- **Classes:** `Player`, `Team`, `Match`
- **Relations:** `PLAYED_FOR`, `CAPTAINED`
- **Constraints:** `PLAYED_FOR` must go from a `Player` to a `Team`
- **Hierarchies and rules:** `Batsman` is a kind of `Player`; `CAPTAINED` implies `PLAYED_FOR`

Ontologies matter for RAG because LLM extraction is messy (`played_for`, `plays_for`, `represented`, …). A schema forces a clean graph, which gives reliable traversal and better answers.

### RDFS vs OWL
- **RDF** stores all data as triples.
- **RDFS** adds basic vocabulary: `Class`, `subClassOf`, `domain`, `range`, `label`.
- **OWL** adds logic on top of RDFS: inverse, transitive, symmetric, cardinality, disjoint, `sameAs`. It is powerful, but its reasoners (HermiT, Pellet) are slow and heavy.

### Juuri's approach: borrow the useful 20%
- From RDFS: subclasses, domain and range
- From OWL: inverse, transitive, symmetric and `sameAs` (entity resolution)
- Export to Turtle/OWL so graphs work with Protégé and SPARQL databases

```python
schema = jr.Schema(
    classes=["Player", "Team", "Match"],
    relations={
        "PLAYED_FOR": ("Player", "Team"),
        "CAPTAINED":  ("Player", "Team"),
    },
    rules={"CAPTAINED": "implies:PLAYED_FOR"},
)
```

---

## 6. Juuri vs NetworkX

| Feature | NetworkX | Juuri |
|---|---|---|
| Data model | Generic graph | Typed triples (S–P–O) |
| Ontology/schema | ❌ | ✅ Lightweight RDFS/OWL subset |
| Indexes | Adjacency dicts | S/P/O indexes |
| Memory use | High (dict-of-dicts) | Compact (int IDs, CSR) |
| Text → triples | ❌ | ✅ LLM extraction |
| Entity resolution | ❌ | ✅ Fuzzy + embeddings |
| Provenance | Manual attrs | ✅ Source, chunk, confidence |
| Temporal facts | ❌ | ✅ valid_from / valid_to |
| Node embeddings | ❌ | ✅ + ANN index |
| Hybrid retrieval | ❌ | ✅ Vector seed + k-hop expand |
| Query language | Python API only | Pattern match + fluent traversal |
| Text-to-query | ❌ | ✅ |
| Community summaries | Clustering only | ✅ Leiden + LLM summaries |
| Graph → prompt | ❌ | ✅ Token-budgeted, cited |
| Reasoning rules | ❌ | ✅ Inverse, transitive, sameAs |
| Grounding check | ❌ | ✅ |
| Incremental updates | Manual | ✅ Per-document re-ingest |
| RDF/Turtle export | ❌ | ✅ |
| Graph algorithms | ✅ 500+ | Retrieval-only core set |
| Maturity | ✅ 20 years | New |

---

## 7. Core API (five verbs)

```python
import juuri as jr

web = jr.Web(schema=my_schema)            # schema: rules for the graph
web.ingest(docs, llm=ollama)              # text → triples
hits = web.retrieve("question...")        # graph retrieval
ctx = hits.to_context(tokens=2000)        # cited prompt
web.verify(answer, hits)                  # grounding check
```

---

## 8. Reference use case: ErrorPal

### Problem
- Folders A → B → C → D form a tree. Each folder holds its own error files.
- An error in A may be caused by an error in B, and so on down to D.
- Goal: given an error, find its **root cause** across folders and give a cited answer.

```
Folder A  (A-error-files)
 └── Folder B  (B-error-files)
      └── Folder C  (C-error-files)
           └── Folder D  (D-error-files)
```

### Schema
```python
errorpal_schema = jr.Schema(
    classes=["Folder", "ErrorFile", "Error", "Fix"],
    relations={
        "CHILD_OF":   ("Folder", "Folder"),
        "CONTAINS":   ("Folder", "ErrorFile"),
        "REPORTS":    ("ErrorFile", "Error"),
        "CAUSED_BY":  ("Error", "Error"),
        "RELATED_TO": ("Error", "Error"),
        "FIXED_BY":   ("Error", "Fix"),
    },
    rules={
        "CHILD_OF":   "transitive",
        "CAUSED_BY":  "transitive",
        "RELATED_TO": "symmetric",
    },
)
```

### Ingestion
```python
web = jr.Web(schema=errorpal_schema)
web.add_tree("A/", levels=["A", "B", "C", "D"])
web.ingest("A/**/*.log", llm=ollama)
```
- **Error normalisation:** strips timestamps, IDs and paths, so the same error doesn't become many nodes.
- **Entity resolution:** one error appearing in several folders becomes a single node, which links the folders.
- **Provenance:** file, line number and timestamp on every fact.
- **`CAUSED_BY` inference:** from stack traces, timing and LLM extraction, each with a confidence score.

### Retrieval
```python
hits = web.retrieve(
    "ConnectionTimeout in A/app.log",
    expand="CAUSED_BY", direction="down", hops=3,
)
```
1. Link the pasted error to its `Error` node in A.
2. Walk down through `CAUSED_BY`: A → B → C → D.
3. Treat the deepest error with no further cause as the root cause.
4. Attach any known `FIXED_BY` nodes.

### Context sent to the LLM
```
ROOT CAUSE CHAIN:
A: ConnectionTimeout      [A/app.log:142]
 ← B: PoolExhausted        [B/db.log:88]
  ← C: SlowQuery (>30 s)    [C/sql.log:17]
   ← D: DiskFull /var       [D/sys.log:5]
KNOWN FIX: Clear /var/tmp + log rotation  [fix #12]
```

### Why plain vector RAG fails here
- It can't follow the chain across folders.
- It returns similar-looking chunks with no order and no root cause.
- It can't say where in the tree the problem started.

### Extras
- **Blast radius:** `direction="up"` shows what a D error will break.
- **Temporal filter:** errors in the last 24 hours.
- **Recurrence:** how often the same root cause comes back.
- The schema handles branching trees too (A → B and A → C).

---

## 9. Roadmap

### v0.1 (minimum KG-RAG loop)
1. Typed triples + S/P/O indexes + compact storage
2. Schema/ontology guard
3. LLM triple extraction (Ollama first)
4. Provenance on every edge
5. Node embeddings + hybrid seed-and-expand retrieval
6. Token-budgeted context with citations

### v0.2
- Entity resolution (fuzzy + embeddings + `sameAs`)
- Path retrieval + root-cause and blast-radius traversal
- Grounding check (`verify`)
- Incremental per-document re-ingest

### v0.3
- Temporal facts
- Community summaries (Leiden + LLM)
- Text-to-query
- Retrieval evaluation metrics, conflict detection
- Turtle/OWL export, `to_networkx()` export

### Test projects
- **ErrorPal:** cross-folder root-cause tracing
- **Cricket KG RAG:** entity-heavy, multi-hop, temporal questions

---

## 10. Immediate next steps
- [ ] Reserve `juuri` on PyPI (placeholder 0.0.1)
- [ ] Reserve the GitHub repo/org
- [ ] Scaffold the package: `pyproject.toml`, `juuri/` core modules
- [ ] Write the README with the name, pronunciation and tagline
- [ ] Build v0.1 against the ErrorPal folder dataset
