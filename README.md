# juuri

**Juuri** (pronounced "JooRi", Finnish for *root*): Knowledge Graphs for RAG. Nothing else.

```
pip install juuri
```

```python
import juuri as jr
```

## Status

Early development (v0.1 in progress). Currently available:

- `TripleStore`: typed (subject, predicate, object) triples with integer-interned strings, columnar storage and S/P/O indexes.

```python
from juuri import TripleStore

store = TripleStore()
store.add("Dhoni", "PLAYED_FOR", "CSK")
store.add("Raina", "PLAYED_FOR", "CSK")

list(store.find(obj="CSK"))                       # all facts about CSK
list(store.find("Dhoni", "PLAYED_FOR"))           # None is a wildcard
```

## Roadmap

See `juuri-base.md` for the design notes and the v0.1 to v0.3 roadmap.
