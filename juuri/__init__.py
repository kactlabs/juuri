from .community import Community, Summary
from .conflicts import Conflict, ConflictReport
from .evaluate import CaseResult, EvalCase, Evaluation
from .extract import ExtractionError
from .llm import LLMError, OllamaEmbedder, OllamaLLM
from .paths import Chain
from .persist import PersistenceError
from .provenance import Provenance
from .query import QueryError, QueryResult, Row
from .resolve import MergeCandidate
from .retrieve import Fact, Hit, Hits
from .schema import Schema, SchemaError
from .store import TripleStore
from .verify import Claim, Verification

__version__ = "0.4.0"

__all__ = [
    "CaseResult", "Chain", "Claim", "Community", "Conflict", "ConflictReport", "EvalCase", "Evaluation", "ExtractionError", "Fact", "Hit", "Hits", "LLMError", "MergeCandidate", "OllamaEmbedder", "OllamaLLM",
    "PersistenceError", "Provenance", "QueryError", "QueryResult", "Row", "Schema", "SchemaError", "Summary", "TripleStore", "Verification",
]
