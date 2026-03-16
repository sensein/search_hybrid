# -*- coding: utf-8 -*-
"""
FastAPI application for hybrid retrieval and re-ranking based concept mapping
"""

import os
import re
import time
import logging
from typing import Any, List, Optional, Dict, Union
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

# Load .env file automatically when present (works with direct uvicorn invocation too)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers

from db_layer import OntologyDB
from retrieval import HybridRetriever, RetrievalCandidate
from reranking import create_reranker, RerankingResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================


class ConceptMappingRequest(BaseModel):
    """Single concept mapping — identical field set to /map/search"""
    text: str = Field(..., min_length=1, max_length=500, description="Concept or text to map")
    context: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional context for disambiguation (e.g. clinical setting, related symptoms)"
    )
    max_results: int = Field(5, ge=1, le=20, description="Maximum results to return")
    ontologies: Optional[str] = Field(None, description="Comma-separated ontology acronyms, e.g. SNOMEDCT,MESH,MONDO")
    openrouter_api_key: Optional[str] = Field(None, description="OpenRouter API key (overrides env)")
    openrouter_model: Optional[str] = Field(None, description="OpenRouter model name (overrides env)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "kidney disease",
                "context": "progressive decline in GFR",
                "max_results": 5,
                "ontologies": None,
                "openrouter_api_key": None,
                "openrouter_model": None,
            }
        }
    }


class BatchMappingRequest(BaseModel):
    """Batch concept mapping — list of {text, context} objects (up to 20)"""
    text: Union[str, List[Union[str, Dict]]] = Field(
        ...,
        description=(
            "Concepts to map. Preferred: list of objects with optional context. "
            "Also accepts: comma-separated string or list of plain strings."
        )
    )
    max_results: int = Field(5, ge=1, le=20, description="Maximum results per concept")
    ontologies: Optional[str] = Field(None, description="Comma-separated ontology acronyms, e.g. SNOMEDCT,MESH,MONDO")
    openrouter_api_key: Optional[str] = Field(None, description="OpenRouter API key (overrides env)")
    openrouter_model: Optional[str] = Field(None, description="OpenRouter model name (overrides env)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": [
                    {"text": "kidney disease", "context": "progressive decline in GFR"},
                    {"text": "T2DM", "context": "type 2 diabetes with insulin resistance"},
                    {"text": "astrocyte"}
                ],
                "max_results": 5,
                "ontologies": None,
                "openrouter_api_key": None,
                "openrouter_model": None,
            }
        }
    }


class ContextualSearchRequest(BaseModel):
    """Contextual search — identical field set to /map/concept"""
    text: str = Field(..., min_length=1, max_length=500, description="Primary query text")
    context: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional context for disambiguation (e.g. clinical setting, related symptoms)"
    )
    max_results: int = Field(5, ge=1, le=20, description="Maximum results to return")
    ontologies: Optional[str] = Field(None, description="Comma-separated ontology acronyms, e.g. SNOMEDCT,MESH,MONDO")
    openrouter_api_key: Optional[str] = Field(None, description="OpenRouter API key (overrides env)")
    openrouter_model: Optional[str] = Field(None, description="OpenRouter model name (overrides env)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "kidney disease",
                "context": "progressive decline in GFR",
                "max_results": 5,
                "ontologies": None,
                "openrouter_api_key": None,
                "openrouter_model": None,
            }
        }
    }


class ResultItem(BaseModel):
    """Individual result item"""
    rank: int
    ontology_id: str
    ontology_label: str
    ontology: str
    original_score: float
    llm_score: float
    late_interaction_score: float
    final_score: float
    retrieval_scores: Optional[Dict[str, float]] = None


class ConceptMappingResponse(BaseModel):
    """Response for single concept mapping"""
    query: str
    type: str = "single"
    results: List[ResultItem]
    total_results: int
    processing_time_ms: float


class BatchMappingResponse(BaseModel):
    """Response for batch mapping"""
    query: str
    type: str = "batch"
    results: Dict[str, List[ResultItem]]
    total_results: int
    processing_time_ms: float


class OntologyItem(BaseModel):
    """Ontology metadata"""
    id: str
    name: str
    num_classes: int
    status: str


class OntologiesResponse(BaseModel):
    """Response with ontologies list"""
    total: int
    ontologies: List[OntologyItem]


class StatsResponse(BaseModel):
    """Database and index statistics"""
    database: Dict[str, int]
    indexes: Dict[str, Any]
    configuration: Dict[str, str]


# ============================================================================
# Global State
# ============================================================================

# Database and retrieval/reranking instances (initialized at startup)
db: Optional[OntologyDB] = None
retriever: Optional[HybridRetriever] = None
reranker = None
_indexing_complete = False


# ============================================================================
# Startup and Shutdown
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown"""
    # Startup
    logger.info("=" * 80)
    logger.info("STARTING Ontology Database Concept Mapping Tool")
    logger.info("=" * 80)
    
    try:
        # Initialize database
        global db, retriever, reranker, _indexing_complete
        db = OntologyDB(db_path="bioportal.db")
        
        # Log database stats
        stats = db.get_stats()
        logger.info(f"Database loaded:")
        logger.info(f"  - Ontologies: {stats['num_ontologies']}")
        logger.info(f"  - Classes: {stats['num_classes']:,}")
        logger.info(f"  - Synonyms: {stats['num_synonyms']:,}")
        
        # Initialize retriever
        from retrieval import BM25Retriever, DenseRetriever
        bm25_weight = float(os.getenv("BM25_WEIGHT", "0.3"))
        dense_weight = float(os.getenv("DENSE_WEIGHT", "0.7"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        chroma_path = os.getenv("CHROMA_DB_PATH", ".cache/chroma_db")

        # VECTOR_BACKEND controls the dense index backend:
        #   faiss  — recommended for >1M vectors (exact cosine, builds in seconds)
        #   numpy  — in-memory .npy, exact cosine, ~300ms search
        #   chroma — ChromaDB HNSW, NOT recommended for >1M vectors
        vector_backend = os.getenv("VECTOR_BACKEND", "faiss").lower()
        use_faiss  = vector_backend == "faiss"
        use_chroma = vector_backend == "chroma"

        retriever = HybridRetriever(
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            bm25_model=BM25Retriever(),
            dense_model=DenseRetriever(
                model_name=embedding_model,
                use_chroma=use_chroma,
                chroma_path=chroma_path,
                embed_cache_dir=os.getenv("EMBED_CACHE_DIR", ".cache/embed_indexes"),
                use_faiss=use_faiss,
            ),
        )
        logger.info(f"HybridRetriever initialized: BM25={bm25_weight}, Dense={dense_weight} (backend={vector_backend})")
        
        # Initialize reranker
        reranker_type = os.getenv("RERANKER_TYPE", "ensemble")
        reranker = create_reranker(reranker_type)
        logger.info(f"Reranker initialized: {type(reranker).__name__}")
        
        logger.info("=" * 80)
        logger.info("STARTUP COMPLETE - API ready at /docs (indexing in background)")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

    # ── Launch indexing in the background so FastAPI serves immediately ──────
    # Endpoints protected by _indexing_complete return 503 until ready.
    import asyncio

    async def _run_indexing_bg():
        global _indexing_complete
        try:
            await asyncio.get_running_loop().run_in_executor(None, _build_indexes)
            # Warm up the embedding model + FAISS index so the first real
            # request doesn't pay model-init / page-in costs (~1–2s on cold start)
            await asyncio.get_running_loop().run_in_executor(None, _warmup)
            _indexing_complete = True
            logger.info("✓ All indexes ready — API fully operational")
        except Exception as exc:
            logger.error(f"Background indexing failed: {exc}", exc_info=True)

    asyncio.create_task(_run_indexing_bg())

    yield  # Running

    # Shutdown
    logger.info("Shutting down Ontology Database Concept Mapping Tool")
    db = None
    retriever = None
    reranker = None


def _build_indexes():
    """Build BM25 and dense indexes, using disk caches when valid.

    Three-tier startup:

    Tier 1 — everything cached (fastest, zero DB scanning):
        Indexes AND concepts-map pickle both match the DB count.

    Tier 2 — indexes cached, concepts-map missing/stale:
        Fast single-table scan (no JOINs) to rebuild only the pickle.

    Tier 3 — at least one index is missing/stale:
        Full enriched DB scan. Concepts-map pickle is saved *before*
        index building so a crash mid-build does not lose the concepts
        cache — the next restart falls to Tier 2, not Tier 3.
        Each sub-retriever skips its own already-cached portion.
    """
    global retriever, db, _concepts_map

    if not retriever or not db:
        logger.error("Retriever or database not initialized")
        return

    import json as _json
    import pickle

    try:
        cache_dir = os.getenv("INDEX_CACHE_DIR", ".cache/ontology_indexes")
        os.makedirs(cache_dir, exist_ok=True)
        concepts_cache_path = os.path.join(cache_dir, "concepts_cache.pkl")
        concepts_meta_path  = os.path.join(cache_dir, "concepts_cache_meta.json")

        # Single COUNT(*) — milliseconds regardless of DB size
        db_count = db.get_stats()["num_classes"]
        logger.info(f"DB contains {db_count:,} concepts")

        cached_count = 0
        if os.path.exists(concepts_meta_path):
            try:
                with open(concepts_meta_path) as f:
                    cached_count = _json.load(f).get("count", 0)
            except Exception:
                pass

        # ── Tier 1: all caches valid ─────────────────────────────────────────
        if (
            cached_count == db_count
            and os.path.exists(concepts_cache_path)
            and retriever.load_indexes("main", db_count)
        ):
            logger.info(f"Loading concepts map from cache ({db_count:,} concepts)...")
            with open(concepts_cache_path, "rb") as f:
                minimal = pickle.load(f)
            _concepts_map = {i: c for i, c in enumerate(minimal)}
            logger.info("✓ All indexes and concepts map loaded from cache — ready")
            return

        # ── Tier 2: indexes cached, concepts-map pickle missing/stale ────────
        if retriever.load_indexes("main", db_count):
            logger.info(
                f"Retrieval indexes cached; rebuilding concepts map via fast DB scan "
                f"({db_count:,} rows, no JOINs)..."
            )
            minimal = db.get_all_minimal_concepts()
            with open(concepts_cache_path, "wb") as f:
                pickle.dump(minimal, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(concepts_meta_path, "w") as f:
                _json.dump({"count": len(minimal)}, f)
            _concepts_map = {i: c for i, c in enumerate(minimal)}
            logger.info(f"✓ Concepts map rebuilt and cached ({len(minimal):,} entries)")
            return

        # ── Tier 3: at least one index is missing/stale — full enriched scan ─
        logger.info("Fetching concepts from database for indexing (with rich metadata)...")
        all_concepts: List[Dict] = []
        processed = 0
        for batch in db.get_all_concepts_for_indexing(batch_size=500):
            all_concepts.extend(batch)
            processed += len(batch)
            if processed % 5000 == 0:
                logger.info(f"  Processed {processed:,} concepts...")

        logger.info(f"Total concepts to index: {len(all_concepts):,}")

        # Save concepts-map cache BEFORE building indexes so a crash during
        # index building still preserves the concepts data (Tier 2 next time).
        minimal = [
            {
                "class_uri":       c.get("class_uri", ""),
                "preferred_label": c.get("preferred_label", ""),
                "ontology_id":     c.get("ontology_id", ""),
            }
            for c in all_concepts
        ]
        with open(concepts_cache_path, "wb") as f:
            pickle.dump(minimal, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(concepts_meta_path, "w") as f:
            _json.dump({"count": len(all_concepts)}, f)
        logger.info(f"Concepts map cache saved ({len(all_concepts):,} entries)")

        # Build indexes — each sub-retriever skips its own cached portion
        logger.info("Building BM25 and dense indexes (dense uses rich embeddings)...")
        retriever.build_index("main", all_concepts)

        _concepts_map = {i: c for i, c in enumerate(all_concepts)}
        logger.info(f"✓ Indexes built for {len(all_concepts):,} concepts")
        logger.info("✓ Rich embeddings include: labels + definitions + synonyms + parents")

    except Exception as e:
        logger.error(f"Index building failed: {e}", exc_info=True)
        raise


def _warmup():
    """Run a dummy retrieval to pre-load the embedding model and warm the FAISS
    index into OS page cache. Eliminates the 'first request is slow' effect."""
    if not retriever:
        return
    try:
        retriever.retrieve(
            query="diabetes",
            corpus_name="main",
            k=5,
            concepts_map=_concepts_map,
        )
        logger.info("✓ Warm-up query complete — first real request will be fast")
    except Exception as e:
        logger.warning(f"Warm-up query failed (non-fatal): {e}")


# Concepts map for retrieval lookups
_concepts_map: Dict[int, Dict] = {}

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Ontology Database Concept Mapping Tool",
    description="""
Hybrid BM25 + dense retrieval with configurable re-ranking for biomedical ontology concept mapping.

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /map/concept` | Map a single term to ontology concepts |
| `POST /map/search` | Map with optional context for disambiguation |
| `POST /map/batch` | Map up to 20 concepts in one request |
| `GET /ontologies` | List all available ontologies |
| `GET /stats` | Database and index statistics |
| `GET /health` | Readiness check (`indexing_complete` must be `true` before searching) |

## Request fields

Only **`text`** is required on all endpoints. Everything else is optional.

| Field | Type | Description |
|-------|------|-------------|
| `text` | string or list | Term(s) to map |
| `context` | string | Clinical/domain context for disambiguation (`/map/search` and batch objects) |
| `max_results` | int 1–20 | Results per concept (default: 5) |
| `ontologies` | string | Comma-separated ontology filter, e.g. `"SNOMEDCT,MESH,MONDO"` |
| `openrouter_api_key` | string | Override `OPENROUTER_API_KEY` env var |
| `openrouter_model` | string | Override `OPENROUTER_MODEL` env var |

## Re-ranker modes (`RERANKER_TYPE`)

| Value | Components | API key needed |
|-------|-----------|----------------|
| `dual_late` *(default)* | late_interaction + biomedical | No |
| `llm_late` | LLM + late_interaction | Yes |
| `llm_biomedical` | LLM + biomedical | Yes |
| `ensemble` | all three | Yes |
| `biomedical` | biomedical only | No |
| `late_interaction` | late_interaction only | No |
| `llm` | LLM only | Yes |

## Notes

- Search endpoints return **503** until `indexing_complete: true` in `/health`.
- Trailing commas in JSON are accepted (e.g. `{"text": "diabetes",}` is valid).
- LLM calls are concurrent — latency scales with the slowest single call, not all calls combined.
""",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Lenient JSON middleware — strips trailing commas so clients that send
#    e.g. {"text": "foo", "max_results": 5,} don't get a 422 ────────────────
class LenientJSONMiddleware(BaseHTTPMiddleware):
    _TRAILING_COMMA = re.compile(r",\s*([}\]])")

    async def dispatch(self, request: Request, call_next):
        ct = request.headers.get("content-type", "")
        if "application/json" in ct:
            raw = await request.body()
            # Always rebuild the receive callable so the body can be re-read
            # by the downstream handler (BaseHTTPMiddleware consumes the stream)
            body_to_send = raw
            if raw:
                try:
                    cleaned = self._TRAILING_COMMA.sub(r"\1", raw.decode("utf-8"))
                    body_to_send = cleaned.encode("utf-8")
                except Exception:
                    pass  # Leave body untouched; FastAPI will return its own 422

            async def receive():
                # "more_body": False tells ASGI the body is complete (no more chunks)
                return {"type": "http.request", "body": body_to_send, "more_body": False}

            request = Request(request.scope, receive)
        return await call_next(request)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LenientJSONMiddleware)


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database_ready": db is not None,
        "indexing_complete": _indexing_complete,
        "retriever_ready": retriever is not None,
        "reranker_ready": reranker is not None,
    }


@app.post("/map/concept", response_model=ConceptMappingResponse, tags=["mapping"])
def map_single_concept(request: ConceptMappingRequest):
    """
    Map a single concept to ranked ontology identifiers.

    - **text** *(required)*: term to map, e.g. `"kidney disease"`
    - **context** *(optional)*: clinical/domain context to improve disambiguation, e.g. `"progressive decline in GFR"`
    - **max_results** *(optional)*: 1–20, default 5
    - **ontologies** *(optional)*: comma-separated filter, e.g. `"SNOMEDCT,MESH,MONDO"`
    - **openrouter_api_key / openrouter_model** *(optional)*: override env vars for LLM reranking
    """
    start_time = time.time()
    
    if not db or not retriever or not reranker or not _indexing_complete:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Parse ontology filter
        ontology_list = None
        if request.ontologies:
            ontology_list = [o.strip().upper() for o in request.ontologies.split(",") if o.strip()]
        
        # Combine text + context for retrieval (same logic as /map/search)
        retrieval_query = f"{request.text} {request.context}" if request.context else request.text
        logger.info(f"[/map/concept] query='{request.text}' max_results={request.max_results}")

        _t_retrieve_start = time.time()
        candidates = retriever.retrieve(
            query=retrieval_query,
            corpus_name="main",
            k=int(os.getenv("MAX_CANDIDATES", "20")),
            concepts_map=_concepts_map,
        )
        _t_retrieve_ms = round((time.time() - _t_retrieve_start) * 1000, 1)
        logger.info(f"[/map/concept] retrieval={_t_retrieve_ms}ms  got {len(candidates)} candidates")

        if not candidates:
            return ConceptMappingResponse(
                query=request.text,
                type="single",
                results=[],
                total_results=0,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Convert candidates to dict format for reranking
        candidates_list = [
            {
                "class_uri": c.class_uri,
                "preferred_label": c.preferred_label,
                "ontology_id": c.ontology_id,
                "definition": "",  # Would fetch from DB if needed
                "original_score": c.combined_score,
            }
            for c in candidates
        ]

        # Re-rank candidates (request-level API key/model override env values)
        _t_rerank_start = time.time()
        reranked = reranker.rerank(
            query=request.text,
            candidates=candidates_list,
            top_k=request.max_results,
            openrouter_api_key=request.openrouter_api_key,
            openrouter_model=request.openrouter_model,
        )
        _t_rerank_ms = round((time.time() - _t_rerank_start) * 1000, 1)
        logger.info(f"[/map/concept] reranking={_t_rerank_ms}ms  top {len(reranked)} results")

        # Format results
        results = [
            ResultItem(
                rank=i + 1,
                ontology_id=r.class_uri,
                ontology_label=r.preferred_label,
                ontology=r.ontology_id,
                original_score=r.original_score,
                llm_score=r.llm_score,
                late_interaction_score=r.late_interaction_score,
                final_score=r.final_score,
                retrieval_scores={
                    "bm25": next(
                        (c.bm25_score for c in candidates if c.class_uri == r.class_uri),
                        0.0
                    ),
                    "dense": next(
                        (c.embedding_score for c in candidates if c.class_uri == r.class_uri),
                        0.0
                    ),
                }
            )
            for i, r in enumerate(reranked)
        ]
        
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Mapping complete: {len(results)} results in {elapsed_ms}ms")
        
        return ConceptMappingResponse(
            query=request.text,
            type="single",
            results=results,
            total_results=len(results),
            processing_time_ms=elapsed_ms,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Concept mapping failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/map/batch", response_model=BatchMappingResponse, tags=["mapping"])
def map_batch_concepts(request: BatchMappingRequest):
    """
    Map up to 1000 concepts in one request.

    **`text`** *(required)* — three accepted formats:

    1. **List of objects with optional context** *(recommended)*:
       ```json
       {"text": [{"text": "kidney disease", "context": "progressive decline in GFR"}, {"text": "T2DM"}]}
       ```
    2. **List of strings**: `{"text": ["diabetes", "asthma"]}`
    3. **Comma-separated string**: `{"text": "diabetes,asthma"}`

    - **max_results** *(optional)*: 1–20 per concept, default 5
    - **ontologies** *(optional)*: comma-separated filter, e.g. `"SNOMEDCT,MESH"`

    3. **List of objects with context (recommended for better accuracy):**
       ```json
       {
         "text": [
           {"text": "diabetes", "context": "Type 2 with complications"},
           {"text": "cancer", "context": "lung cancer with metastasis"},
           {"text": "kidney disease", "context": "progressive decline in GFR"}
         ],
         "max_results": 3
       }
       ```
    
    - **max_results**: Results per concept (1-20, default: 5)
    - **ontologies**: Optional ontology filter (comma-separated)
    
    Returns: Mapping for each concept separately with scores
    """
    start_time = time.time()
    
    if not db or not retriever or not reranker or not _indexing_complete:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Parse concepts and contexts from various input formats
        concept_context_pairs: List[tuple] = []  # [(concept, context), ...]
        original_text = ""
        
        if isinstance(request.text, str):
            # Format 1: Comma-separated string
            original_text = request.text
            concepts = [c.strip() for c in request.text.split(",") if c.strip()]
            concept_context_pairs = [(c, None) for c in concepts]
            
        elif isinstance(request.text, list):
            # Format 2 & 3: List of concepts or objects
            original_text = str(request.text)
            for item in request.text:
                if isinstance(item, str):
                    # Format 2: Simple string
                    concept_context_pairs.append((item.strip(), None))
                elif isinstance(item, dict):
                    # Format 3: Object with text and optional context
                    concept_text = item.get("text", "").strip()
                    concept_context = item.get("context", "")
                    if concept_text:
                        concept_context_pairs.append((concept_text, concept_context or None))
                else:
                    logger.warning(f"Skipping invalid item type: {type(item)}")
        
        if not concept_context_pairs:
            raise HTTPException(status_code=400, detail="No valid concepts provided")

        if len(concept_context_pairs) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 concepts per request")
        
        logger.info(f"[/map/batch] mapping {len(concept_context_pairs)} concepts")
        for concept, context in concept_context_pairs:
            context_note = f" (context: {context[:100]}...)" if context else ""
            logger.info(f"  - {concept}{context_note}")
        
        # Parse ontologies
        ontology_list = None
        if request.ontologies:
            ontology_list = [o.strip().upper() for o in request.ontologies.split(",") if o.strip()]
        
        # Map each concept with optional context
        results_dict = {}
        for concept_text, concept_context in concept_context_pairs:
            _t_concept_start = time.time()
            # Use context if provided for better retrieval
            retrieval_query = concept_text
            reranking_query = concept_text

            if concept_context:
                # Use combined query for retrieval (includes context for semantic understanding)
                retrieval_query = f"{concept_text} {concept_context}"
                # But keep original for reranking scoring clarity
                reranking_query = concept_text

            # Retrieve candidates
            _t_r0 = time.time()
            candidates = retriever.retrieve(
                query=retrieval_query,
                corpus_name="main",
                k=int(os.getenv("MAX_CANDIDATES", "20")),
                concepts_map=_concepts_map,
            )
            _t_r1 = time.time()
            logger.info(
                f"[/map/batch] '{concept_text}' retrieval={round((_t_r1-_t_r0)*1000,1)}ms "
                f"candidates={len(candidates)}"
            )

            if not candidates:
                results_dict[concept_text] = []
                continue

            # Re-rank using primary concept text
            candidates_list = [
                {
                    "class_uri": c.class_uri,
                    "preferred_label": c.preferred_label,
                    "ontology_id": c.ontology_id,
                    "definition": "",
                    "original_score": c.combined_score,
                }
                for c in candidates
            ]

            _t_rr0 = time.time()
            reranked = reranker.rerank(
                query=reranking_query,
                candidates=candidates_list,
                top_k=request.max_results,
                openrouter_api_key=request.openrouter_api_key,
                openrouter_model=request.openrouter_model,
            )
            _t_rr1 = time.time()
            logger.info(
                f"[/map/batch] '{concept_text}' reranking={round((_t_rr1-_t_rr0)*1000,1)}ms "
                f"concept_total={round((_t_rr1-_t_concept_start)*1000,1)}ms"
            )
            
            # Format results
            results_dict[concept_text] = [
                ResultItem(
                    rank=i + 1,
                    ontology_id=r.class_uri,
                    ontology_label=r.preferred_label,
                    ontology=r.ontology_id,
                    original_score=r.original_score,
                    llm_score=r.llm_score,
                    late_interaction_score=r.late_interaction_score,
                    final_score=r.final_score,
                )
                for i, r in enumerate(reranked)
            ]
        
        total_results = sum(len(v) for v in results_dict.values())
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Batch mapping complete: {total_results} total results in {elapsed_ms}ms")
        
        return BatchMappingResponse(
            query=original_text,
            type="batch",
            results=results_dict,
            total_results=total_results,
            processing_time_ms=elapsed_ms,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch mapping failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/map/search", response_model=ConceptMappingResponse, tags=["mapping"])
def contextual_search(request: ContextualSearchRequest):
    """
    Map with context for disambiguation — same fields as `/map/concept`.

    - **text** *(required)*: term to map
    - **context** *(optional)*: clinical/domain context, e.g. `"upper respiratory infection"` for `"cold"`
    - **max_results** *(optional)*: 1–20, default 5
    - **ontologies** *(optional)*: comma-separated filter, e.g. `"SNOMEDCT,MESH,MONDO"`
    - **openrouter_api_key / openrouter_model** *(optional)*: override env vars for LLM reranking
    """
    start_time = time.time()
    
    if not db or not retriever or not reranker or not _indexing_complete:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Combine query and context for retrieval
        combined_query = request.text
        if request.context:
            combined_query = f"{request.text} {request.context}"
        
        logger.info(f"[/map/search] query='{request.text}' max_results={request.max_results}")

        # Retrieve candidates
        _t_retrieve_start = time.time()
        candidates = retriever.retrieve(
            query=combined_query,
            corpus_name="main",
            k=int(os.getenv("MAX_CANDIDATES", "30")),  # Slightly larger for context
            concepts_map=_concepts_map,
        )
        _t_retrieve_ms = round((time.time() - _t_retrieve_start) * 1000, 1)
        logger.info(f"[/map/search] retrieval={_t_retrieve_ms}ms  got {len(candidates)} candidates")

        if not candidates:
            return ConceptMappingResponse(
                query=request.text,
                type="single",
                results=[],
                total_results=0,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Re-rank (only on primary text for scoring consistency)
        candidates_list = [
            {
                "class_uri": c.class_uri,
                "preferred_label": c.preferred_label,
                "ontology_id": c.ontology_id,
                "definition": "",
                "original_score": c.combined_score,
            }
            for c in candidates
        ]

        _t_rerank_start = time.time()
        reranked = reranker.rerank(
            query=request.text,
            candidates=candidates_list,
            top_k=request.max_results,
            openrouter_api_key=request.openrouter_api_key,
            openrouter_model=request.openrouter_model,
        )
        _t_rerank_ms = round((time.time() - _t_rerank_start) * 1000, 1)
        logger.info(f"[/map/search] reranking={_t_rerank_ms}ms  top {len(reranked)} results")

        # Format results
        results = [
            ResultItem(
                rank=i + 1,
                ontology_id=r.class_uri,
                ontology_label=r.preferred_label,
                ontology=r.ontology_id,
                original_score=r.original_score,
                llm_score=r.llm_score,
                late_interaction_score=r.late_interaction_score,
                final_score=r.final_score,
            )
            for i, r in enumerate(reranked)
        ]
        
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Contextual search complete: {len(results)} results in {elapsed_ms}ms")
        
        return ConceptMappingResponse(
            query=request.text,
            type="single",
            results=results,
            total_results=len(results),
            processing_time_ms=elapsed_ms,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Contextual search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ontologies", response_model=OntologiesResponse, tags=["metadata"])
async def get_ontologies():
    """
    Retrieve available ontologies
    
    Returns list of all ontologies in the database with metadata
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        ontologies = db.get_ontologies()
        return OntologiesResponse(
            total=len(ontologies),
            ontologies=[
                OntologyItem(
                    id=o.id,
                    name=o.name,
                    num_classes=o.num_classes,
                    status=o.status,
                )
                for o in ontologies
            ]
        )
    except Exception as e:
        logger.error(f"Failed to retrieve ontologies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse, tags=["metadata"])
async def get_stats():
    """
    Get database and index statistics
    
    Returns information about database size, indexes, and configuration
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        db_stats = db.get_stats()
        
        # Get configuration
        config = {
            "BM25_WEIGHT": os.getenv("BM25_WEIGHT", "0.3"),
            "DENSE_WEIGHT": os.getenv("DENSE_WEIGHT", "0.7"),
            "RERANKER_TYPE": os.getenv("RERANKER_TYPE", "ensemble"),
            "MAX_CANDIDATES": os.getenv("MAX_CANDIDATES", "20"),
        }
        
        return StatsResponse(
            database=db_stats,
            indexes={
                "bm25_indexed": _indexing_complete,
                "dense_indexed": _indexing_complete,
                "cache_dir": os.getenv("INDEX_CACHE_DIR", ".cache/ontology_indexes"),
                "num_indexed_concepts": len(_concepts_map),
            },
            configuration=config,
        )
    except Exception as e:
        logger.error(f"Failed to retrieve stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config", tags=["metadata"])
async def get_config():
    """
    Return current server configuration (non-sensitive settings only).

    Useful for logging the active config alongside test results, confirming
    which model / reranker / backend is active without reading .env files.
    """
    return {
        "retrieval": {
            "bm25_weight":       float(os.getenv("BM25_WEIGHT",   "0.3")),
            "dense_weight":      float(os.getenv("DENSE_WEIGHT",  "0.7")),
            "max_candidates":    int(os.getenv("MAX_CANDIDATES",  "20")),
            "embedding_model":   os.getenv("EMBEDDING_MODEL",     "BAAI/bge-small-en-v1.5"),
            "vector_backend":    os.getenv("VECTOR_BACKEND",      "faiss"),
        },
        "reranking": {
            "reranker_type":            os.getenv("RERANKER_TYPE",            "dual_late"),
            "llm_weight":               float(os.getenv("LLM_WEIGHT",               "0.5")),
            "late_interaction_weight":  float(os.getenv("LATE_INTERACTION_WEIGHT",  "0.3")),
            "biomedical_weight":        float(os.getenv("BIOMEDICAL_WEIGHT",        "0.2")),
            "late_interaction_model":   os.getenv("LATE_INTERACTION_MODEL",   "jinaai/jina-colbert-v2"),
            "openrouter_model":         os.getenv("OPENROUTER_MODEL",         "openrouter/auto"),
            "openrouter_api_key_set":   bool(os.getenv("OPENROUTER_API_KEY")),
        },
        "server": {
            "host":       os.getenv("API_HOST",     "0.0.0.0"),
            "port":       int(os.getenv("API_PORT", "8000")),
            "log_level":  os.getenv("LOG_LEVEL",    "INFO"),
        },
        "cache": {
            "index_cache_dir": os.getenv("INDEX_CACHE_DIR", ".cache/ontology_indexes"),
            "embed_cache_dir": os.getenv("EMBED_CACHE_DIR", ".cache/embed_indexes"),
            "database_path":   os.getenv("DATABASE_PATH",  "bioportal.db"),
        },
        "status": {
            "indexing_complete":  _indexing_complete,
            "num_indexed_concepts": len(_concepts_map) if _concepts_map else 0,
        },
    }


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
