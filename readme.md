## Configuration

Environment variables can be set in `.env` (auto-loaded) or exported in the shell.

### Retrieval
```bash
BM25_WEIGHT=0.3              # BM25 share of hybrid score (default: 0.3)
DENSE_WEIGHT=0.7             # Dense share of hybrid score (default: 0.7)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2   # Embedding model

VECTOR_BACKEND=faiss         # faiss (default) | numpy | chroma
EMBED_CACHE_DIR=.cache/embed_indexes    # Where .npy and FAISS index are stored
CHROMA_DB_PATH=.cache/chroma_db        # Only used when VECTOR_BACKEND=chroma
```

### Re-ranking
```bash
RERANKER_TYPE=ensemble       # Options: ensemble, llm, late_interaction, biomedical

# Ensemble weights (auto-normalized)
LLM_WEIGHT=0.5
LATE_INTERACTION_WEIGHT=0.3
BIOMEDICAL_WEIGHT=0.2
```

### LLM Re-ranker (OpenRouter)
```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxx   # Required for LLM reranking
OPENROUTER_MODEL=openrouter/auto     # Model selection

# Available models (examples):
#   openrouter/auto                   (auto-select best available)
#   google/gemini-2.0-flash-001       (fast, high quality)
#   anthropic/claude-3.5-sonnet:beta  (highest quality)
#   meta-llama/llama-2-70b-chat       (open source)
#   mistral/mistral-7b-instruct:free  (free tier)
```