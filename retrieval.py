# -*- coding: utf-8 -*-
"""
Retrieval module: BM25 lexical search + Dense vector search (ANN)
Provides candidate set for re-ranking

Supports multiple vector database backends:
- In-memory numpy (default)
- Chroma DB (persistent, scalable)
- HNSWLIB (fast ANN)
"""

import json
import logging
import threading
import time
import numpy as np
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
import os


logger = logging.getLogger(__name__)


def build_rich_concept_text(concept_data: Dict[str, Any]) -> str:
    """
    Build rich text representation combining multiple concept fields
    for better embedding generation
    
    Args:
        concept_data: Dictionary with concept information
        
    Returns:
        Combined text string for embedding
    """
    parts = []
    
    # Primary label (highest weight)
    if concept_data.get("preferred_label"):
        parts.append(concept_data["preferred_label"])
    
    # Definition (important semantic content)
    if concept_data.get("definition"):
        definition = concept_data["definition"]
        # Use full definition to preserve semantic content
        parts.append(definition)
    
    # Other labels (synonyms, alt labels)
    if concept_data.get("labels"):
        labels = concept_data["labels"]
        if isinstance(labels, list):
            parts.extend([l for l in labels if l])
        elif isinstance(labels, str):
            parts.append(labels)
    
    # Synonyms (alternative names)
    if concept_data.get("synonyms"):
        synonyms = concept_data["synonyms"]
        if isinstance(synonyms, list):
            parts.extend([s for s in synonyms if s])
        elif isinstance(synonyms, str):
            parts.append(synonyms)
    
    # Parent labels (hierarchical context)
    if concept_data.get("parent_labels"):
        parent_labels = concept_data["parent_labels"]
        if isinstance(parent_labels, list):
            parts.extend([p for p in parent_labels if p])
        elif isinstance(parent_labels, str):
            parts.append(parent_labels)
    
    # Ontology context
    if concept_data.get("ontology_id"):
        parts.append(f"ontology: {concept_data['ontology_id']}")
    
    # Join and clean
    combined = " ".join(str(p).strip() for p in parts if p)
    return combined


@dataclass
class RetrievalCandidate:
    """A candidate concept with its scores"""
    class_uri: str
    preferred_label: str
    ontology_id: str
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    combined_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_uri": self.class_uri,
            "preferred_label": self.preferred_label,
            "ontology_id": self.ontology_id,
            "bm25_score": float(self.bm25_score),
            "embedding_score": float(self.embedding_score),
            "combined_score": float(self.combined_score),
        }


class BM25Retriever:
    """BM25 lexical retrieval using bm25s"""

    def __init__(self, cache_dir: str = ".cache/bm25_indexes"):
        """
        Initialize BM25 retriever
        
        Args:
            cache_dir: Directory to store/load BM25 indexes
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.indexes = {}  # corpus_name -> corpus
        self.retrievers = {}  # corpus_name -> BM25
        self._meta_cache: Dict[str, Any] = {}  # in-memory cache to avoid file I/O per query
        logger.info(f"BM25Retriever initialized with cache_dir={cache_dir}")

    def _meta_path(self, corpus_name: str) -> str:
        return os.path.join(self.cache_dir, f"{corpus_name}_meta.json")

    def _index_dir(self, corpus_name: str) -> str:
        return os.path.join(self.cache_dir, corpus_name)

    def _read_meta(self, corpus_name: str) -> Dict[str, Any]:
        if corpus_name in self._meta_cache:
            return self._meta_cache[corpus_name]
        p = self._meta_path(corpus_name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    meta = json.load(f)
                self._meta_cache[corpus_name] = meta
                return meta
            except Exception:
                pass
        return {}

    def _write_meta(self, corpus_name: str, count: int) -> None:
        with open(self._meta_path(corpus_name), "w") as f:
            json.dump({"count": count}, f)
        self._meta_cache[corpus_name] = {"count": count}

    def build_index(self, corpus_name: str, texts: List[str]):
        """
        Build (or load from cache) a BM25 index for a corpus.

        Cache behaviour
        ---------------
        - Count matches stored count  → load from disk, skip rebuild entirely.
        - Count differs               → full rebuild (BM25 IDF must be recalculated)
                                        and save to disk for next restart.

        Args:
            corpus_name: Name of the corpus
            texts: List of documents to index
        """
        try:
            import bm25s

            index_dir = self._index_dir(corpus_name)
            meta = self._read_meta(corpus_name)

            if meta.get("count") == len(texts) and os.path.isdir(index_dir):
                logger.info(
                    f"Loading BM25 index '{corpus_name}' from cache ({len(texts):,} docs)"
                )
                # load_corpus=False → retrieve() returns integer indices, not corpus token lists
                retriever = bm25s.BM25.load(index_dir, load_corpus=False)
                self.retrievers[corpus_name] = retriever
                return

            if meta.get("count") and meta["count"] != len(texts):
                logger.info(
                    f"BM25 cache stale ({meta['count']:,} stored vs {len(texts):,}), rebuilding"
                )

            logger.info(f"Building BM25 index '{corpus_name}' ({len(texts):,} docs)...")
            corpus = [text.lower().split() for text in texts]
            retriever = bm25s.BM25()
            retriever.index(corpus)
            # Save corpus to disk for future loads, then clear it from memory
            # so retrieve() returns integer indices (not corpus token lists)
            os.makedirs(index_dir, exist_ok=True)
            retriever.save(index_dir, corpus=corpus)
            retriever.corpus = None
            self.retrievers[corpus_name] = retriever
            self._write_meta(corpus_name, len(texts))
            logger.info(f"BM25 index saved to {index_dir}")

        except ImportError:
            logger.warning("bm25s not installed, using fallback implementation")
            self._build_fallback_index(corpus_name, texts)

    def load_cached_index(self, corpus_name: str, count: int) -> bool:
        """Load BM25 index from disk cache if the stored count matches *count*.

        Returns True if the cache was loaded successfully, False otherwise.
        """
        index_dir = self._index_dir(corpus_name)
        meta = self._read_meta(corpus_name)
        if meta.get("count") != count or not os.path.isdir(index_dir):
            return False
        try:
            import bm25s
            retriever = bm25s.BM25.load(index_dir, load_corpus=False)
            self.retrievers[corpus_name] = retriever
            logger.info(f"BM25 index '{corpus_name}' loaded from cache ({count:,} docs)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 cache for '{corpus_name}': {e}")
            return False

    def _build_fallback_index(self, corpus_name: str, texts: List[str]):
        """Fallback simple BM25-like implementation"""
        # Store texts and build simple inverted index
        self.indexes[corpus_name] = {
            "texts": texts,
            "inverted_index": self._build_inverted_index(texts)
        }

    def _build_inverted_index(self, texts: List[str]) -> Dict[str, List[int]]:
        """Build simple inverted index"""
        inverted_index = {}
        for doc_id, text in enumerate(texts):
            tokens = text.lower().split()
            for token in set(tokens):
                if token not in inverted_index:
                    inverted_index[token] = []
                inverted_index[token].append(doc_id)
        return inverted_index

    def retrieve(
        self,
        query: str,
        corpus_name: str = "default",
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents for query using BM25
        
        Args:
            query: Query text
            corpus_name: Name of corpus to search
            k: Number of results to return
            
        Returns:
            List of (doc_index, score) tuples
        """
        if corpus_name not in self.retrievers and corpus_name not in self.indexes:
            logger.warning(f"Corpus '{corpus_name}' not found")
            return []

        try:
            import bm25s
            import numpy as np
            if corpus_name in self.retrievers:
                retriever = self.retrievers[corpus_name]
                # stopwords=None avoids any file download; common stopwords
                # have negligible impact on retrieval quality for short queries
                query_tokens = bm25s.tokenize(query, stopwords=None)
                # num_docs from stored meta (corpus not loaded into memory)
                meta = self._read_meta(corpus_name)
                num_docs = meta.get("count", 0) or getattr(retriever, "num_docs", 0)
                safe_k = min(k, num_docs) if num_docs > 0 else k
                # corpus=None → retrieve() returns (indices, scores), both shape (1, k)
                indices, scores = retriever.retrieve(query_tokens, corpus=None, k=safe_k)
                idx = indices[0]
                sc = scores[0]
                return list(zip(idx.tolist(), sc.tolist()))
        except ImportError:
            # Fallback retrieval
            if corpus_name in self.indexes:
                return self._fallback_retrieve(query, corpus_name, k)
        return []

    def _fallback_retrieve(self, query: str, corpus_name: str, k: int) -> List[Tuple[int, float]]:
        """Fallback retrieval using simple TF-IDF"""
        index = self.indexes[corpus_name]
        inverted_index = index["inverted_index"]
        query_tokens = set(query.lower().split())
        scores = {}
        
        for token in query_tokens:
            if token in inverted_index:
                for doc_id in inverted_index[token]:
                    scores[doc_id] = scores.get(doc_id, 0) + 1
        
        # Sort by score and return top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]


class DenseRetriever:
    """Dense vector retrieval using sentence transformers with optional Chroma DB backend"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_chroma: bool = False,
        chroma_path: str = ".cache/chroma_db",
        embed_cache_dir: str = ".cache/embed_indexes",
        use_faiss: bool = False,
    ):
        """
        Initialize dense retriever

        Args:
            model_name: Hugging Face model name for embeddings
            use_chroma: Use Chroma DB for vector storage
            chroma_path: Path to Chroma DB storage
            embed_cache_dir: Directory for caching embeddings (.npy and FAISS index)
            use_faiss: Use FAISS IndexFlatIP for fast exact cosine search (recommended
                       for >1M vectors; builds in seconds, searches in ~50ms)
        """
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            if torch.backends.mps.is_available():
                device = "mps"      # Apple Silicon GPU (Metal Performance Shaders)
            elif torch.cuda.is_available():
                device = "cuda"     # NVIDIA GPU
            else:
                device = "cpu"
            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
            logger.info(f"DenseRetriever initialized with {model_name} on {device.upper()}")
        except ImportError:
            logger.warning("sentence-transformers not installed")
            self.model = None

        self.corpus_embeddings: Dict[str, Any] = {}
        self.corpus_texts: Dict[str, Any] = {}
        self.corpus_metadata: Dict[str, Any] = {}

        # Small LRU cache for query embeddings — avoids re-encoding the same
        # query (e.g. repeated test calls, warm-up probe, identical batch terms)
        self._query_embed_cache: Dict[str, Any] = {}
        self._query_cache_max = 256

        # MPS (Apple Silicon) and CUDA are NOT thread-safe for concurrent
        # encode() calls from different OS threads.  A single lock ensures
        # only one thread calls model.encode() at a time, preventing hangs
        # when the Starlette thread pool runs route handlers concurrently.
        self._encode_lock = threading.Lock()

        self.embed_cache_dir = embed_cache_dir
        os.makedirs(embed_cache_dir, exist_ok=True)

        # FAISS takes priority over Chroma when both are requested
        self.use_faiss = use_faiss and self._try_init_faiss()
        self.faiss_indexes: Dict[str, Any] = {}

        self.use_chroma = (not self.use_faiss) and use_chroma and self._try_init_chroma(chroma_path)
        self.chroma_path = chroma_path
        self.chroma_collections: Dict[str, Any] = {}

        backend = "FAISS" if self.use_faiss else ("Chroma" if self.use_chroma else "numpy")
        logger.info(f"DenseRetriever vector backend: {backend}")

    def _try_init_faiss(self) -> bool:
        """Check that faiss-cpu (or faiss-gpu) is importable."""
        try:
            import faiss  # noqa: F401
            logger.info("FAISS available — using IndexFlatIP (exact cosine)")
            return True
        except ImportError:
            logger.warning("faiss-cpu not installed — run: pip install faiss-cpu")
            return False

    def _faiss_paths(self, corpus_name: str) -> Tuple[str, str]:
        """Return (index_path, meta_path) for the FAISS index."""
        base = os.path.join(self.embed_cache_dir, corpus_name)
        return base + "_faiss.index", base + "_faiss_meta.json"

    def _try_init_chroma(self, chroma_path: str) -> bool:
        """Try to initialize Chroma DB"""
        try:
            import chromadb
            os.makedirs(chroma_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            logger.info(f"Chroma DB initialized at {chroma_path}")
            return True
        except ImportError:
            logger.warning("chromadb not installed, using in-memory embeddings")
            return False
        except Exception as e:
            logger.warning(f"Chroma DB initialization failed: {e}")
            return False

    def build_index(self, corpus_name: str, concepts_data: List[Dict[str, Any]]):
        """
        Build embedding index with rich concept text

        Args:
            corpus_name: Name of the corpus
            concepts_data: List of concept dicts with rich information
        """
        if self.model is None:
            logger.warning("Model not available, skipping embedding index")
            return

        try:
            texts = [build_rich_concept_text(c) for c in concepts_data]
            logger.info(f"Building dense index for {corpus_name} with {len(texts)} concepts")

            if self.use_faiss:
                self._build_faiss_index(corpus_name, texts, concepts_data)
            elif self.use_chroma:
                self._build_chroma_index(corpus_name, texts, concepts_data)
            else:
                self._build_memory_index(corpus_name, texts, concepts_data)

        except Exception as e:
            logger.error(f"Failed to build dense index: {e}")
            self.use_chroma = False

    def _build_faiss_index(self, corpus_name: str, texts: List[str], concepts_data: List[Dict[str, Any]]):
        """Build or load a FAISS IndexFlatIP (exact cosine similarity).

        Reuses the .npy embedding cache so encoding is skipped when vectors
        are already on disk (e.g. built by a previous numpy or Chroma run).

        Index type: IndexFlatIP with L2-normalised vectors → cosine similarity.
        Build:  O(n) — just copies the float32 array into a C++ structure.
        Search: ~50ms for 4M × 384 on a modern CPU.
        """
        import faiss

        index_path, meta_path = self._faiss_paths(corpus_name)
        npy_path, npy_meta_path = self._embed_cache_paths(corpus_name)
        current_count = len(texts)

        # ── Already up to date ───────────────────────────────────────────────
        stored = 0
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    stored = json.load(f).get("count", 0)
            except Exception:
                pass
        if stored == current_count and os.path.exists(index_path):
            logger.info(
                f"FAISS index '{corpus_name}' up to date ({current_count:,}) "
                f"— memory-mapping from disk (IO_FLAG_MMAP)"
            )
            self.faiss_indexes[corpus_name] = faiss.read_index(
                index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            )
            return

        # ── Get embeddings: reuse .npy cache if available ────────────────────
        npy_stored = 0
        if os.path.exists(npy_meta_path):
            try:
                with open(npy_meta_path) as f:
                    npy_stored = json.load(f).get("count", 0)
            except Exception:
                pass

        if npy_stored == current_count and os.path.exists(npy_path):
            logger.info(f"Reusing existing .npy cache for FAISS build ({current_count:,} vectors)")
            embeddings = np.load(npy_path)
        else:
            assert self.model is not None
            logger.info(f"Encoding {current_count:,} documents for FAISS (batch_size=512)...")
            embeddings = self.model.encode(
                texts, show_progress_bar=True, batch_size=512, convert_to_numpy=True
            )
            np.save(npy_path, embeddings)
            with open(npy_meta_path, "w") as f:
                json.dump({"count": current_count}, f)
            logger.info(f"Embedding cache saved ({current_count:,} vectors)")

        # ── Build FAISS index ────────────────────────────────────────────────
        logger.info(f"Building FAISS IndexFlatIP for '{corpus_name}' ({current_count:,} vectors)...")
        emb_f32 = embeddings.astype(np.float32)
        faiss.normalize_L2(emb_f32)          # in-place L2 norm → cosine via IP
        dim = emb_f32.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb_f32)

        faiss.write_index(index, index_path)
        with open(meta_path, "w") as f:
            json.dump({"count": current_count}, f)
        self.faiss_indexes[corpus_name] = index
        logger.info(f"FAISS index built and saved: {current_count:,} vectors, dim={dim}")

    def _embed_cache_paths(self, corpus_name: str) -> Tuple[str, str]:
        """Return (npy_path, meta_path) for the given corpus."""
        base = os.path.join(self.embed_cache_dir, corpus_name)
        return base + "_embeddings.npy", base + "_meta.json"

    def _build_memory_index(self, corpus_name: str, texts: List[str], concepts_data: List[Dict[str, Any]]):
        """Build (or incrementally update) an in-memory numpy embedding index.

        Cache behaviour
        ---------------
        - Count matches stored count  → load .npy from disk, skip all encoding.
        - stored < current (additions) → load existing .npy, encode only new
                                          texts, concatenate, save updated .npy.
        - stored > current (deletions) → full re-encode and save.
        - No cache                     → full encode and save.
        """
        npy_path, meta_path = self._embed_cache_paths(corpus_name)

        stored_count = 0
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    stored_count = json.load(f).get("count", 0)
            except Exception:
                stored_count = 0

        current_count = len(texts)

        if stored_count == current_count and os.path.exists(npy_path):
            logger.info(
                f"Loading in-memory embeddings from cache '{corpus_name}' ({current_count:,} docs)"
            )
            embeddings = np.load(npy_path)

        elif 0 < stored_count < current_count and os.path.exists(npy_path):
            assert self.model is not None  # guarded by build_index caller
            new_count = current_count - stored_count
            logger.info(
                f"Incrementally encoding {new_count:,} new embeddings "
                f"(existing {stored_count:,} loaded from cache)"
            )
            existing = np.load(npy_path)
            new_embeddings = self.model.encode(
                texts[stored_count:],
                show_progress_bar=True,
                batch_size=512,
                convert_to_numpy=True,
            )
            embeddings = np.vstack([existing, new_embeddings])
            np.save(npy_path, embeddings)
            with open(meta_path, "w") as f:
                json.dump({"count": current_count}, f)
            logger.info(f"Updated embedding cache: {current_count:,} total")

        else:
            assert self.model is not None  # guarded by build_index caller
            if stored_count > current_count:
                logger.info(
                    f"Embedding cache has more entries than DB "
                    f"({stored_count:,} vs {current_count:,}) — full re-encode"
                )
            else:
                logger.info(f"Encoding {current_count:,} documents (batch_size=512)...")
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=512,
                convert_to_numpy=True,
            )
            np.save(npy_path, embeddings)
            with open(meta_path, "w") as f:
                json.dump({"count": current_count}, f)
            logger.info(f"Embedding cache saved: {current_count:,} docs")

        self.corpus_embeddings[corpus_name] = embeddings
        self.corpus_texts[corpus_name] = texts
        self.corpus_metadata[corpus_name] = concepts_data
        logger.info(f"In-memory dense index ready: {current_count:,} embeddings")

    def _chroma_meta_path(self, corpus_name: str) -> str:
        return os.path.join(self.chroma_path, f"{corpus_name}_index_meta.json")

    def _chroma_read_meta(self, corpus_name: str) -> Dict[str, Any]:
        p = self._chroma_meta_path(corpus_name)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _chroma_write_meta(self, corpus_name: str, count: int) -> None:
        with open(self._chroma_meta_path(corpus_name), "w") as f:
            json.dump({"count": count}, f)

    def _chroma_add_batch(self, collection: Any, corpus_name: str, start: int, end: int,
                          embeddings: Any, texts: List[str], concepts_data: List[Dict[str, Any]]) -> None:
        """Insert one chunk of embeddings into a Chroma collection."""
        collection.add(
            ids=[f"{corpus_name}_{i}" for i in range(start, end)],
            embeddings=embeddings[start:end].tolist() if hasattr(embeddings, "tolist") else list(embeddings[start:end]),
            documents=texts[start:end],
            metadatas=[
                {
                    "class_uri": concepts_data[i].get("class_uri", ""),
                    "preferred_label": concepts_data[i].get("preferred_label", ""),
                    "ontology_id": concepts_data[i].get("ontology_id", ""),
                    "index": str(i),
                }
                for i in range(start, end)
            ],
        )

    def _build_chroma_index(self, corpus_name: str, texts: List[str], concepts_data: List[Dict[str, Any]]):
        """Build or incrementally update a Chroma DB index.

        Cache behaviour (positional IDs: ``{corpus_name}_{i}``)
        --------------------------------------------------------
        - stored count == current count  → reuse collection as-is.
        - stored count  < current count  → additions only: encode only new
                                           texts and add them; existing
                                           embeddings are untouched.
        - stored count  > current count  → deletions detected (positions
                                           shift); delete and rebuild fully.
        - Collection missing             → build from scratch.

        The stored count is kept in a small sidecar JSON next to the Chroma
        directory so we never need to query all IDs from the collection.
        """
        assert self.model is not None  # guarded by build_index caller
        try:
            meta = self._chroma_read_meta(corpus_name)
            stored_count: int = meta.get("count", 0)
            current_count = len(texts)

            # ── Try to get or create the collection ──────────────────────────
            try:
                collection = self.chroma_client.get_collection(name=corpus_name)
                collection_exists = True
            except Exception:
                collection_exists = False
                collection = None

            # ── Case 1: fully up to date ──────────────────────────────────────
            if collection_exists and stored_count == current_count:
                logger.info(
                    f"Chroma collection '{corpus_name}' is up to date "
                    f"({current_count:,} embeddings) — skipping rebuild"
                )
                self.chroma_collections[corpus_name] = collection
                return

            # ── Case 2: deletions detected → full rebuild ─────────────────────
            if collection_exists and stored_count > current_count:
                logger.info(
                    f"Chroma: deletions detected ({stored_count:,} stored vs "
                    f"{current_count:,} in DB) — rebuilding from scratch"
                )
                self.chroma_client.delete_collection(name=corpus_name)
                collection_exists = False
                stored_count = 0

            # ── Case 3: additions only ────────────────────────────────────────
            if collection_exists and 0 < stored_count < current_count:
                new_count = current_count - stored_count
                logger.info(
                    f"Chroma: encoding {new_count:,} new embeddings "
                    f"(existing {stored_count:,} unchanged)"
                )
                new_embeddings = self.model.encode(
                    texts[stored_count:],
                    show_progress_bar=True,
                    batch_size=512,
                    convert_to_numpy=True,
                )
                # Merge new embeddings into existing .npy cache
                npy_path, embed_meta_path = self._embed_cache_paths(corpus_name)
                if os.path.exists(npy_path):
                    existing_emb = np.load(npy_path)
                    merged = np.vstack([existing_emb, new_embeddings])
                else:
                    merged = new_embeddings
                np.save(npy_path, merged)
                with open(embed_meta_path, "w") as f:
                    json.dump({"count": current_count}, f)

                chroma_chunk = 5_000
                for start in range(0, new_count, chroma_chunk):
                    end = min(start + chroma_chunk, new_count)
                    abs_start = stored_count + start
                    abs_end = stored_count + end
                    self._chroma_add_batch(
                        collection, corpus_name,
                        abs_start, abs_end,
                        new_embeddings, texts, concepts_data,
                    )
                    self._chroma_write_meta(corpus_name, abs_end)
                    logger.info(f"  Chroma: inserted {abs_end:,}/{current_count:,}")
                self.chroma_collections[corpus_name] = collection
                self._chroma_write_meta(corpus_name, current_count)
                logger.info(f"Chroma index updated: {current_count:,} total embeddings")
                return

            # ── Case 4: build from scratch ────────────────────────────────────
            # Collection may exist with no matching meta (e.g. after a crash
            # before meta was written).  Delete it so create_collection works.
            if collection_exists:
                logger.info(
                    f"Chroma: stale collection '{corpus_name}' found with no valid "
                    f"meta — deleting before rebuild"
                )
                self.chroma_client.delete_collection(name=corpus_name)

            logger.info(f"Building Chroma index '{corpus_name}' ({current_count:,} docs)...")
            collection = self.chroma_client.create_collection(
                name=corpus_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Encoding {current_count:,} documents (batch_size=512)...")
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=512,
                convert_to_numpy=True,
            )

            # ── Save .npy immediately after encoding ──────────────────────────
            # Encodings take hours for large corpora.  Persist them now so they
            # survive a kill/crash during the Chroma insertion loop below, and
            # so switching to USE_CHROMA=false later requires no re-encoding.
            npy_path, embed_meta_path = self._embed_cache_paths(corpus_name)
            np.save(npy_path, embeddings)
            with open(embed_meta_path, "w") as f:
                json.dump({"count": current_count}, f)
            logger.info(f"Embedding cache saved to {npy_path} ({current_count:,} vectors)")

            chroma_chunk = 5_000
            for start in range(0, current_count, chroma_chunk):
                end = min(start + chroma_chunk, current_count)
                self._chroma_add_batch(
                    collection, corpus_name, start, end, embeddings, texts, concepts_data
                )
                # Checkpoint: update stored count after each successful batch so
                # a mid-insertion restart can resume via the additions-only path.
                self._chroma_write_meta(corpus_name, end)
                logger.info(f"  Chroma: inserted {end:,}/{current_count:,}")

            self.chroma_collections[corpus_name] = collection
            self._chroma_write_meta(corpus_name, current_count)
            logger.info(f"Chroma index built: {current_count:,} embeddings stored")

        except Exception as e:
            logger.error(f"Chroma indexing failed: {e}, falling back to memory")
            self.use_chroma = False
            self._build_memory_index(corpus_name, texts, concepts_data)

    def build_index_streaming(self, corpus_name: str, db, total_count: int) -> None:
        """Build FAISS index from a DB object in streaming chunks to avoid OOM.

        Instead of loading all concept texts into RAM at once, this method:
        1. Streams concepts from ``db.get_all_concepts_for_indexing()`` in small
           batches, encodes each batch, and writes vectors directly to a
           ``np.memmap`` file on disk (peak encoding RAM ≈ one batch, ~few MB).
        2. After encoding, builds a FAISS ``IndexFlatIP`` by adding vectors in
           chunks of 200 k from the memmap (avoids a second 6.5 GB allocation;
           FAISS still accumulates ~6.5 GB in its own internal buffer, but no
           extra copy is held).

        Only supported when ``self.use_faiss`` is True.  Falls back to a
        warning if the model or FAISS is unavailable.
        """
        if self.model is None:
            logger.warning("Model not available, skipping embedding index")
            return
        if not self.use_faiss:
            logger.warning(
                "build_index_streaming requires FAISS backend; "
                "set VECTOR_BACKEND=faiss or pass --backend faiss"
            )
            return

        import faiss

        index_path, meta_path = self._faiss_paths(corpus_name)
        npy_path, npy_meta_path = self._embed_cache_paths(corpus_name)
        dim = self.model.get_sentence_embedding_dimension()

        # ── Phase 1: encode to memmap (skip if npy already complete) ─────────
        npy_ok = False
        if os.path.exists(npy_meta_path) and os.path.exists(npy_path):
            try:
                with open(npy_meta_path) as f:
                    if json.load(f).get("count") == total_count:
                        npy_ok = True
            except Exception:
                pass

        if not npy_ok:
            try:
                import torch
                _dev = str(self.model.device)
                on_mps  = torch.backends.mps.is_available() and _dev == "mps"
                on_cuda = torch.cuda.is_available()  and _dev.startswith("cuda")
            except Exception:
                on_mps = on_cuda = False

            # Detect GPU count and VRAM *before* starting the pool — spawning
            # worker processes can disturb the main-process CUDA context and
            # cause get_device_properties() to throw after pool start.
            n_gpus = torch.cuda.device_count() if on_cuda else 0

            if on_mps:
                # MPS: Metal GPU allocator OOMs fast — keep small
                encode_batch = 32
            elif on_cuda:
                # BERT attention is O(batch × heads × seq_len²).
                # For bge-small (12 heads, max 512 tokens): ~12 MB per sample peak.
                # Use 30% of total VRAM as budget per GPU, scaled by GPU count.
                # OOM auto-retry below will halve further if needed.
                # Query device 0 directly — self.model.device may be stale after pool spawn.
                try:
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    encode_batch = min(4096 * max(1, n_gpus), max(256, int(vram_gb * 25 * max(1, n_gpus))))
                    logger.info(f"CUDA VRAM: {vram_gb:.1f} GB × {n_gpus} GPU(s) → encode_batch={encode_batch}")
                except Exception:
                    encode_batch = 512 * max(1, n_gpus)

            # Multi-GPU: start process pool *after* encode_batch is calculated
            multi_gpu_pool = None
            if n_gpus > 1:
                target_devices = [f"cuda:{i}" for i in range(n_gpus)]
                multi_gpu_pool = self.model.start_multi_process_pool(target_devices=target_devices)
                logger.info(f"Multi-GPU encoding pool started: {n_gpus} GPUs ({', '.join(target_devices)})")
            else:
                # CPU: scale with available system RAM
                # ~8 samples per GB is conservative for CPU inference
                try:
                    import psutil
                    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
                    encode_batch = min(2048, max(64, int(ram_gb * 8)))
                    logger.info(f"System RAM: {ram_gb:.1f} GB → encode_batch={encode_batch}")
                except Exception:
                    encode_batch = 256
            # Resume from partial .npy if a previous run was killed mid-encoding.
            # The memmap file may exist but the meta was not written yet.
            partial_offset = 0
            if os.path.exists(npy_path):
                try:
                    existing = np.memmap(npy_path, dtype="float32", mode="r", shape=(total_count, dim))
                    # Find last non-zero row as a heuristic for resume point
                    # (safe because zero vectors are extremely rare in real embeddings)
                    nonzero_rows = np.any(existing != 0, axis=1)
                    partial_offset = int(np.argmin(nonzero_rows)) if not nonzero_rows.all() else total_count
                    del existing
                    if 0 < partial_offset < total_count:
                        logger.info(f"Resuming encoding from offset {partial_offset:,} / {total_count:,}")
                except Exception:
                    partial_offset = 0

            _device_label = "MPS" if on_mps else (f"CUDA×{n_gpus}" if on_cuda and n_gpus > 1 else ("CUDA" if on_cuda else "CPU"))
            logger.info(
                f"Streaming encoding {total_count:,} docs → memmap "
                f"(dim={dim}, db_batch=500, encode_batch={encode_batch} [buffered], device={_device_label})..."
            )
            import time, sys
            try:
                from tqdm import tqdm as _tqdm
                _has_tqdm = True
            except ImportError:
                _has_tqdm = False

            _interactive = sys.stdout.isatty()
            _use_tqdm    = _has_tqdm and _interactive  # tqdm only for interactive sessions

            mm = np.memmap(npy_path, dtype="float32", mode="w+" if partial_offset == 0 else "r+", shape=(total_count, dim))
            offset = 0
            text_buf: list = []   # accumulate texts across DB batches up to encode_batch
            offs_buf: list = []   # corresponding (start, end) positions in memmap
            last_logged_milestone = partial_offset // 100_000  # track last 100k boundary logged
            _t_start = time.time()

            def _flush_buf():
                nonlocal encode_batch
                if not text_buf:
                    return
                # Multi-GPU: distribute batch across all GPUs via process pool
                if multi_gpu_pool is not None:
                    per_gpu = max(64, encode_batch // n_gpus)
                    vecs = self.model.encode_multi_process(
                        text_buf, multi_gpu_pool, batch_size=per_gpu
                    )
                    vecs = np.array(vecs, dtype="float32")
                else:
                    # Single device: auto-retry with halved batch on CUDA OOM
                    while True:
                        try:
                            vecs = self.model.encode(
                                text_buf, batch_size=encode_batch, show_progress_bar=False, convert_to_numpy=True
                            )
                            break
                        except RuntimeError as e:
                            if on_cuda and "out of memory" in str(e).lower() and encode_batch > 64:
                                try: torch.cuda.empty_cache()
                                except Exception: pass
                                encode_batch = max(64, encode_batch // 2)
                                logger.warning(f"CUDA OOM — reducing encode_batch to {encode_batch} and retrying")
                            else:
                                raise
                buf_offset = 0
                for (s, e) in offs_buf:
                    mm[s:e] = vecs[buf_offset : buf_offset + (e - s)].astype("float32")
                    buf_offset += e - s
                text_buf.clear()
                offs_buf.clear()
                if on_mps:
                    try: torch.mps.empty_cache()
                    except Exception: pass
                elif on_cuda and multi_gpu_pool is None:
                    try: torch.cuda.empty_cache()
                    except Exception: pass

            _pbar = _tqdm(
                total=total_count,
                initial=partial_offset,
                desc="Encoding",
                unit="doc",
                unit_scale=True,
                dynamic_ncols=True,
                mininterval=5.0,
            ) if _use_tqdm else None

            for db_batch in db.get_all_concepts_for_indexing(batch_size=500):
                n = len(db_batch)
                # Skip already-encoded batches when resuming
                if offset + n <= partial_offset:
                    offset += n
                    if _pbar:
                        _pbar.update(n)
                    continue
                text_buf.extend(build_rich_concept_text(c) for c in db_batch)
                offs_buf.append((offset, offset + n))
                offset += n
                if _pbar:
                    _pbar.update(n)
                # Encode once we have a full GPU batch
                if len(text_buf) >= encode_batch:
                    _flush_buf()
                    if not _use_tqdm:
                        # fallback when tqdm not installed
                        current_milestone = offset // 100_000
                        if current_milestone > last_logged_milestone:
                            last_logged_milestone = current_milestone
                            mm.flush()
                            elapsed   = time.time() - _t_start
                            rate      = max(1, offset - partial_offset) / elapsed
                            remaining = (total_count - offset) / rate
                            h, m = divmod(int(remaining), 3600)
                            m, s = divmod(m, 60)
                            logger.info(
                                f"  Encoded {offset:,} / {total_count:,}"
                                f"  ({offset/total_count*100:.1f}%)"
                                f"  {rate:,.0f} docs/s"
                                f"  ETA {h}h{m:02d}m{s:02d}s"
                            )
            _flush_buf()  # encode any remaining texts
            if _pbar:
                _pbar.close()
            if multi_gpu_pool is not None:
                self.model.stop_multi_process_pool(multi_gpu_pool)
                logger.info("Multi-GPU pool stopped")
            mm.flush()
            del mm
            with open(npy_meta_path, "w") as f:
                json.dump({"count": total_count}, f)
            logger.info(f"Embeddings written to {npy_path} ({total_count:,} vectors, dim={dim})")
        else:
            logger.info(f"Reusing .npy cache for FAISS build ({total_count:,} vectors)")

        # ── Phase 2: build FAISS index in chunks from memmap ─────────────────
        CHUNK = 200_000
        logger.info(
            f"Building FAISS IndexFlatIP from memmap "
            f"(chunks of {CHUNK:,}, total={total_count:,})..."
        )
        index = faiss.IndexFlatIP(dim)
        mm_r = np.memmap(npy_path, dtype="float32", mode="r", shape=(total_count, dim))
        for start in range(0, total_count, CHUNK):
            end = min(start + CHUNK, total_count)
            chunk = np.array(mm_r[start:end], dtype="float32")
            faiss.normalize_L2(chunk)
            index.add(chunk)
            del chunk
            logger.info(f"  FAISS: added {end:,} / {total_count:,}")
        del mm_r

        faiss.write_index(index, index_path)
        with open(meta_path, "w") as f:
            json.dump({"count": total_count}, f)
        self.faiss_indexes[corpus_name] = index
        logger.info(f"FAISS index built and saved: {total_count:,} vectors, dim={dim}")

    def load_cached_index(self, corpus_name: str, count: int) -> bool:
        """Load dense index from disk cache if the stored count matches *count*.

        Returns True if the cache was loaded successfully, False otherwise.
        Priority: FAISS > Chroma > numpy .npy
        """
        try:
            # ── FAISS ────────────────────────────────────────────────────────
            if self.use_faiss:
                import faiss
                index_path, meta_path = self._faiss_paths(corpus_name)
                if os.path.exists(meta_path) and os.path.exists(index_path):
                    with open(meta_path) as f:
                        if json.load(f).get("count") == count:
                            self.faiss_indexes[corpus_name] = faiss.read_index(
                                index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                            )
                            logger.info(f"FAISS index '{corpus_name}' mmap-loaded from cache ({count:,})")
                            return True
                # FAISS index missing/stale — fall through to .npy below
                logger.info(f"FAISS index not ready for '{corpus_name}'; checking .npy fallback...")
                npy_path, meta_path2 = self._embed_cache_paths(corpus_name)
                if os.path.exists(meta_path2) and os.path.exists(npy_path):
                    with open(meta_path2) as f:
                        if json.load(f).get("count") == count:
                            # Build FAISS from existing .npy (seconds)
                            logger.info(f"Building FAISS index from .npy cache ({count:,} vectors)...")
                            embeddings = np.load(npy_path)
                            emb_f32 = embeddings.astype(np.float32)
                            faiss.normalize_L2(emb_f32)
                            index = faiss.IndexFlatIP(emb_f32.shape[1])
                            index.add(emb_f32)
                            idx_path, idx_meta = self._faiss_paths(corpus_name)
                            faiss.write_index(index, idx_path)
                            with open(idx_meta, "w") as f:
                                json.dump({"count": count}, f)
                            self.faiss_indexes[corpus_name] = index
                            logger.info(f"FAISS index built from .npy and cached ({count:,})")
                            return True
                return False

            if self.use_chroma:
                meta = self._chroma_read_meta(corpus_name)
                if meta.get("count") == count:
                    collection = self.chroma_client.get_collection(name=corpus_name)
                    self.chroma_collections[corpus_name] = collection
                    logger.info(f"Chroma collection '{corpus_name}' loaded from cache ({count:,} docs)")
                    return True
                # Chroma not complete — fall back to .npy if the embedding cache
                # is valid (saved before Chroma insertion in _build_chroma_index).
                logger.info(
                    f"Chroma meta mismatch for '{corpus_name}' "
                    f"(stored={meta.get('count', 0):,}, expected={count:,}); "
                    f"checking .npy fallback..."
                )

            # numpy path (also used as Chroma fallback)
            npy_path, meta_path = self._embed_cache_paths(corpus_name)
            if not os.path.exists(meta_path) or not os.path.exists(npy_path):
                return False
            with open(meta_path) as f:
                if json.load(f).get("count") != count:
                    return False
            self.corpus_embeddings[corpus_name] = np.load(npy_path)
            if self.use_chroma:
                # Disable Chroma for this session — insertion never completed
                self.use_chroma = False
                logger.info(
                    f"Loaded {count:,} embeddings from .npy fallback "
                    f"(Chroma insertion was incomplete — using in-memory search)"
                )
            else:
                logger.info(f"Embeddings for '{corpus_name}' loaded from cache ({count:,} docs)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load dense cache for '{corpus_name}': {e}")
            return False

    def retrieve(
        self,
        query: str,
        corpus_name: str = "default",
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents using dense search
        
        Args:
            query: Query text (will use rich embedding)
            corpus_name: Name of corpus to search
            k: Number of results to return
            
        Returns:
            List of (doc_index, score) tuples
        """
        no_index = (
            corpus_name not in self.faiss_indexes
            and corpus_name not in self.chroma_collections
            and corpus_name not in self.corpus_embeddings
        )
        if self.model is None or no_index:
            logger.warning(f"Dense index not available for '{corpus_name}'")
            return []

        try:
            if query in self._query_embed_cache:
                query_embedding = self._query_embed_cache[query]
            else:
                with self._encode_lock:
                    # Re-check cache after acquiring lock (another thread may have encoded
                    # the same query while we were waiting)
                    if query in self._query_embed_cache:
                        query_embedding = self._query_embed_cache[query]
                    else:
                        query_embedding = np.asarray(
                            self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
                        )
                        if len(self._query_embed_cache) >= self._query_cache_max:
                            self._query_embed_cache.pop(next(iter(self._query_embed_cache)))
                        self._query_embed_cache[query] = query_embedding

            if self.use_faiss and corpus_name in self.faiss_indexes:
                return self._retrieve_faiss(query_embedding, corpus_name, k)
            elif self.use_chroma and corpus_name in self.chroma_collections:
                return self._retrieve_chroma(query_embedding, corpus_name, k)
            else:
                return self._retrieve_memory(query_embedding, corpus_name, k)
                
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []

    def _retrieve_memory(
        self,
        query_embedding: np.ndarray,
        corpus_name: str,
        k: int,
    ) -> List[Tuple[int, float]]:
        """Retrieve from in-memory index"""
        corpus_embeddings = self.corpus_embeddings[corpus_name]
        
        # Cosine similarity
        similarities = np.dot(corpus_embeddings, query_embedding) / (
            np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Top-k indices and scores
        top_k_indices = np.argsort(-similarities)[:k]
        top_k_scores = similarities[top_k_indices]
        
        return list(zip(top_k_indices, top_k_scores))

    def _retrieve_faiss(
        self,
        query_embedding: np.ndarray,
        corpus_name: str,
        k: int,
    ) -> List[Tuple[int, float]]:
        """Exact cosine search via FAISS IndexFlatIP (~50ms for 4M vectors)."""
        import faiss
        index = self.faiss_indexes[corpus_name]
        # embedding already L2-normalised by encode(normalize_embeddings=True)
        q = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = index.search(q, k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def _retrieve_chroma(
        self,
        query_embedding,
        corpus_name: str,
        k: int,
    ) -> List[Tuple[int, float]]:
        """Retrieve from Chroma DB"""
        try:
            collection = self.chroma_collections[corpus_name]
            
            # Query Chroma (with distance metric conversion)
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
            )
            
            if not results or not results["ids"]:
                return []
            
            # Parse results
            doc_ids = results["ids"][0]
            distances = results["distances"][0]
            
            # Convert distances to similarities (Chroma returns distances)
            # For cosine distance: similarity = 1 - distance
            similarities = [1.0 - d for d in distances]
            
            # Extract index from doc_id
            indices = []
            for doc_id in doc_ids:
                try:
                    idx = int(doc_id.split("_")[-1])
                    indices.append(idx)
                except:
                    continue
            
            return list(zip(indices, similarities))
            
        except Exception as e:
            logger.error(f"Chroma retrieval failed: {e}")
            return []

    def get_concept_for_index(self, corpus_name: str, index: int) -> Optional[Dict[str, Any]]:
        """Get concept data by index"""
        if corpus_name in self.corpus_metadata:
            metadata = self.corpus_metadata[corpus_name]
            if 0 <= index < len(metadata):
                return metadata[index]
        return None


class HybridRetriever:
    """Combines BM25 and dense retrieval with configurable weighting"""

    def __init__(
        self,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        bm25_model: Optional[BM25Retriever] = None,
        dense_model: Optional[DenseRetriever] = None,
    ):
        """
        Initialize hybrid retriever
        
        Args:
            bm25_weight: Weight for BM25 score
            dense_weight: Weight for dense score
            bm25_model: BM25 retriever instance
            dense_model: Dense retriever instance
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25_model = bm25_model or BM25Retriever()
        self.dense_model = dense_model or DenseRetriever()
        logger.info(
            f"HybridRetriever initialized: bm25_weight={bm25_weight}, dense_weight={dense_weight}"
        )

    def build_index(self, corpus_name: str, concepts_data: List[Dict[str, Any]]):
        """
        Build both BM25 and dense indexes

        Args:
            corpus_name: Name of the corpus
            concepts_data: List of concept dicts with full information:
                - preferred_label (required)
                - class_uri
                - ontology_id
                - definition (optional)
                - synonyms (optional, list)
                - labels (optional, list)
                - parent_labels (optional, list)
        """
        # Extract rich texts for BM25 (label + synonyms + alt labels)
        texts = [build_rich_concept_text(c) for c in concepts_data]

        logger.info(f"Building hybrid indexes for {corpus_name} (BM25 + dense in parallel)...")
        # BM25 is CPU/numpy-bound; dense encoding is GPU/PyTorch-bound.
        # Both release the GIL for their heavy inner loops, so running them
        # concurrently gives a real wall-clock speedup.
        with ThreadPoolExecutor(max_workers=2) as pool:
            bm25_fut  = pool.submit(self.bm25_model.build_index,  corpus_name, texts)
            dense_fut = pool.submit(self.dense_model.build_index, corpus_name, concepts_data)
            done, _ = wait([bm25_fut, dense_fut], return_when=FIRST_EXCEPTION)
            for f in done:
                f.result()  # re-raise any exception immediately
            # Wait for the remaining future if the first one succeeded
            for f in [bm25_fut, dense_fut]:
                f.result()

    def load_indexes(self, corpus_name: str, count: int) -> bool:
        """Load all indexes from disk cache without touching the database.

        Returns True only if **both** BM25 and dense caches loaded successfully.
        """
        bm25_ok = self.bm25_model.load_cached_index(corpus_name, count)
        dense_ok = self.dense_model.load_cached_index(corpus_name, count)
        if bm25_ok and dense_ok:
            logger.info(f"All indexes loaded from cache for '{corpus_name}' ({count:,} docs)")
        return bm25_ok and dense_ok

    def retrieve(
        self,
        query: str,
        corpus_name: str = "default",
        k: int = 10,
        concepts_map: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[RetrievalCandidate]:
        """
        Hybrid retrieval: combine BM25 and dense results
        
        Args:
            query: Query text
            corpus_name: Name of corpus
            k: Number of results to return
            concepts_map: Mapping of doc_index to concept metadata
            
        Returns:
            List of RetrievalCandidate objects sorted by combined score
        """
        t0 = time.perf_counter()
        bm25_results = self.bm25_model.retrieve(query, corpus_name, k * 2)
        t1 = time.perf_counter()
        dense_results = self.dense_model.retrieve(query, corpus_name, k * 2)
        t2 = time.perf_counter()
        logger.info(
            f"[retrieve] BM25={1000*(t1-t0):.1f}ms  Dense={1000*(t2-t1):.1f}ms  "
            f"candidates={len(bm25_results)}+{len(dense_results)}"
        )

        # Combine results
        combined = {}
        
        # Add BM25 scores
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results) or 1.0
            for doc_idx, score in bm25_results:
                combined[doc_idx] = {
                    "bm25_score": float(score) / max_bm25_score,
                    "embedding_score": 0.0,
                }

        # Add dense scores
        if dense_results:
            for doc_idx, score in dense_results:
                normalized_score = (float(score) + 1.0) / 2.0  # Normalize [-1, 1] to [0, 1]
                if doc_idx not in combined:
                    combined[doc_idx] = {"bm25_score": 0.0, "embedding_score": 0.0}
                combined[doc_idx]["embedding_score"] = normalized_score

        # Calculate combined scores
        candidates = []
        for doc_idx, scores in combined.items():
            combined_score = (
                self.bm25_weight * scores["bm25_score"] +
                self.dense_weight * scores["embedding_score"]
            )
            
            if concepts_map and doc_idx in concepts_map:
                concept = concepts_map[doc_idx]
                candidate = RetrievalCandidate(
                    class_uri=concept.get("class_uri", ""),
                    preferred_label=concept.get("preferred_label", ""),
                    ontology_id=concept.get("ontology_id", ""),
                    bm25_score=scores["bm25_score"],
                    embedding_score=scores["embedding_score"],
                    combined_score=combined_score,
                )
                candidates.append(candidate)

        # Sort by combined score
        candidates.sort(key=lambda x: x.combined_score, reverse=True)
        t3 = time.perf_counter()
        logger.info(
            f"[retrieve] merge+sort={1000*(t3-t2):.1f}ms  "
            f"top-{len(candidates[:k])} ready  total={1000*(t3-t0):.1f}ms"
        )
        return candidates[:k]
