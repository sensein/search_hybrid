#!/usr/bin/env python3
"""
Standalone index builder — run this BEFORE starting the API server to
pre-generate BM25 and dense embeddings offline (useful for pushing cached
indexes to Hugging Face Hub or sharing between machines).

Usage:
    python build_index.py
    python build_index.py --backend faiss       # default
    python build_index.py --backend numpy       # no faiss-cpu required
    python build_index.py --backend chroma      # ChromaDB (slow for >1M)
    python build_index.py --force               # rebuild even if caches exist

After completion, start the server normally — it will hit Tier 1
(all caches valid) and serve immediately without any indexing delay.

Pushing to Hugging Face Hub:
    huggingface-cli login
    python build_index.py
    python build_index.py --push-to-hub your-org/ontology-indexes

Pulling on another machine:
    python build_index.py --pull-from-hub your-org/ontology-indexes
"""

import os
import sys
import json
import pickle
import logging
import argparse
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def push_to_hub(repo_id: str, index_cache_dir: str, embed_cache_dir: str) -> None:
    """Upload all index caches to a Hugging Face Hub dataset repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    folders = [
        (index_cache_dir, "index_cache"),
        (embed_cache_dir, "embed_cache"),
    ]
    for local_dir, remote_prefix in folders:
        if not os.path.isdir(local_dir):
            logger.warning(f"Skipping {local_dir} (does not exist)")
            continue
        logger.info(f"Uploading {local_dir} → {repo_id}/{remote_prefix}/")
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo=remote_prefix,
            repo_id=repo_id,
            repo_type="dataset",
        )

    logger.info(f"✓ Indexes pushed to https://huggingface.co/datasets/{repo_id}")
    logger.info(
        f"Pull on another machine with:\n"
        f"  python build_index.py --pull-from-hub {repo_id}"
    )


def pull_from_hub(repo_id: str, index_cache_dir: str, embed_cache_dir: str) -> None:
    """Download index caches from a Hugging Face Hub dataset repository."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed: pip install huggingface_hub")
        sys.exit(1)

    import shutil

    logger.info(f"Downloading indexes from {repo_id}...")
    local_snapshot = snapshot_download(repo_id=repo_id, repo_type="dataset")

    mappings = [
        (os.path.join(local_snapshot, "index_cache"), index_cache_dir),
        (os.path.join(local_snapshot, "embed_cache"),  embed_cache_dir),
    ]
    for src, dst in mappings:
        if not os.path.isdir(src):
            logger.warning(f"Remote folder not found: {src}")
            continue
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        logger.info(f"  {src} → {dst}")

    logger.info("✓ Indexes downloaded. Start the server — it will load from cache (Tier 1).")


def _meta_count(path: str) -> int:
    """Read 'count' from a JSON meta file; return 0 on any error."""
    try:
        with open(path) as f:
            return json.load(f).get("count", 0)
    except Exception:
        return 0


def build(args: argparse.Namespace) -> None:
    from db_layer import OntologyDB
    from retrieval import BM25Retriever, DenseRetriever, build_rich_concept_text

    index_cache_dir = os.getenv("INDEX_CACHE_DIR", ".cache/ontology_indexes")
    embed_cache_dir = os.getenv("EMBED_CACHE_DIR", ".cache/embed_indexes")
    bm25_cache_dir  = os.getenv("BM25_CACHE_DIR",  ".cache/bm25_indexes")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_path     = os.getenv("CHROMA_DB_PATH",  ".cache/chroma_db")

    use_faiss  = args.backend == "faiss"
    use_chroma = args.backend == "chroma"

    logger.info(f"Connecting to database: {args.db}")
    db = OntologyDB(args.db)

    os.makedirs(index_cache_dir, exist_ok=True)
    concepts_cache_path = os.path.join(index_cache_dir, "concepts_cache.pkl")
    concepts_meta_path  = os.path.join(index_cache_dir, "concepts_cache_meta.json")

    db_count = db.get_stats()["num_classes"]
    logger.info(f"Database contains {db_count:,} concepts")

    # ── Lightweight file-only cache checks (no index loading into RAM) ────────
    concepts_ok = (
        not args.force
        and _meta_count(concepts_meta_path) == db_count
        and os.path.exists(concepts_cache_path)
    )

    bm25_index_dir  = os.path.join(bm25_cache_dir, "main")
    bm25_meta_path  = os.path.join(bm25_cache_dir, "main_meta.json")
    bm25_cached = (
        not args.force
        and os.path.isdir(bm25_index_dir)
        and _meta_count(bm25_meta_path) == db_count
    )

    faiss_index_path = os.path.join(embed_cache_dir, "main_faiss.index")
    faiss_meta_path  = os.path.join(embed_cache_dir, "main_faiss_meta.json")
    dense_cached = (
        not args.force
        and use_faiss
        and os.path.exists(faiss_index_path)
        and _meta_count(faiss_meta_path) == db_count
    )

    logger.info(
        f"Cache status — concepts: {'OK' if concepts_ok else 'MISSING'}, "
        f"BM25: {'OK' if bm25_cached else 'MISSING'}, "
        f"dense/FAISS: {'OK' if dense_cached else 'MISSING'}"
    )

    if concepts_ok and bm25_cached and dense_cached:
        logger.info("✓ All indexes already up-to-date — nothing to do")
        logger.info("  Use --force to rebuild anyway.")
        return

    # ── 1. Concepts cache (fast no-JOIN scan, very low RAM) ───────────────────
    if not concepts_ok:
        logger.info("Building concepts cache (fast scan, no joins)...")
        minimal = db.get_all_minimal_concepts()
        with open(concepts_cache_path, "wb") as f:
            pickle.dump(minimal, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(concepts_meta_path, "w") as f:
            json.dump({"count": len(minimal)}, f)
        logger.info(f"Concepts cache saved ({len(minimal):,} entries)")
        del minimal
    else:
        logger.info("Concepts cache already valid — skipping")

    # ── 2. BM25 index (needs all texts in RAM for bm25s.BM25.index()) ────────
    if not bm25_cached:
        bm25_model = BM25Retriever(cache_dir=bm25_cache_dir)
        logger.info("Building BM25 index (streaming rich-text scan)...")
        all_texts: List[str] = []
        processed = 0
        for batch in db.get_all_concepts_for_indexing(batch_size=500):
            all_texts.extend(build_rich_concept_text(c) for c in batch)
            processed += len(batch)
            if processed % 200_000 == 0:
                logger.info(f"  BM25 texts: {processed:,} / {db_count:,}")
        logger.info(f"BM25 corpus ready ({len(all_texts):,} texts)")
        bm25_model.build_index("main", all_texts)
        del all_texts
    else:
        logger.info("BM25 index already valid — skipping")

    # ── 3. Dense / FAISS index (streaming chunked encoding — no OOM) ─────────
    if not dense_cached:
        dense_model = DenseRetriever(
            model_name=embedding_model,
            use_chroma=use_chroma,
            chroma_path=chroma_path,
            embed_cache_dir=embed_cache_dir,
            use_faiss=use_faiss,
        )
        if use_faiss:
            logger.info("Building FAISS index via streaming chunked encoding (low RAM)...")
            dense_model.build_index_streaming("main", db, db_count)
        else:
            # Non-FAISS backends: fall back to loading all texts (chroma/numpy)
            logger.info("Building dense index (non-FAISS: loading all concept texts)...")
            all_concepts: List[Dict] = []
            for batch in db.get_all_concepts_for_indexing(batch_size=500):
                all_concepts.extend(batch)
            dense_model.build_index("main", all_concepts)
            del all_concepts
    else:
        logger.info("Dense/FAISS index already valid — skipping")

    logger.info("✓ All indexes built successfully")
    logger.info(f"  BM25 index  → {bm25_cache_dir}/main/")
    logger.info(f"  Embeddings  → {embed_cache_dir}/main_embeddings.npy")
    if use_faiss:
        logger.info(f"  FAISS index → {embed_cache_dir}/main_faiss.index")

    return index_cache_dir, embed_cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-build BM25 and dense indexes for the Ontology Mapping API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backend",
        choices=["faiss", "numpy", "chroma"],
        default=os.getenv("VECTOR_BACKEND", "faiss"),
        help="Vector backend to use (default: $VECTOR_BACKEND or faiss)",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("DATABASE_PATH", "bioportal.db"),
        help="Path to SQLite database (default: $DATABASE_PATH or bioportal.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if caches already exist",
    )
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="Push built indexes to Hugging Face Hub (e.g. your-org/ontology-indexes)",
    )
    parser.add_argument(
        "--pull-from-hub",
        metavar="REPO_ID",
        help="Download pre-built indexes from Hugging Face Hub instead of building",
    )
    args = parser.parse_args()

    index_cache_dir = os.getenv("INDEX_CACHE_DIR", ".cache/ontology_indexes")
    embed_cache_dir = os.getenv("EMBED_CACHE_DIR", ".cache/embed_indexes")

    if args.pull_from_hub:
        pull_from_hub(args.pull_from_hub, index_cache_dir, embed_cache_dir)
        return

    build(args)

    if args.push_to_hub:
        push_to_hub(args.push_to_hub, index_cache_dir, embed_cache_dir)


if __name__ == "__main__":
    main()
