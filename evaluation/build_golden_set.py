#!/usr/bin/env python3
"""
Build (or refresh) evaluation/golden_set.json from the live bioportal.db.

Generates a stratified golden set with:
  - Exact-match queries          → tested via /map/concept
  - Definition-as-context        → tested via /map/search
  - Abbreviation/synonym queries → tested via /map/concept or /map/search
  - Disambiguation (multi-sense) → tested via /map/search
  - Batch groups                 → tested via /map/batch (groups of 2–5)

Each entry carries an `ontology_id` field for dataset-level statistics.
The `meta` block includes a per-ontology breakdown of the single entries.

Usage:
    python evaluation/build_golden_set.py                          # default: ~1000 entries
    python evaluation/build_golden_set.py --size 500               # fewer entries
    python evaluation/build_golden_set.py --db /path/to/db         # custom DB path
    python evaluation/build_golden_set.py --out evaluation/my_set.json  # custom output path
"""

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).parent
_DEFAULT_DB  = os.getenv("DATABASE_PATH", "bioportal.db")
_DEFAULT_OUT = str(_THIS_DIR / "golden_set.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _is_valid_query(s: Any) -> bool:
    """Reject non-ASCII text and queries containing special symbols."""
    s = str(s)
    return s.isascii() and not any(c in s for c in ('+', '/', '\\', '|', '#', '@', '%'))


def _clean(s: Optional[str], maxlen: int = 200) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    return s[:maxlen] if s else None


def _syns(cur, class_id: int, limit: int = 5) -> List[str]:
    cur.execute("SELECT synonym FROM synonyms WHERE class_id = ? LIMIT ?", (class_id, limit))
    return [r[0] for r in cur.fetchall() if r[0] and r[0].strip()]


def _labels(cur, class_id: int, limit: int = 5) -> List[str]:
    cur.execute("SELECT label FROM labels WHERE class_id = ? LIMIT ?", (class_id, limit))
    return [r[0] for r in cur.fetchall() if r[0] and r[0].strip()]


def _make_entry(
    query: Any,
    acceptable: List[str],
    context: Optional[str],
    endpoint: str,
    note: str,
    ontology_id: Any,
    context_source: Optional[str] = None,
) -> dict:
    return {
        "query": query,
        "context": context,
        "context_source": context_source,   # "definition" | "synonyms" | null
        "acceptable_labels": [a.lower() for a in acceptable],
        "endpoint": endpoint,   # "concept" | "search" | "batch"
        "note": note,
        "ontology_id": ontology_id,
    }


# ── sampling strategies ───────────────────────────────────────────────────────

def sample_exact_match_no_context(cur, rng: random.Random, n: int) -> List[Any]:
    """
    Real concept preferred labels — no context.
    Tested via /map/concept.
    """
    prefer_onts = (
        'MONDO', 'HP', 'DOID', 'NCIT', 'MEDDRA', 'CTCAE', 'COSTART', 'HFO',
        'UPHENO', 'MP', 'RCD', 'SNOMEDCT', 'MESH', 'OMIM', 'ORDO',
        'GO', 'UBERON', 'CL', 'BTO', 'SO',
    )
    ph = ','.join('?' * len(prefer_onts))
    cur.execute(f"""
        SELECT c.id, c.preferred_label, c.ontology_id
        FROM classes c
        WHERE c.ontology_id IN ({ph})
          AND c.preferred_label IS NOT NULL
          AND length(c.preferred_label) BETWEEN 4 AND 60
          AND c.preferred_label NOT GLOB '*[0-9][0-9][0-9]*'
          AND c.preferred_label NOT LIKE '%:%'
        ORDER BY RANDOM()
        LIMIT ?
    """, list(prefer_onts) + [n * 3])
    rows = cur.fetchall()
    seen = set()
    entries = []
    for row in rows:
        label = row['preferred_label']
        if not _is_valid_query(label):
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append(_make_entry(
            query=label,
            acceptable=[label],
            context=None,
            endpoint="concept",
            note="exact_match_no_context",
            ontology_id=row['ontology_id'],
            context_source=None,
        ))
        if len(entries) >= n:
            break
    return entries


def sample_definition_context(cur, rng: random.Random, n: int) -> List[Any]:
    """
    Concept labels paired with their real DB definition as context.
    Tested via /map/search.
    """
    cur.execute("""
        SELECT c.id, c.preferred_label, c.definition, c.ontology_id
        FROM classes c
        WHERE c.definition IS NOT NULL AND length(c.definition) > 50
          AND c.preferred_label IS NOT NULL
          AND length(c.preferred_label) BETWEEN 4 AND 60
          AND c.preferred_label NOT GLOB '*[0-9][0-9][0-9]*'
        ORDER BY RANDOM()
        LIMIT ?
    """, (n * 3,))
    rows = cur.fetchall()
    seen = set()
    entries = []
    for row in rows:
        label = row['preferred_label']
        if not _is_valid_query(label):
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        ctx = _clean(row['definition'], 200)
        entries.append(_make_entry(
            query=label,
            acceptable=[label],
            context=ctx,
            endpoint="search",
            note="exact_match_with_definition_context",
            ontology_id=row['ontology_id'],
            context_source="definition",
        ))
        if len(entries) >= n:
            break
    return entries


def sample_synonym_as_query(cur, rng: random.Random, n: int) -> List[Any]:
    """
    Use a synonym as the query; acceptable label is the preferred label.
    Includes real synonyms as context.
    Tested via /map/concept.
    """
    cur.execute("""
        SELECT c.id, c.preferred_label, c.ontology_id, s.synonym
        FROM classes c
        JOIN synonyms s ON s.class_id = c.id
        WHERE c.preferred_label IS NOT NULL
          AND length(c.preferred_label) BETWEEN 4 AND 60
          AND length(s.synonym) BETWEEN 2 AND 80
          AND s.synonym != c.preferred_label
          AND c.preferred_label NOT GLOB '*[0-9][0-9][0-9]*'
          AND s.synonym NOT GLOB '*[0-9][0-9][0-9]*'
        ORDER BY RANDOM()
        LIMIT ?
    """, (n * 3,))
    rows = cur.fetchall()
    seen = set()
    entries = []
    for row in rows:
        syn_query = row['synonym'].strip()
        preferred = row['preferred_label']
        if not _is_valid_query(syn_query) or not _is_valid_query(preferred):
            continue
        key = syn_query.lower()
        if key in seen or key == preferred.lower():
            continue
        seen.add(key)
        extra_syns = _syns(cur, row['id'], 3)
        ctx = ", ".join(s for s in extra_syns if s.lower() != syn_query.lower())[:150] or None
        entries.append(_make_entry(
            query=syn_query,
            acceptable=[preferred],
            context=ctx,
            endpoint="concept",
            note="synonym_as_query_with_context" if ctx else "synonym_as_query_without_context",
            ontology_id=row['ontology_id'],
            context_source="synonyms" if ctx else None,
        ))
        if len(entries) >= n:
            break
    return entries


def sample_alt_label_as_query(cur, rng: random.Random, n: int) -> List[Any]:
    """
    Use an alt label (labels table) as the query; acceptable = preferred label.
    Tested via /map/search with definition context.
    """
    cur.execute("""
        SELECT c.id, c.preferred_label, c.definition, c.ontology_id, l.label
        FROM classes c
        JOIN labels l ON l.class_id = c.id
        WHERE c.preferred_label IS NOT NULL
          AND length(c.preferred_label) BETWEEN 4 AND 60
          AND length(l.label) BETWEEN 4 AND 80
          AND l.label != c.preferred_label
          AND c.preferred_label NOT GLOB '*[0-9][0-9][0-9]*'
          AND l.label NOT GLOB '*[0-9][0-9][0-9]*'
        ORDER BY RANDOM()
        LIMIT ?
    """, (n * 3,))
    rows = cur.fetchall()
    seen = set()
    entries = []
    for row in rows:
        alt_query = row['label'].strip()
        preferred = row['preferred_label']
        if not _is_valid_query(alt_query) or not _is_valid_query(preferred):
            continue
        key = alt_query.lower()
        if key in seen or key == preferred.lower():
            continue
        seen.add(key)
        ctx = _clean(row['definition'], 150) or None
        entries.append(_make_entry(
            query=alt_query,
            acceptable=[preferred],
            context=ctx,
            endpoint="search",
            note="alt_label_as_query_with_definition_context",
            ontology_id=row['ontology_id'],
            context_source="definition" if ctx else None,
        ))
        if len(entries) >= n:
            break
    return entries


def sample_batch_groups(entries_pool: List[Any], rng: random.Random, n_groups: int, group_size: int = 3):
    """
    Group concept entries into batch test items.
    Each batch entry contains multiple (query, acceptable) pairs.
    Tested via /map/batch.

    Query texts within each group MUST be unique because the /map/batch API
    returns results as a dict keyed by concept text — duplicate texts in the
    same request would collide, silently discarding all but the last result.

    Includes synonym entries (which carry context) in the pool.
    Each query in the batch carries its own note and ontology_id.

    Returns (batches, used_query_texts_lower) so the caller can remove used
    entries from the single list to ensure cross-list uniqueness.
    """
    # Include all concept entries (with or without context) so synonym entries
    # carrying synonym-list context are eligible for batch too.
    # Deduplicate by query text so the pool itself is free of collisions.
    pool_seen: set = set()
    pool: list = []
    for e in entries_pool:
        if e["endpoint"] == "concept":
            key = e["query"].lower()
            if key not in pool_seen:
                pool_seen.add(key)
                pool.append(e)

    rng.shuffle(pool)

    batches: list = []
    used_query_texts: set = set()

    for chunk in _chunks(pool, group_size):
        if len(chunk) < 2:
            break
        # Final guard: ensure every query text in this chunk is unique.
        seen_in_chunk: set = set()
        deduped_chunk = []
        for e in chunk:
            key = e["query"].lower()
            if key not in seen_in_chunk:
                seen_in_chunk.add(key)
                deduped_chunk.append(e)
        if len(deduped_chunk) < 2:
            continue
        per_query_notes = ", ".join(e["note"] for e in deduped_chunk)
        batches.append({
            "queries": [
                {
                    "text": e["query"],
                    "context": e.get("context"),
                    "note": e["note"],
                    "ontology_id": e["ontology_id"],
                }
                for e in deduped_chunk
            ],
            "acceptable_per_concept": [e["acceptable_labels"] for e in deduped_chunk],
            "endpoint": "batch",
            "note": f"batch_group_size_{len(deduped_chunk)} [{per_query_notes}]",
        })
        for e in deduped_chunk:
            used_query_texts.add(e["query"].lower())
        if len(batches) >= n_groups:
            break
    return batches, used_query_texts


# ── main ──────────────────────────────────────────────────────────────────────

def build(db_path: str, out_path: str, total_size: int, seed: int) -> None:
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    rng = random.Random(seed)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Budget per group (rough proportions)
    n_exact   = int(total_size * 0.35)   # 35% — exact no-context
    n_defctx  = int(total_size * 0.25)   # 25% — definition context
    n_syn     = int(total_size * 0.20)   # 20% — synonym as query
    n_alt     = int(total_size * 0.10)   # 10% — alt-label as query
    n_batches = max(10, int(total_size * 0.10 / 3))  # ~10% in batch groups

    logger.info(f"Sampling: exact={n_exact} def-ctx={n_defctx} syn={n_syn} alt={n_alt} batches={n_batches}×3")

    exact   = sample_exact_match_no_context(cur, rng, n_exact)
    defctx  = sample_definition_context(cur, rng, n_defctx)
    syn     = sample_synonym_as_query(cur, rng, n_syn)
    alt     = sample_alt_label_as_query(cur, rng, n_alt)

    con.close()

    # Global cross-sampler deduplication keyed on (query_lower, context_lower).
    # Entries with the same text but different contexts are kept as distinct
    # test cases; entries that are fully identical are dropped.
    global_seen: set = set()
    deduped: List[Any] = []
    for entry in exact + defctx + syn + alt:
        key = (entry["query"].lower(), (entry["context"] or "").lower())
        if key not in global_seen:
            global_seen.add(key)
            deduped.append(entry)

    removed = len(exact) + len(defctx) + len(syn) + len(alt) - len(deduped)
    if removed:
        logger.info(f"Removed {removed} cross-sampler duplicate entries (same query+context)")

    # Build batch groups AFTER global dedup so the pool is already unique.
    batches = sample_batch_groups(deduped, rng, n_batches, group_size=3)[0]

    # Batch tests /map/batch; single entries test /map/concept and /map/search.
    # The same query may appear in both lists — it is testing a different endpoint,
    # not a duplicate. Total flat = single + batch queries.
    all_entries = list(deduped)
    rng.shuffle(all_entries)

    # Per-ontology breakdown of single entries (useful for dataset stats).
    ontology_counts = dict(Counter(e["ontology_id"] for e in all_entries))

    result = {
        "single": all_entries,    # entries for /map/concept and /map/search
        "batch":  batches,        # entries for /map/batch
        "meta": {
            "total_single": len(all_entries),
            "total_batch_groups": len(batches),
            "breakdown": {
                "exact_match_no_context":                      len(exact),
                "exact_match_with_definition_context":         len(defctx),
                "synonym_as_query_with_context":               sum(1 for e in syn if e["note"] == "synonym_as_query_with_context"),
                "synonym_as_query_without_context":            sum(1 for e in syn if e["note"] == "synonym_as_query_without_context"),
                "alt_label_as_query_with_definition_context":  len(alt),
                "batch_groups":                                len(batches),
            },
            "ontology_breakdown": dict(sorted(ontology_counts.items(), key=lambda x: x[1], reverse=True)),
            "db": db_path,
            "seed": seed,
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved {len(all_entries)} single + {len(batches)} batch groups → {out_path}")
    logger.info(f"Breakdown: {result['meta']['breakdown']}")
    logger.info(f"Ontologies: {ontology_counts}")


def main():
    parser = argparse.ArgumentParser(description="Build golden_set.json from bioportal.db")
    parser.add_argument("--db",   default=_DEFAULT_DB,  help=f"Source database (default: {_DEFAULT_DB})")
    parser.add_argument("--out",  default=_DEFAULT_OUT, help=f"Output path (default: {_DEFAULT_OUT})")
    parser.add_argument("--size", type=int, default=1500, help="Target number of single entries (default: 1000)")
    parser.add_argument("--seed", type=int, default=422,   help="Random seed (default: 422)")
    args = parser.parse_args()
    build(args.db, args.out, args.size, args.seed)


if __name__ == "__main__":
    main()
