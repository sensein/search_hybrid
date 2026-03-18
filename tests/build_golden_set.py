#!/usr/bin/env python3
"""
Build (or refresh) tests/golden_set.json from the live bioportal.db.

Generates a stratified golden set with:
  - Exact-match queries          → tested via /map/concept
  - Definition-as-context        → tested via /map/search
  - Abbreviation/synonym queries → tested via /map/concept or /map/search
  - Disambiguation (multi-sense) → tested via /map/search
  - Batch groups                 → tested via /map/batch (groups of 2–5)

Usage:
    python tests/build_golden_set.py                          # default: ~1000 entries, writes tests/golden_set.json
    python tests/build_golden_set.py --size 500               # fewer entries
    python tests/build_golden_set.py --db /path/to/db         # custom DB path
    python tests/build_golden_set.py --out tests/my_set.json  # custom output path
"""

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).parent
_DEFAULT_DB  = os.getenv("DATABASE_PATH", "bioportal.db")
_DEFAULT_OUT = str(_THIS_DIR / "golden_set.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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
    query: str,
    acceptable: List[str],
    context: Optional[str],
    endpoint: str,
    note: str,
) -> dict:
    return {
        "query": query,
        "context": context,
        "acceptable_labels": [a.lower() for a in acceptable],
        "endpoint": endpoint,   # "concept" | "search" | "batch"
        "note": note,
    }


# ── sampling strategies ───────────────────────────────────────────────────────

def sample_exact_match_no_context(cur, rng: random.Random, n: int) -> List[dict]:
    """
    Real concept preferred labels — no context.
    Tested via /map/concept.
    """
    # Focus on disease/phenotype/anatomy ontologies for recognisable labels
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
    """, list(prefer_onts) + [n * 3])  # over-sample then deduplicate
    rows = cur.fetchall()
    seen = set()
    entries = []
    for row in rows:
        label = row['preferred_label']
        key   = label.lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append(_make_entry(
            query=label,
            acceptable=[label],
            context=None,
            endpoint="concept",
            note="exact_match_no_context",
        ))
        if len(entries) >= n:
            break
    return entries


def sample_definition_context(cur, rng: random.Random, n: int) -> List[dict]:
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
        key   = label.lower()
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
        ))
        if len(entries) >= n:
            break
    return entries


def sample_synonym_as_query(cur, rng: random.Random, n: int) -> List[dict]:
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
        key = syn_query.lower()
        if key in seen or key == preferred.lower():
            continue
        seen.add(key)
        # Fetch extra synonyms as context
        extra_syns = _syns(cur, row['id'], 3)
        ctx = ", ".join(s for s in extra_syns if s.lower() != syn_query.lower())[:150] or None
        entries.append(_make_entry(
            query=syn_query,
            acceptable=[preferred],
            context=ctx,
            endpoint="concept",
            note="synonym_as_query",
        ))
        if len(entries) >= n:
            break
    return entries


def sample_alt_label_as_query(cur, rng: random.Random, n: int) -> List[dict]:
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
        ))
        if len(entries) >= n:
            break
    return entries


def sample_batch_groups(entries_pool: List[dict], rng: random.Random, n_groups: int, group_size: int = 3) -> List[dict]:
    """
    Group concept entries into batch test items.
    Each batch entry contains multiple (query, acceptable) pairs.
    Tested via /map/batch.
    """
    # Pick entries that have no context (cleaner for batch test)
    pool = [e for e in entries_pool if e["context"] is None and e["endpoint"] == "concept"]
    rng.shuffle(pool)

    batches = []
    for chunk in _chunks(pool, group_size):
        if len(chunk) < 2:
            break
        batches.append({
            "queries": [{"text": e["query"], "context": e.get("context")} for e in chunk],
            "acceptable_per_concept": [e["acceptable_labels"] for e in chunk],
            "endpoint": "batch",
            "note": f"batch_group_size_{len(chunk)}",
        })
        if len(batches) >= n_groups:
            break
    return batches


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
    batches = sample_batch_groups(exact + syn, rng, n_batches, group_size=3)

    con.close()

    # Flatten into a single list; batch entries are structured differently
    all_entries = exact + defctx + syn + alt
    rng.shuffle(all_entries)

    result = {
        "single": all_entries,    # entries for /map/concept and /map/search
        "batch":  batches,        # entries for /map/batch
        "meta": {
            "total_single": len(all_entries),
            "total_batch_groups": len(batches),
            "breakdown": {
                "exact_match_no_context":                      len(exact),
                "exact_match_with_definition_context":         len(defctx),
                "synonym_as_query":                            len(syn),
                "alt_label_as_query_with_definition_context":  len(alt),
                "batch_groups":                                len(batches),
            },
            "db": db_path,
            "seed": seed,
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved {len(all_entries)} single + {len(batches)} batch groups → {out_path}")
    logger.info(f"Breakdown: {result['meta']['breakdown']}")


def main():
    parser = argparse.ArgumentParser(description="Build golden_set.json from bioportal.db")
    parser.add_argument("--db",   default=_DEFAULT_DB,  help=f"Source database (default: {_DEFAULT_DB})")
    parser.add_argument("--out",  default=_DEFAULT_OUT, help=f"Output path (default: {_DEFAULT_OUT})")
    parser.add_argument("--size", type=int, default=1000, help="Target number of single entries (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,   help="Random seed (default: 42)")
    args = parser.parse_args()
    build(args.db, args.out, args.size, args.seed)


if __name__ == "__main__":
    main()
