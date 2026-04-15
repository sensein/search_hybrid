#!/usr/bin/env python3
"""
Tests for ontology filter support in HybridRetriever.retrieve().

Verifies that passing ontology_ids filters candidates so only concepts
from the requested ontologies are returned — across unit and API levels.

Usage:
    python tests/filter_test.py
    python tests/filter_test.py --url http://localhost:8000   # + API integration
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

try:
    import requests
except ImportError:
    requests = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from retrieval import HybridRetriever, RetrievalCandidate
    RETRIEVAL_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Could not import retrieval.py: {e}")
    RETRIEVAL_AVAILABLE = False

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

passed = 0
failed = 0
skipped = 0


def section(title: str):
    print(f"\n── {title}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    global passed, failed
    if ok:
        passed += 1
        print(f"  [{PASS}] {label}" + (f": {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  [{FAIL}] {label}" + (f": {detail}" if detail else ""))
    return ok


def skip(label: str, reason: str = ""):
    global skipped
    skipped += 1
    print(f"  [{SKIP}] {label}" + (f": {reason}" if reason else ""))


# ── Shared test fixtures ───────────────────────────────────────────────────────

# concepts_map: doc_idx → concept metadata (mixed ontologies)
CONCEPTS_MAP = {
    0: {"class_uri": "http://purl.obolibrary.org/obo/UPHENO_0001001",
        "preferred_label": "renal phenotype",          "ontology_id": "UPHENO"},
    1: {"class_uri": "http://purl.obolibrary.org/obo/UPHENO_0002001",
        "preferred_label": "decreased kidney function", "ontology_id": "UPHENO"},
    2: {"class_uri": "http://purl.obolibrary.org/obo/MONDO_0005240",
        "preferred_label": "kidney disease",           "ontology_id": "MONDO"},
    3: {"class_uri": "http://purl.obolibrary.org/obo/SNOMEDCT_709044004",
        "preferred_label": "chronic kidney disease",   "ontology_id": "SNOMEDCT"},
    4: {"class_uri": "http://purl.obolibrary.org/obo/MESH_D007674",
        "preferred_label": "kidney diseases",          "ontology_id": "MESH"},
}

# All doc indices returned by both mock retrievers (covers all concepts)
ALL_INDICES: List[Tuple[int, float]] = [
    (0, 5.0), (1, 4.5), (2, 4.0), (3, 3.5), (4, 3.0),
]


def _make_retriever(ontology_ids=None) -> HybridRetriever:
    """Build a HybridRetriever with mocked BM25 and dense sub-models."""
    bm25_mock  = MagicMock()
    dense_mock = MagicMock()
    bm25_mock.retrieve.return_value  = ALL_INDICES
    dense_mock.retrieve.return_value = ALL_INDICES

    hr = HybridRetriever(
        bm25_weight=0.3,
        dense_weight=0.7,
        bm25_model=bm25_mock,
        dense_model=dense_mock,
    )
    return hr


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_no_filter_returns_all():
    """Without ontology_ids, all concepts are returned."""
    section("No filter — all ontologies returned")

    hr = _make_retriever()
    results = hr.retrieve("kidney disease", k=10, concepts_map=CONCEPTS_MAP, ontology_ids=None)

    ontologies_found = {r.ontology_id for r in results}
    check("Returns results from multiple ontologies",
          len(ontologies_found) > 1, str(ontologies_found))
    check(f"All 5 concepts returned", len(results) == 5, f"got {len(results)}")


def test_single_ontology_filter():
    """Requesting UPHENO returns only UPHENO concepts."""
    section("Single ontology filter — UPHENO")

    hr = _make_retriever()
    results = hr.retrieve(
        "kidney disease",
        k=10,
        concepts_map=CONCEPTS_MAP,
        ontology_ids=["UPHENO"],
    )

    check("Result count matches UPHENO concepts in map",
          len(results) == 2, f"got {len(results)}")
    check("All results are UPHENO",
          all(r.ontology_id == "UPHENO" for r in results),
          str([r.ontology_id for r in results]))
    check("No MONDO, SNOMEDCT, MESH in results",
          not any(r.ontology_id in {"MONDO", "SNOMEDCT", "MESH"} for r in results))


def test_multiple_ontology_filter():
    """Requesting UPHENO,MONDO returns only those two ontologies."""
    section("Multiple ontology filter — UPHENO,MONDO")

    hr = _make_retriever()
    results = hr.retrieve(
        "kidney disease",
        k=10,
        concepts_map=CONCEPTS_MAP,
        ontology_ids=["UPHENO", "MONDO"],
    )

    ontologies_found = {r.ontology_id for r in results}
    check("Returns 3 concepts (2 UPHENO + 1 MONDO)",
          len(results) == 3, f"got {len(results)}")
    check("Only UPHENO and MONDO in results",
          ontologies_found == {"UPHENO", "MONDO"}, str(ontologies_found))


def test_filter_case_insensitive():
    """Filter should work regardless of case (upheno, Upheno, UPHENO)."""
    section("Case insensitivity — upheno vs UPHENO")

    hr = _make_retriever()
    for variant in ("upheno", "Upheno", "UPHENO"):
        results = hr.retrieve(
            "kidney disease",
            k=10,
            concepts_map=CONCEPTS_MAP,
            ontology_ids=[variant],
        )
        check(f"ontology_ids=['{variant}'] returns UPHENO results",
              len(results) == 2 and all(r.ontology_id == "UPHENO" for r in results),
              f"got {len(results)} results")


def test_unknown_ontology_returns_empty():
    """A filter for an ontology not in the index returns no results."""
    section("Unknown ontology — empty result")

    hr = _make_retriever()
    results = hr.retrieve(
        "kidney disease",
        k=10,
        concepts_map=CONCEPTS_MAP,
        ontology_ids=["NONEXISTENT"],
    )
    check("Returns empty list for unknown ontology",
          results == [], f"got {len(results)}")


def test_results_sorted_by_score():
    """Filtered results must still be sorted by combined_score descending."""
    section("Score ordering preserved after filter")

    hr = _make_retriever()
    results = hr.retrieve(
        "kidney disease",
        k=10,
        concepts_map=CONCEPTS_MAP,
        ontology_ids=["UPHENO", "MONDO", "SNOMEDCT", "MESH"],
    )
    scores = [r.combined_score for r in results]
    check("Results sorted by combined_score descending",
          scores == sorted(scores, reverse=True),
          str([round(s, 4) for s in scores]))


def test_k_limit_respected_after_filter():
    """k cap applies after ontology filter, not before."""
    section("k limit after filter")

    hr = _make_retriever()
    # Without filter there are 5 concepts; request k=2
    results = hr.retrieve(
        "kidney disease",
        k=2,
        concepts_map=CONCEPTS_MAP,
        ontology_ids=None,
    )
    check("k=2 returns at most 2 results", len(results) <= 2, f"got {len(results)}")

    # Filter to UPHENO (2 concepts) with k=1 → should return 1
    results = hr.retrieve(
        "kidney disease",
        k=1,
        concepts_map=CONCEPTS_MAP,
        ontology_ids=["UPHENO"],
    )
    check("UPHENO filter + k=1 returns exactly 1 result",
          len(results) == 1, f"got {len(results)}")
    check("That 1 result is UPHENO",
          results[0].ontology_id == "UPHENO")


# ── API integration tests ─────────────────────────────────────────────────────

def test_api_concept_upheno(url: str):
    """
    End-to-end: POST /map/concept with ontologies=UPHENO returns only UPHENO
    concepts (the kidney disease / GFR example from the spec).
    """
    section(f"API — POST /map/concept with ontologies=UPHENO ({url})")

    if requests is None:
        skip("/map/concept UPHENO", "requests not installed")
        return

    payload = {
        "text": "kidney disease",
        "context": "progressive decline in GFR",
        "max_results": 5,
        "ontologies": "UPHENO",
    }

    try:
        r = requests.post(f"{url}/map/concept", json=payload, timeout=30)
    except Exception as e:
        skip("/map/concept UPHENO", f"connection error: {e}")
        return

    check("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code != 200:
        print(f"    Response: {r.text[:300]}")
        return

    body    = r.json()
    results = body.get("results", [])

    check("Response has results", len(results) > 0, f"got {len(results)}")

    ontologies_in_response = {res.get("ontology") for res in results}
    check("All results belong to UPHENO",
          ontologies_in_response <= {"UPHENO"},
          str(ontologies_in_response))
    check("No non-UPHENO concepts leaked through",
          not any(res.get("ontology") not in {None, "UPHENO"} for res in results),
          str(ontologies_in_response))

    if results:
        top = results[0]
        check("Top result has ontology field",    "ontology"       in top)
        check("Top result has ontology_id field", "ontology_id"    in top)
        check("Top result has final_score field", "final_score"    in top)


def test_api_no_filter_returns_multiple_ontologies(url: str):
    """Without an ontology filter, results span more than one ontology."""
    section(f"API — POST /map/concept without filter (baseline)")

    if requests is None:
        skip("/map/concept no filter", "requests not installed")
        return

    payload = {
        "text": "kidney disease",
        "context": "progressive decline in GFR",
        "max_results": 10,
    }

    try:
        r = requests.post(f"{url}/map/concept", json=payload, timeout=30)
    except Exception as e:
        skip("/map/concept no filter", f"connection error: {e}")
        return

    check("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code != 200:
        return

    results = r.json().get("results", [])
    ontologies_found = {res.get("ontology") for res in results}
    check("Multiple ontologies present without filter",
          len(ontologies_found) > 1, str(ontologies_found))


def test_api_batch_upheno(url: str):
    """POST /map/batch with ontologies=UPHENO — all per-concept results filtered."""
    section(f"API — POST /map/batch with ontologies=UPHENO ({url})")

    if requests is None:
        skip("/map/batch UPHENO", "requests not installed")
        return

    payload = {
        "text": ["kidney disease", "renal failure"],
        "max_results": 3,
        "ontologies": "UPHENO",
    }

    try:
        r = requests.post(f"{url}/map/batch", json=payload, timeout=60)
    except Exception as e:
        skip("/map/batch UPHENO", f"connection error: {e}")
        return

    check("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code != 200:
        return

    body = r.json()
    results_map = body.get("results", {})
    check("Response contains results dict", isinstance(results_map, dict))

    for concept, results in results_map.items():
        ontologies = {res.get("ontology") for res in results}
        check(f"'{concept}' — all results UPHENO",
              ontologies <= {"UPHENO"},
              str(ontologies))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test ontology filter in HybridRetriever")
    parser.add_argument("--url",       default=None,
                        help="API base URL for integration tests (e.g. http://localhost:8000)")
    parser.add_argument("--unit-only", action="store_true",
                        help="Skip API integration tests")
    args = parser.parse_args()

    print("Ontology filter tests")
    print("(HybridRetriever.retrieve() must honour ontology_ids across all retrieval modes)\n")

    if not RETRIEVAL_AVAILABLE:
        print("ERROR: retrieval.py not importable — aborting")
        sys.exit(1)

    # Unit tests (no server needed)
    test_no_filter_returns_all()
    test_single_ontology_filter()
    test_multiple_ontology_filter()
    test_filter_case_insensitive()
    test_unknown_ontology_returns_empty()
    test_results_sorted_by_score()
    test_k_limit_respected_after_filter()

    # API integration tests
    if not args.unit_only and args.url:
        test_api_concept_upheno(args.url)
        test_api_no_filter_returns_multiple_ontologies(args.url)
        test_api_batch_upheno(args.url)
    elif not args.unit_only:
        skip("API integration tests", "no --url provided; use --url http://localhost:8000 to enable")

    total = passed + failed
    print(f"\n{'='*55}")
    if failed == 0:
        print(f"RESULT: {passed}/{total} passed  {skipped} skipped  ALL PASSED")
    else:
        print(f"RESULT: {passed}/{total} passed  {failed} FAILED  {skipped} skipped")
    print(f"{'='*55}\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
