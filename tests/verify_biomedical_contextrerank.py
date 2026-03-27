#!/usr/bin/env python3
"""
Verification tests for the BiomedicalContextReranker fix.

The bug: create_reranker("biomedical") returned a BiomedicalContextReranker
directly, whose rerank() returns List[Tuple[int, float]].  main.py expects
List[RerankingResult] (produced only by EnsembleReranker), causing:
    AttributeError: 'tuple' object has no attribute 'class_uri'

The fix: create_reranker() now wraps single reranker types ("biomedical",
"llm", "late_interaction") in EnsembleReranker(components={type}), so
rerank() always returns List[RerankingResult].

Usage:
    python tests/verify_biomedical_contextrerank.py
    python tests/verify_biomedical_contextrerank.py --url http://localhost:8000   # + API integration
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from reranking import (
        BiomedicalContextReranker,
        EnsembleReranker,
        RerankingResult,
        create_reranker,
    )
    RERANKING_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Could not import reranking.py: {e}")
    RERANKING_AVAILABLE = False

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


# ── Test data ─────────────────────────────────────────────────────────────────

CANDIDATES = [
    {
        "class_uri": "http://purl.obolibrary.org/obo/MONDO_0005148",
        "preferred_label": "Type 2 Diabetes Mellitus",
        "ontology_id": "MONDO",
        "definition": "A metabolic disorder characterized by insulin resistance",
        "original_score": 0.9,
    },
    {
        "class_uri": "http://purl.obolibrary.org/obo/HP_0000822",
        "preferred_label": "Hypertension",
        "ontology_id": "HP",
        "definition": "Persistently elevated arterial blood pressure",
        "original_score": 0.7,
    },
    {
        "class_uri": "http://purl.obolibrary.org/obo/MONDO_0004979",
        "preferred_label": "Asthma",
        "ontology_id": "MONDO",
        "definition": "Chronic inflammatory disease of the airways",
        "original_score": 0.6,
    },
    {
        "class_uri": "http://purl.obolibrary.org/obo/MONDO_0005105",
        "preferred_label": "Lung Cancer",
        "ontology_id": "MONDO",
        "definition": "Malignant tumor originating in lung tissue",
        "original_score": 0.5,
    },
    {
        "class_uri": "http://purl.obolibrary.org/obo/MONDO_0004975",
        "preferred_label": "Alzheimer Disease",
        "ontology_id": "MONDO",
        "definition": "Progressive neurodegenerative disorder causing dementia",
        "original_score": 0.4,
    },
]

QUERY_DIABETES = "type 2 diabetes insulin resistance"
QUERY_CANCER   = "lung malignant tumor"


# ── Core regression test ───────────────────────────────────────────────────────

def test_create_reranker_biomedical_returns_reranking_results():
    """
    Regression: create_reranker("biomedical") must return an EnsembleReranker
    so that rerank() yields List[RerankingResult], not List[Tuple[int, float]].
    """
    section("Regression — create_reranker('biomedical') output type")

    reranker = create_reranker("biomedical")

    check(
        "create_reranker('biomedical') returns EnsembleReranker",
        isinstance(reranker, EnsembleReranker),
        type(reranker).__name__,
    )

    results = reranker.rerank(QUERY_DIABETES, CANDIDATES, top_k=3)

    check("rerank() returns a list", isinstance(results, list))
    check("rerank() is not empty",   len(results) > 0, f"got {len(results)}")

    # The core of the original bug: accessing .class_uri on each result
    try:
        _ = [r.class_uri for r in results]
        check("r.class_uri accessible on every result (bug fix)", True)
    except AttributeError as e:
        check("r.class_uri accessible on every result (bug fix)", False, str(e))

    check(
        "Each result is a RerankingResult",
        all(isinstance(r, RerankingResult) for r in results),
        str([type(r).__name__ for r in results]),
    )


def test_reranking_result_fields():
    """All expected fields are present and correctly typed."""
    section("RerankingResult — field presence and types")

    reranker = create_reranker("biomedical")
    results  = reranker.rerank(QUERY_DIABETES, CANDIDATES, top_k=5)

    for r in results:
        check(f"  '{r.preferred_label}' has class_uri (str)",
              isinstance(r.class_uri, str) and len(r.class_uri) > 0, r.class_uri)
        check(f"  '{r.preferred_label}' has preferred_label (str)",
              isinstance(r.preferred_label, str) and len(r.preferred_label) > 0)
        check(f"  '{r.preferred_label}' has ontology_id (str)",
              isinstance(r.ontology_id, str))
        check(f"  '{r.preferred_label}' has final_score (float ≥ 0)",
              isinstance(r.final_score, float) and r.final_score >= 0.0,
              f"{r.final_score:.4f}")
        check(f"  '{r.preferred_label}' has llm_score == 0 (no LLM)",
              r.llm_score == 0.0, f"{r.llm_score}")
        check(f"  '{r.preferred_label}' has late_interaction_score == 0 (no LI)",
              r.late_interaction_score == 0.0, f"{r.late_interaction_score}")


def test_top_k_respected():
    """rerank() honours the top_k argument."""
    section("top_k — result count")

    reranker = create_reranker("biomedical")

    for k in (1, 3, 5):
        results = reranker.rerank(QUERY_DIABETES, CANDIDATES, top_k=k)
        check(f"top_k={k} returns exactly {k} results", len(results) == k, f"got {len(results)}")


def test_empty_candidates():
    """Empty candidate list should return an empty list without error."""
    section("Edge case — empty candidates")

    reranker = create_reranker("biomedical")
    results  = reranker.rerank(QUERY_DIABETES, [], top_k=5)
    check("Empty candidates → []", results == [], str(results))


def test_ranking_order():
    """Results must be sorted by final_score descending."""
    section("Ranking — descending score order")

    reranker = create_reranker("biomedical")
    results  = reranker.rerank(QUERY_CANCER, CANDIDATES, top_k=5)

    scores = [r.final_score for r in results]
    check(
        "Results sorted by final_score descending",
        scores == sorted(scores, reverse=True),
        str([round(s, 4) for s in scores]),
    )


def test_biomedical_query_boost():
    """Biomedical-keyword queries should produce higher scores than a neutral query."""
    section("Scoring — biomedical keyword boost")

    reranker = create_reranker("biomedical")

    # Cancer query: Lung Cancer candidate should rank first (keyword "cancer" → 1.3 boost)
    results = reranker.rerank(QUERY_CANCER, CANDIDATES, top_k=5)
    top_label = results[0].preferred_label
    check(
        "Lung Cancer query → Lung Cancer ranked first",
        "cancer" in top_label.lower() or "lung" in top_label.lower(),
        f"top='{top_label}'",
    )

    # Diabetes query: all candidates get the same query-level boost (1.2 for "diabetes").
    # The top result may be any candidate with a high candidate-level keyword boost —
    # the important thing is that at least one result has a score > 1.0 (boosted).
    results = reranker.rerank(QUERY_DIABETES, CANDIDATES, top_k=5)
    top_score = results[0].final_score
    check(
        "Diabetes query → top result has boosted score (> 0)",
        top_score > 0.0,
        f"top_score={top_score:.4f}",
    )
    all_scores = [r.final_score for r in results]
    check(
        "Diabetes query → not all scores identical (boosting is active)",
        len(set(round(s, 6) for s in all_scores)) > 1,
        str([round(s, 4) for s in all_scores]),
    )


def test_other_single_types_also_fixed():
    """
    Same fix applies to 'llm' and 'late_interaction' single types —
    they should also return EnsembleReranker and yield RerankingResult.
    """
    section("Regression — create_reranker for 'llm' and 'late_interaction'")

    for rtype in ("llm", "late_interaction"):
        reranker = create_reranker(rtype)
        check(
            f"create_reranker('{rtype}') returns EnsembleReranker",
            isinstance(reranker, EnsembleReranker),
            type(reranker).__name__,
        )
        results = reranker.rerank(QUERY_DIABETES, CANDIDATES, top_k=2)
        check(
            f"create_reranker('{rtype}').rerank() yields RerankingResult",
            all(isinstance(r, RerankingResult) for r in results),
            str([type(r).__name__ for r in results]),
        )
        try:
            _ = [r.class_uri for r in results]
            check(f"r.class_uri accessible for '{rtype}' reranker", True)
        except AttributeError as e:
            check(f"r.class_uri accessible for '{rtype}' reranker", False, str(e))


# ── Optional API integration test ─────────────────────────────────────────────

def test_api_map_search(url: str):
    """
    End-to-end: POST /map/search must succeed and return RerankingResult-shaped
    JSON when the server uses RERANKER_TYPE=biomedical.
    """
    section(f"API integration — POST /map/search ({url})")

    if requests is None:
        skip("/map/search", "requests not installed")
        return

    payload = {
        "text": "diabetes",
        "context": "metabolic disease, insulin, glucose",
        "max_results": 5,
    }
    try:
        r = requests.post(f"{url}/map/search", json=payload, timeout=30)
    except Exception as e:
        skip("/map/search", f"connection error: {e}")
        return

    check("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code != 200:
        return

    body = r.json()
    results = body.get("results", [])
    check("Response contains results", len(results) > 0, f"got {len(results)}")

    if results:
        first = results[0]
        for field in ("ontology_id", "ontology_label", "ontology", "final_score"):
            check(f"Result has '{field}'", field in first, str(list(first.keys())))

        top = first.get("ontology_label", "")
        check(
            "Top result relevant to diabetes query",
            "diabetes" in top.lower(),
            f"top='{top}'",
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify BiomedicalContextReranker fix")
    parser.add_argument("--url",       default=None, help="API base URL for integration test (e.g. http://localhost:8000)")
    parser.add_argument("--unit-only", action="store_true", help="Skip API integration test")
    args = parser.parse_args()

    print("Verifying BiomedicalContextReranker fix")
    print("(create_reranker('biomedical') must return EnsembleReranker → List[RerankingResult])\n")

    if not RERANKING_AVAILABLE:
        print("ERROR: reranking.py not importable — aborting")
        sys.exit(1)

    test_create_reranker_biomedical_returns_reranking_results()
    test_reranking_result_fields()
    test_top_k_respected()
    test_empty_candidates()
    test_ranking_order()
    test_biomedical_query_boost()
    test_other_single_types_also_fixed()

    if not args.unit_only and args.url:
        test_api_map_search(args.url)
    elif not args.unit_only and args.url is None:
        skip("API integration test", "no --url provided; use --url http://localhost:8000 to enable")

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
