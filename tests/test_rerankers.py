#!/usr/bin/env python3
"""
Reranker-specific tests — unit tests for each reranker class plus live API
integration tests that exercise the LLM path with a real OpenRouter key.

Usage:
    python test_rerankers.py                                   # unit tests only (no API key needed)
    python test_rerankers.py --key sk-or-v1-...               # unit + LLM integration tests
    python test_rerankers.py --url http://host:8000 --key ...  # against a remote server
    python test_rerankers.py --unit-only                       # skip all live API tests
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import List, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests not installed — pip install requests")
    sys.exit(1)

# Allow importing reranking.py from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from reranking import (
        LLMReranker,
        LateInteractionReranker,
        BiomedicalContextReranker,
        EnsembleReranker,
        create_reranker,
    )
    RERANKING_AVAILABLE = True
except ImportError as e:
    print(f"  [WARN] Could not import reranking.py: {e} — unit tests will be skipped")
    RERANKING_AVAILABLE = False

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
INFO = "\033[94mINFO\033[0m"

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


def info(msg: str):
    print(f"  [{INFO}] {msg}")


def post(url: str, path: str, payload: dict) -> Optional[requests.Response]:
    try:
        return requests.post(f"{url}{path}", json=payload, timeout=60)
    except Exception:
        return None


# ── Shared test data ──────────────────────────────────────────────────────────

CANDIDATES = [
    {"preferred_label": "Type 2 Diabetes Mellitus", "definition": "A metabolic disorder characterized by insulin resistance"},
    {"preferred_label": "Hypertension",              "definition": "Persistently elevated arterial blood pressure"},
    {"preferred_label": "Asthma",                    "definition": "Chronic inflammatory disease of the airways"},
    {"preferred_label": "Lung Cancer",               "definition": "Malignant tumor originating in lung tissue"},
    {"preferred_label": "Alzheimer Disease",         "definition": "Progressive neurodegenerative disorder causing dementia"},
]

QUERY_DIABETES    = "type 2 diabetes insulin resistance"
QUERY_LUNG_CANCER = "lung malignant tumor"

# Small set of queries for live API smoke tests
API_QUERIES = [
    ("diabetes",      "Type 2 Diabetes"),
    ("hypertension",  "hypertension"),
    ("lung cancer",   "lung"),
]


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_unit_late_interaction():
    section("Unit — LateInteractionReranker")
    model_name = os.getenv("LATE_INTERACTION_MODEL", "jinaai/jina-colbert-v2")
    reranker = LateInteractionReranker(model_name=model_name)

    results = reranker.rerank(QUERY_DIABETES, CANDIDATES, top_k=3)
    check("Returns list",          isinstance(results, list))
    check("Returns top_k results", len(results) == 3, f"got {len(results)}")
    check("Each item is (int, float)", all(isinstance(i, int) and isinstance(s, float) for i, s in results))

    # Diabetes query should rank the diabetes candidate highest
    top_idx = results[0][0]
    top_label = CANDIDATES[top_idx]["preferred_label"]
    check("Top result relevant to query", "diabetes" in top_label.lower(), f"top='{top_label}'")

    # Accepts extra kwargs without raising
    try:
        reranker.rerank(QUERY_DIABETES, CANDIDATES, openrouter_api_key="dummy", openrouter_model="dummy")
        check("Accepts openrouter_api_key kwarg without error", True)
    except TypeError as e:
        check("Accepts openrouter_api_key kwarg without error", False, str(e))

    # Empty candidates
    check("Empty candidates → []", reranker.rerank("query", []) == [])


def test_unit_biomedical():
    section("Unit — BiomedicalContextReranker")
    reranker = BiomedicalContextReranker()

    results = reranker.rerank(QUERY_LUNG_CANCER, CANDIDATES, top_k=3)
    check("Returns list",          isinstance(results, list))
    check("Returns top_k results", len(results) == 3, f"got {len(results)}")
    check("Each item is (int, float)", all(isinstance(i, int) and isinstance(s, float) for i, s in results))

    # Cancer keyword should boost lung cancer candidate
    top_idx = results[0][0]
    top_label = CANDIDATES[top_idx]["preferred_label"]
    check("Top result relevant to cancer query", "cancer" in top_label.lower() or "lung" in top_label.lower(),
          f"top='{top_label}'")

    # Accepts extra kwargs without raising (the bug that was reported)
    try:
        reranker.rerank(QUERY_DIABETES, CANDIDATES, openrouter_api_key="dummy", openrouter_model="dummy")
        check("Accepts openrouter_api_key kwarg without error", True)
    except TypeError as e:
        check("Accepts openrouter_api_key kwarg without error", False, str(e))

    # All scores should be ≥ 1.0 (biomedical reranker only multiplies, never divides)
    all_results = reranker.rerank(QUERY_DIABETES, CANDIDATES)
    check("All scores ≥ 1.0", all(s >= 1.0 for _, s in all_results), str([s for _, s in all_results]))

    # Empty candidates
    check("Empty candidates → []", reranker.rerank("query", []) == [])


def test_unit_llm_no_key():
    section("Unit — LLMReranker (no API key — graceful fallback)")
    reranker = LLMReranker(api_key=None)

    # Without a key it should return zero scores, not raise
    results = reranker.rerank(QUERY_DIABETES, CANDIDATES, top_k=3)
    check("Returns list without key",        isinstance(results, list))
    check("Returns zero scores (no key)",    all(s == 0.0 for _, s in results), str(results[:2]))

    # Accepts openrouter_api_key/openrouter_model kwargs
    try:
        reranker.rerank(QUERY_DIABETES, CANDIDATES, openrouter_api_key=None, openrouter_model=None)
        check("Accepts openrouter_api_key kwarg without error", True)
    except TypeError as e:
        check("Accepts openrouter_api_key kwarg without error", False, str(e))


def test_unit_llm_with_key(api_key: str):
    section("Unit — LLMReranker (live OpenRouter call)")
    reranker = LLMReranker(api_key=api_key)

    t0 = time.time()
    results = reranker.rerank(QUERY_DIABETES, CANDIDATES, top_k=5)
    elapsed_ms = (time.time() - t0) * 1000

    check("Returns list",               isinstance(results, list))
    check("Returns 5 results",          len(results) == 5, f"got {len(results)}")
    check("Scores are floats",          all(isinstance(s, float) for _, s in results))

    has_nonzero = any(s > 0.0 for _, s in results)
    check("Has non-zero scores (LLM responded)", has_nonzero, str([(CANDIDATES[i]["preferred_label"], round(s, 3)) for i, s in results]))

    top_idx = results[0][0]
    top_label = CANDIDATES[top_idx]["preferred_label"]
    check("Top result relevant to diabetes query", "diabetes" in top_label.lower(), f"top='{top_label}'")
    info(f"LLM rerank took {elapsed_ms:.0f}ms")


def test_unit_ensemble_no_llm():
    section("Unit — EnsembleReranker (dual_late, no LLM)")
    reranker = create_reranker("dual_late")

    results = reranker.rerank(QUERY_DIABETES, CANDIDATES, top_k=3)
    check("Returns RerankingResult list", isinstance(results, list) and len(results) == 3)
    check("Has final_score",  all(hasattr(r, "final_score") for r in results))
    check("Has llm_score=0",  all(r.llm_score == 0.0 for r in results), "expected 0 (no LLM)")
    check("Has late_interaction_score", all(isinstance(r.late_interaction_score, float) for r in results))

    top_label = results[0].preferred_label
    check("Top result relevant to diabetes", "diabetes" in top_label.lower(), f"top='{top_label}'")


def test_unit_ensemble_with_key(api_key: str):
    section("Unit — EnsembleReranker (full ensemble with LLM key)")
    reranker = create_reranker("ensemble", openrouter_api_key=api_key)

    t0 = time.time()
    results = reranker.rerank(QUERY_DIABETES, CANDIDATES, top_k=5, openrouter_api_key=api_key)
    elapsed_ms = (time.time() - t0) * 1000

    check("Returns 5 results",          len(results) == 5, f"got {len(results)}")
    check("Has non-zero llm_score",     any(r.llm_score > 0.0 for r in results))
    check("Has non-zero late_score",    any(r.late_interaction_score > 0.0 for r in results))
    check("Final scores sum ≤ 1",       all(r.final_score <= 1.01 for r in results))

    top_label = results[0].preferred_label
    check("Top result relevant to diabetes", "diabetes" in top_label.lower(), f"top='{top_label}'")
    info(f"Ensemble rerank took {elapsed_ms:.0f}ms")
    info("Scores: " + "  ".join(f"{r.preferred_label[:20]}={r.final_score:.3f}" for r in results))


# ── Live API integration tests ─────────────────────────────────────────────────

def test_api_biomedical(url: str):
    section("API integration — biomedical reranker (key not required)")
    for query, expected_substr in API_QUERIES:
        r = post(url, "/map/concept", {"text": query, "max_results": 5})
        if r is None or r.status_code != 200:
            check(f"'{query}'", False, f"status={getattr(r, 'status_code', 'err')}")
            continue
        results = r.json().get("results", [])
        ok = len(results) > 0
        top = results[0]["ontology_label"] if results else "—"
        check(f"'{query}' returns results", ok, f"top='{top}'")


def test_api_llm(url: str, api_key: str):
    section("API integration — LLM reranker (key passed per-request)")
    for query, expected_substr in API_QUERIES:
        payload = {
            "text": query,
            "max_results": 5,
            "openrouter_api_key": api_key,
        }
        r = post(url, "/map/concept", payload)
        if r is None or r.status_code != 200:
            check(f"'{query}' with LLM key", False, f"status={getattr(r, 'status_code', 'err')}")
            continue
        results = r.json().get("results", [])
        ok = len(results) > 0
        top = results[0]["ontology_label"] if results else "—"
        check(f"'{query}' with LLM key returns results", ok, f"top='{top}'")

    # Verify /map/search also works with key
    r = post(url, "/map/search", {
        "text": "cold",
        "context": "rhinovirus, common cold, respiratory",
        "max_results": 5,
        "openrouter_api_key": api_key,
    })
    if r and r.status_code == 200:
        results = r.json().get("results", [])
        top = results[0]["ontology_label"] if results else "—"
        check("/map/search with LLM key", len(results) > 0, f"top='{top}'")
    else:
        check("/map/search with LLM key", False, f"status={getattr(r, 'status_code', 'err')}")


def _check_indexing(url: str) -> bool:
    """Return True if server is ready for search."""
    try:
        r = requests.get(f"{url}/health", timeout=10)
        return r.status_code == 200 and r.json().get("indexing_complete", False)
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global passed, failed, skipped

    parser = argparse.ArgumentParser(description="Reranker-specific tests")
    parser.add_argument("--url",       default="http://localhost:8000", help="API base URL")
    parser.add_argument("--key",       default=None,                    help="OpenRouter API key for LLM tests")
    parser.add_argument("--unit-only", action="store_true",             help="Skip live API tests")
    args = parser.parse_args()

    api_key = args.key or os.getenv("OPENROUTER_API_KEY")

    print(f"Testing rerankers  (server: {args.url})")
    if api_key:
        info(f"OpenRouter key provided — LLM tests will run")
    else:
        info("No OpenRouter key — LLM unit tests limited to fallback behaviour; API integration skipped")

    # ── Unit tests ────────────────────────────────────────────────────────────
    if RERANKING_AVAILABLE:
        test_unit_late_interaction()
        test_unit_biomedical()
        test_unit_llm_no_key()
        test_unit_ensemble_no_llm()

        if api_key:
            test_unit_llm_with_key(api_key)
            test_unit_ensemble_with_key(api_key)
        else:
            skip("Unit — LLMReranker (live call)", "no --key provided")
            skip("Unit — EnsembleReranker (with LLM)", "no --key provided")
    else:
        skip("All unit tests", "reranking.py not importable")

    # ── Live API integration tests ────────────────────────────────────────────
    if args.unit_only:
        skip("All API integration tests", "--unit-only flag set")
    else:
        ready = _check_indexing(args.url)
        if not ready:
            skip("API integration tests", f"server at {args.url} not ready (indexing incomplete or unreachable)")
        else:
            test_api_biomedical(args.url)
            if api_key:
                test_api_llm(args.url, api_key)
            else:
                skip("API integration — LLM reranker", "no --key provided")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*50}")
    if failed == 0:
        print(f"RESULT: {passed}/{total} passed  {skipped} skipped  ALL PASSED")
    else:
        print(f"RESULT: {passed}/{total} passed  {failed} FAILED  {skipped} skipped")
    print(f"{'='*50}\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
