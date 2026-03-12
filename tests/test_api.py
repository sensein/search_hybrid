#!/usr/bin/env python3
"""
API test script — functional tests, accuracy evaluation, and performance benchmarking.

Golden set is loaded from tests/golden_set.json.
Regenerate it with:
    python tests/build_golden_set.py          # ~1000 entries (default)
    python tests/build_golden_set.py --size 500

Usage:
    python tests/test_api.py                              # functional tests only
    python tests/test_api.py --accuracy                   # + accuracy (Hit@k, MRR) per endpoint
    python tests/test_api.py --perf                       # + performance benchmark
    python tests/test_api.py --all                        # everything
    python tests/test_api.py --url http://host:8000 --all --verbose
    python tests/test_api.py --wait                       # poll /health until indexing done
    python tests/test_api.py --accuracy --max-accuracy 200  # limit accuracy queries
    python tests/test_api.py --all --plot --plot-dir test_results/
    python tests/test_api.py --all --plot --csv           # also save CSV files
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("ERROR: requests not installed — pip install requests")
    sys.exit(1)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
INFO = "\033[94mINFO\033[0m"
WARN = "\033[93mWARN\033[0m"

passed = 0
failed = 0

_THIS_DIR = Path(__file__).parent


def section(title: str):
    print(f"\n── {title}")


def check(label: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [{PASS}] {label}" + (f": {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  [{FAIL}] {label}" + (f": {detail}" if detail else ""))
    return ok


def info(msg: str):
    print(f"  [{INFO}] {msg}")


def warn(msg: str):
    print(f"  [{WARN}] {msg}")


def post(url, path, payload, verbose=False):
    try:
        r = requests.post(f"{url}{path}", json=payload, timeout=60)
        if verbose:
            print(json.dumps(r.json(), indent=2))
        return r
    except Exception:
        return None


def get(url, path, verbose=False):
    try:
        r = requests.get(f"{url}{path}", timeout=30)
        if verbose:
            print(json.dumps(r.json(), indent=2))
        return r
    except Exception:
        return None


def fetch_config(url: str) -> dict:
    """Fetch /config for embedding into plot titles/captions."""
    r = get(url, "/config")
    if r and r.status_code == 200:
        return r.json()
    return {}


# ── Golden set ────────────────────────────────────────────────────────────────

def load_golden(path: Path = _THIS_DIR / "golden_set.json") -> dict:
    """
    Load golden set. Returns dict with keys:
      - "single": list of {query, context, acceptable_labels, endpoint, note}
      - "batch":  list of {queries:[{text, context}], acceptable_per_concept:[[str]], endpoint, note}
      - "meta":   stats dict
    """
    if not path.exists():
        warn(f"golden_set.json not found at {path}")
        warn("Run:  python tests/build_golden_set.py   to generate it")
        return {"single": [], "batch": [], "meta": {}}

    with open(path) as f:
        data = json.load(f)

    # Support legacy flat-list format (plain list of entries)
    if isinstance(data, list):
        return {"single": data, "batch": [], "meta": {"total_single": len(data)}}

    return data


GOLDEN = load_golden()


def _matches(results: list, acceptable: List[str]) -> Optional[int]:
    """
    Determine the rank of the correct answer within the API's returned results list.

    This function drives all Hit@k and MRR computations — it converts a raw results
    list into a single integer rank (or None) that downstream metrics are computed from.

    Args:
        results:    Ordered list of result dicts from the API (index 0 = top/best result).
                    Each dict has at least a "preferred_label" key (str).
        acceptable: List of lowercase string fragments any of which count as a correct match.
                    e.g. ["type 2 diabetes mellitus", "diabetes type ii"]
                    A result matches if ANY acceptable string is a substring of its
                    lowercased preferred_label.  This is intentionally lenient — it
                    handles synonym variations and label reformulations.

    Returns:
        1-based integer rank if the correct answer appears anywhere in `results`,
        e.g. rank=1 means the very first result is correct, rank=3 means the third.
        Returns None if NO result in the list matches any acceptable label (= not found).

    Rank → metric mapping (see _endpoint_stats):
        rank=1  → contributes to Hit@1, Hit@3, Hit@5; RR = 1.0
        rank=2  → contributes to Hit@3, Hit@5;        RR = 0.5
        rank=3  → contributes to Hit@3, Hit@5;        RR = 0.333
        rank=4  → contributes to Hit@5 only;          RR = 0.25
        rank=5  → contributes to Hit@5 only;          RR = 0.2
        None    → contributes to nothing;             RR = 0.0
    """
    for i, r in enumerate(results):
        # Lowercase the returned label for case-insensitive substring matching
        label = r.get("preferred_label", "").lower()
        # Accept if ANY of the acceptable label fragments appears in this label
        if any(a in label for a in acceptable):
            return i + 1  # 1-based: first result in list is rank 1
    return None  # correct answer not found in the top-k results returned by the API


# ── Functional tests ──────────────────────────────────────────────────────────

def test_health(url: str, verbose: bool) -> bool:
    section("Health check")
    r = get(url, "/health", verbose)
    if not check("GET /health reachable", r is not None and r.status_code == 200,
                 f"status={getattr(r,'status_code','err')}"):
        return False
    data = r.json()
    check("database_ready",  data.get("database_ready"),  str(data.get("database_ready")))
    check("retriever_ready", data.get("retriever_ready"), str(data.get("retriever_ready")))
    check("reranker_ready",  data.get("reranker_ready"),  str(data.get("reranker_ready")))
    indexing = data.get("indexing_complete", False)
    if not indexing:
        print(f"  [{SKIP}] indexing_complete=False — search endpoints return 503 until ready")
    else:
        check("indexing_complete", True, "True")
    return indexing


def test_root(url, verbose):
    section("Root")
    r = get(url, "/", verbose)
    check("GET / returns 200", r is not None and r.status_code == 200,
          f"status={getattr(r,'status_code','err')}")


def test_ontologies(url, verbose):
    section("Ontologies list")
    r = get(url, "/ontologies", verbose)
    if not check("GET /ontologies", r is not None and r.status_code == 200,
                 f"status={getattr(r,'status_code','err')}"):
        return
    data = r.json()
    total = data.get("total", 0)
    check("Has ontologies", total > 0, f"{total} ontologies")
    if total > 0:
        first = data["ontologies"][0]
        info(f"Sample: {first['id']} — {first['name']} ({first['num_classes']:,} classes)")


def test_stats(url, verbose):
    section("Stats")
    r = get(url, "/stats", verbose)
    if not check("GET /stats", r is not None and r.status_code == 200,
                 f"status={getattr(r,'status_code','err')}"):
        return
    data = r.json()
    db  = data.get("database", {})
    idx = data.get("indexes", {})
    info(f"classes={db.get('num_classes',0):,}  ontologies={db.get('num_ontologies',0):,}")
    info(f"indexed={idx.get('num_indexed_concepts',0):,}  bm25={idx.get('bm25_indexed')}  dense={idx.get('dense_indexed')}")


def test_single_concept(url, verbose):
    """
    /map/concept: single term, optional context.
    Retrieves MAX_CANDIDATES=20 candidates, returns retrieval_scores in response.
    Best for simple term lookups.
    """
    section("Single concept  POST /map/concept")
    for text, max_results in [("diabetes", 5), ("hypertension", 3), ("lung cancer", 5)]:
        r = post(url, "/map/concept", {"text": text, "max_results": max_results}, verbose)
        if r is None or r.status_code != 200:
            check(f"'{text}'", False, f"status={getattr(r,'status_code','err')}"); continue
        data = r.json()
        results = data.get("results", [])
        ok = len(results) > 0
        check(f"'{text}'", ok,
              f"{len(results)} results  top='{results[0]['preferred_label']}'  {data['processing_time_ms']:.0f}ms"
              if ok else "no results")
    # Verify retrieval_scores are present (feature of /map/concept only)
    r = post(url, "/map/concept", {"text": "diabetes", "max_results": 1})
    if r and r.status_code == 200:
        results = r.json().get("results", [])
        has_scores = bool(results and results[0].get("retrieval_scores"))
        check("retrieval_scores present in /map/concept response", has_scores)


def test_contextual_search(url, verbose):
    """
    /map/search: same as /map/concept but uses MAX_CANDIDATES=30 (larger pool)
    and does NOT return retrieval_scores. Designed for context-heavy disambiguation.
    """
    section("Contextual search  POST /map/search")
    cases = [
        ("cold",    "rhinovirus, respiratory infection, common cold"),
        ("mercury", "heavy metal poisoning, methylmercury, neurotoxicity"),
    ]
    for text, ctx in cases:
        r = post(url, "/map/search", {"text": text, "context": ctx, "max_results": 5}, verbose)
        if r is None or r.status_code != 200:
            check(f"'{text}'+ctx", False, f"status={getattr(r,'status_code','err')}"); continue
        results = r.json().get("results", [])
        ok = len(results) > 0
        check(f"'{text}'+ctx", ok,
              f"top='{results[0]['preferred_label']}'  {r.json()['processing_time_ms']:.0f}ms"
              if ok else "no results")


def test_batch(url, verbose):
    """
    /map/batch: map up to 20 concepts in one request.
    Accepts comma-separated string, list of strings, or list of {text, context} objects.
    Returns dict keyed by concept term.
    """
    section("Batch  POST /map/batch")
    # Format 1: comma-separated string
    r = post(url, "/map/batch", {"text": "diabetes,hypertension,asthma", "max_results": 3}, verbose)
    if r is not None and r.status_code == 200:
        d = r.json()
        check("Format 1 comma-separated", len(d["results"]) == 3,
              f"{d['processing_time_ms']:.0f}ms")
    else:
        check("Format 1 comma-separated", False, f"status={getattr(r,'status_code','err')}")

    # Format 2: list of strings
    r = post(url, "/map/batch", {"text": ["chronic kidney disease", "heart failure"], "max_results": 2}, verbose)
    if r is not None and r.status_code == 200:
        check("Format 2 list of strings", len(r.json()["results"]) == 2,
              f"{r.json()['processing_time_ms']:.0f}ms")
    else:
        check("Format 2 list of strings", False, f"status={getattr(r,'status_code','err')}")

    # Format 3: list of objects with per-concept context
    r = post(url, "/map/batch",
             {"text": [{"text": "diabetes",      "context": "type 2 insulin resistance"},
                       {"text": "lung cancer",   "context": "NSCLC, adenocarcinoma, EGFR"}],
              "max_results": 3}, verbose)
    if r is not None and r.status_code == 200:
        check("Format 3 objects+context", len(r.json()["results"]) == 2,
              f"{r.json()['processing_time_ms']:.0f}ms")
    else:
        check("Format 3 objects+context", False, f"status={getattr(r,'status_code','err')}")


def test_edge_cases(url, verbose):
    section("Edge cases / validation")
    r = post(url, "/map/concept", {"text": "",            "max_results": 5},  verbose)
    check("Empty text → 422",       r is not None and r.status_code == 422,        f"status={getattr(r,'status_code','err')}")
    r = post(url, "/map/concept", {"text": "diabetes",    "max_results": 99}, verbose)
    check("max_results=99 → 422",   r is not None and r.status_code == 422,        f"status={getattr(r,'status_code','err')}")
    r = post(url, "/map/batch",   {"text": [f"c{i}" for i in range(25)], "max_results": 1}, verbose)
    check("Batch >20 → 400/422",    r is not None and r.status_code in (400, 422), f"status={getattr(r,'status_code','err')}")
    r = post(url, "/map/concept", {"text": "xyzzy_nonexistent_zqpwrx", "max_results": 3}, verbose)
    check("Unknown term graceful",  r is not None and r.status_code == 200,        f"total={r.json().get('total_results',0) if r else 'err'}")
    # /map/search with empty context is still valid
    r = post(url, "/map/search",  {"text": "diabetes", "max_results": 3}, verbose)
    check("/map/search no context → 200", r is not None and r.status_code == 200,  f"status={getattr(r,'status_code','err')}")


# ── Accuracy evaluation ───────────────────────────────────────────────────────

def _eval_single(url: str, entry: dict, top_k: int):
    """Evaluate one single entry via /map/concept or /map/search. Returns (rank_or_None, top_label)."""
    endpoint = "/map/concept" if entry.get("endpoint") != "search" else "/map/search"
    payload = {"text": entry["query"], "max_results": top_k}
    if entry.get("context"):
        payload["context"] = entry["context"]
    r = post(url, endpoint, payload)
    if r is None or r.status_code != 200:
        return None, "—"
    results = r.json().get("results", [])
    rank = _matches(results, entry["acceptable_labels"])
    top_label = results[0]["preferred_label"] if results else "—"
    return rank, top_label


def _eval_batch_group(url: str, group: dict, top_k: int) -> List[Optional[int]]:
    """Evaluate one batch group via /map/batch. Returns list of rank_or_None per concept."""
    queries = group["queries"]
    acceptables = group["acceptable_per_concept"]
    payload = {"text": queries, "max_results": top_k}
    r = post(url, "/map/batch", payload)
    if r is None or r.status_code != 200:
        return [None] * len(queries)
    results_dict = r.json().get("results", {})
    ranks = []
    for i, q in enumerate(queries):
        key = q["text"]
        concept_results = results_dict.get(key, [])
        rank = _matches(concept_results, acceptables[i]) if concept_results else None
        ranks.append(rank)
    return ranks


def _endpoint_stats(ranks: List[Optional[int]], ks=(1, 3, 5)) -> dict:
    """
    Compute Hit@k and MRR from a flat list of per-query ranks.

    This is the core metric computation function.  It is called once per endpoint
    (concept / search / batch) and once more for the combined overall result.

    ── Hit@k ────────────────────────────────────────────────────────────────────
    Hit@k answers: "What fraction of queries had the correct answer in the top-k?"

      Hit@k = (number of queries where rank ≤ k) / N

    Where rank is the 1-based position returned by _matches(), or None (not found).
    A query with rank=None contributes 0 to every Hit@k.

    Example for N=4 queries with ranks [1, 3, None, 2]:
      Hit@1 = 1/4 = 25%   (only rank=1 qualifies)
      Hit@3 = 3/4 = 75%   (ranks 1, 2, 3 all qualify)
      Hit@5 = 3/4 = 75%   (None still excluded)

    Stored as raw count in hit_at[k]; divide by n to get percentage.

    ── MRR (Mean Reciprocal Rank) ───────────────────────────────────────────────
    MRR answers: "On average, how high in the ranked list does the correct answer appear?"

      RR_i = 1 / rank_i   if found,  0.0 if not found (rank=None)
      MRR  = mean(RR_i)   over all N queries

    MRR rewards getting the correct answer early: rank=1 → RR=1.0, rank=2 → RR=0.5, etc.
    A score of 0.0 means the correct answer was never found across any query.

    Example for ranks [1, 3, None, 2]:
      RRs  = [1.0, 0.333, 0.0, 0.5]
      MRR  = (1.0 + 0.333 + 0.0 + 0.5) / 4 = 0.458

    Args:
        ranks: List of rank values from _matches() — int (1-based) or None (not found).
               Collected in ep_ranks[endpoint] during test_accuracy().
        ks:    Tuple of k values to compute Hit@k for (default: 1, 3, 5).

    Returns:
        dict with:
          "n"      → number of queries evaluated
          "hit_at" → {k: raw_count} for each k in ks (divide by n for percentage)
          "mrr"    → float MRR in [0, 1]
    """
    n = len(ranks)
    if n == 0:
        return {"n": 0, "hit_at": {k: 0 for k in ks}, "mrr": 0.0}

    # RR per query: 1/rank if found, 0 if not found (None → 0.0)
    rrs = [1.0 / r if r else 0.0 for r in ranks]

    # Hit@k: count queries where rank is defined (not None) and ≤ k
    hit_at = {k: sum(1 for r in ranks if r and r <= k) for k in ks}

    # MRR: arithmetic mean of per-query reciprocal ranks
    return {"n": n, "hit_at": hit_at, "mrr": statistics.mean(rrs)}


def test_accuracy(url: str, verbose: bool, top_k: int = 5, max_queries: Optional[int] = None):
    golden_single = GOLDEN.get("single", [])
    golden_batch  = GOLDEN.get("batch",  [])

    if not golden_single and not golden_batch:
        warn("No golden set — skipping accuracy tests. Run: python tests/build_golden_set.py")
        return None

    # Optionally limit how many we evaluate (faster smoke-test)
    if max_queries and len(golden_single) > max_queries:
        import random
        rng = random.Random(0)
        sample = golden_single[:]
        rng.shuffle(sample)
        golden_single = sample[:max_queries]

    total_single = len(golden_single)
    total_batch_concepts = sum(len(g["queries"]) for g in golden_batch)
    section(
        f"Accuracy evaluation  "
        f"({total_single} single + {len(golden_batch)} batch groups "
        f"[{total_batch_concepts} concepts], top_k={top_k})"
    )

    meta = GOLDEN.get("meta", {})
    if meta.get("breakdown"):
        info("Golden set breakdown: " + "  ".join(f"{k}={v}" for k, v in meta["breakdown"].items()))

    # Per-endpoint rank accumulation.
    # ep_ranks collects the raw rank (int or None) for every evaluated query,
    # keyed by which endpoint was used.  After the loop, _endpoint_stats() converts
    # these lists into Hit@k counts and MRR — one stats dict per endpoint + overall.
    ep_ranks: Dict[str, List[Optional[int]]] = {"concept": [], "search": [], "batch": []}
    per_query_rows: List[dict] = []  # for CSV export — one row per query
    errors = 0

    # ── Single entries (/map/concept and /map/search) ────────────────────────
    info(f"Evaluating {total_single} single queries…")
    for _qi, entry in enumerate(golden_single):
        rank, top_label = _eval_single(url, entry, top_k)
        if not verbose and (_qi + 1) % 50 == 0:
            print(f"  ... {_qi + 1}/{total_single} queries evaluated", flush=True)
        ep = entry.get("endpoint", "concept")  # "concept" or "search"

        if rank is None and top_label == "—":
            warn(f"'{entry['query']}' → request failed")
            errors += 1
            ep_ranks[ep].append(None)
        else:
            ep_ranks[ep].append(rank)

        # Reciprocal rank for this query: 1/rank if found, 0.0 if not found.
        # These per-query RR values are averaged in _endpoint_stats() to produce MRR.
        rr = 1.0 / rank if rank else 0.0

        # Build the per-query CSV row.
        # hit_at_1/3/5 are binary (0/1) per-query indicators — their column mean
        # equals the Hit@k percentage for this endpoint.  reciprocal_rank is the
        # per-query contribution to MRR (also 0/1 range; its column mean = MRR).
        per_query_rows.append({
            "query":       entry["query"],
            "endpoint":    f"/map/{ep}",
            "note":        entry.get("note", ""),
            "rank":        rank if rank else "",         # empty string = not found
            "not_found":   "1" if not rank else "0",
            "hit_at_1":    "1" if rank and rank <= 1 else "0",  # 1 iff correct answer is rank 1
            "hit_at_3":    "1" if rank and rank <= 3 else "0",  # 1 iff correct answer is in top 3
            "hit_at_5":    "1" if rank and rank <= 5 else "0",  # 1 iff correct answer is in top 5
            "reciprocal_rank": f"{rr:.4f}",             # 1/rank (0.0 if not found)
            "top_label":   top_label,
        })

        if verbose:
            mark = PASS if rank == 1 else (WARN if rank else FAIL)
            rank_str = f"rank={rank}" if rank else "NOT FOUND"
            ctx_hint = f" [ctx]" if entry.get("context") else ""
            print(f"  [{mark}] [{ep}]{ctx_hint} '{entry['query']}' → {rank_str}  top='{top_label}'")

    # ── Batch groups (/map/batch) ────────────────────────────────────────────
    # Each batch group contains multiple concepts sent together in one /map/batch request.
    # _eval_batch_group() returns one rank (or None) per concept within the group.
    # We flatten these into ep_ranks["batch"] and per_query_rows exactly like single entries,
    # so _endpoint_stats() treats them identically when computing Hit@k and MRR.
    info(f"Evaluating {len(golden_batch)} batch groups ({total_batch_concepts} concepts)…")
    for _gi, group in enumerate(golden_batch):
        if not verbose:
            print(f"  ... batch group {_gi + 1}/{len(golden_batch)}", flush=True)
        ranks = _eval_batch_group(url, group, top_k)
        for q, rank in zip(group["queries"], ranks):
            # Accumulate rank into the "batch" bucket for per-endpoint stats
            ep_ranks["batch"].append(rank)
            rr = 1.0 / rank if rank else 0.0  # per-query reciprocal rank (0 if not found)
            per_query_rows.append({
                "query":       q["text"],
                "endpoint":    "/map/batch",
                "note":        group.get("note", ""),
                "rank":        rank if rank else "",
                "not_found":   "1" if not rank else "0",
                "hit_at_1":    "1" if rank and rank <= 1 else "0",
                "hit_at_3":    "1" if rank and rank <= 3 else "0",
                "hit_at_5":    "1" if rank and rank <= 5 else "0",
                "reciprocal_rank": f"{rr:.4f}",
                "top_label":   "",  # batch response doesn't expose a single top label per concept
            })
        if verbose:
            for q, rank in zip(group["queries"], ranks):
                mark = PASS if rank == 1 else (WARN if rank else FAIL)
                print(f"  [{mark}] [batch] '{q['text']}' → {'rank='+str(rank) if rank else 'NOT FOUND'}")

    # ── Per-endpoint stats ────────────────────────────────────────────────────
    # At this point ep_ranks["concept"], ep_ranks["search"], ep_ranks["batch"] each hold
    # a flat list of rank values (int or None) for every query evaluated via that endpoint.
    #
    # _endpoint_stats() computes from those lists:
    #   • hit_at[k]  = count of queries where rank is not None and rank ≤ k
    #   • mrr        = mean(1/rank for found, 0 for not-found)
    #
    # "overall" merges all three endpoint rank lists into one and runs the same computation,
    # giving a single aggregate score across all N evaluated queries.
    ep_stats = {ep: _endpoint_stats(ep_ranks[ep]) for ep in ("concept", "search", "batch")}
    all_ranks = ep_ranks["concept"] + ep_ranks["search"] + ep_ranks["batch"]
    overall   = _endpoint_stats(all_ranks)

    # ── Summary table ─────────────────────────────────────────────────────────
    cols = [
        ("concept", f"/map/concept (n={ep_stats['concept']['n']})"),
        ("search",  f"/map/search  (n={ep_stats['search']['n']})"),
        ("batch",   f"/map/batch   (n={ep_stats['batch']['n']})"),
        ("overall", f"Overall      (n={overall['n']})"),
    ]
    header = f"  {'Metric':<12}" + "".join(f"{lbl:>28}" for _, lbl in cols)
    print(f"\n{header}")
    print(f"  {'-'*(12 + 28*len(cols))}")

    def _pct(stats, k, n_fallback=None):
        n = stats["n"] or n_fallback or 1
        return f"{stats['hit_at'][k]/n*100:>27.1f}%" if stats["n"] else f"{'—':>28}"

    for metric, key in [("Hit@1", 1), ("Hit@3", 3), ("Hit@5", 5)]:
        row = f"  {metric:<12}"
        for ep, _ in cols:
            s = ep_stats[ep] if ep != "overall" else overall
            row += _pct(s, key)
        print(row)

    row_mrr = f"  {'MRR':<12}"
    for ep, _ in cols:
        s = ep_stats[ep] if ep != "overall" else overall
        row_mrr += f"{s['mrr']:>28.3f}" if s["n"] else f"{'—':>28}"
    print(row_mrr)
    if errors:
        print(f"  {'Errors':<12}{errors:>28}")

    n_total = overall["n"]
    check(f"Hit@1 ≥ 50% (overall, n={n_total})", overall["hit_at"][1] / n_total >= 0.50 if n_total else False,
          f"{overall['hit_at'][1]/n_total*100:.1f}%" if n_total else "no data")
    check(f"Hit@3 ≥ 70% (overall)", overall["hit_at"][3] / n_total >= 0.70 if n_total else False,
          f"{overall['hit_at'][3]/n_total*100:.1f}%" if n_total else "no data")
    check(f"Hit@5 ≥ 80% (overall)", overall["hit_at"][5] / n_total >= 0.80 if n_total else False,
          f"{overall['hit_at'][5]/n_total*100:.1f}%" if n_total else "no data")
    check("MRR ≥ 0.5 (overall)",    overall["mrr"] >= 0.50 if n_total else False,
          f"{overall['mrr']:.3f}" if n_total else "no data")

    return {
        "ep_stats":     ep_stats,
        "ep_ranks":     ep_ranks,
        "overall":      overall,
        "per_query":    per_query_rows,
        "errors":       errors,
        "top_k":        top_k,
        # legacy key kept for compat
        "hit_at":  overall["hit_at"],
        "mrr":     overall["mrr"],
        "n":       n_total,
    }


# ── Performance benchmark ─────────────────────────────────────────────────────

def test_performance(url: str, n_requests: int = 100, concurrency: int = 4):
    section(f"Performance benchmark  ({n_requests} requests, concurrency={concurrency})")

    queries = [
        "diabetes", "hypertension", "asthma", "breast cancer", "Alzheimer disease",
        "chronic kidney disease", "heart failure", "lung adenocarcinoma", "COVID-19",
        "rheumatoid arthritis", "obesity", "stroke", "depression", "pneumonia", "sepsis",
        "type 2 diabetes", "myocardial infarction", "inflammatory bowel disease", "COPD", "epilepsy",
    ]

    latencies: List[float] = []
    errors = 0
    lock = threading.Lock()

    def _worker(q: str):
        nonlocal errors
        t0 = time.time()
        r = post(url, "/map/concept", {"text": q, "max_results": 5})
        elapsed = (time.time() - t0) * 1000
        with lock:
            if r is not None and r.status_code == 200:
                latencies.append(elapsed)
            else:
                errors += 1

    info("Sequential baseline (5 requests)…")
    seq_times = []
    for q in queries[:5]:
        t0 = time.time()
        r = post(url, "/map/concept", {"text": q, "max_results": 5})
        if r and r.status_code == 200:
            seq_times.append((time.time() - t0) * 1000)
    if seq_times:
        info(f"Sequential: avg={statistics.mean(seq_times):.0f}ms  min={min(seq_times):.0f}ms  max={max(seq_times):.0f}ms")

    info(f"Concurrent load ({n_requests} requests, {concurrency} threads)…")
    t_start = time.time()
    threads = [threading.Thread(target=_worker, args=(queries[i % len(queries)],))
               for i in range(n_requests)]
    for i in range(0, len(threads), concurrency):
        batch_t = threads[i : i + concurrency]
        for t in batch_t: t.start()
        for t in batch_t: t.join()
    total_elapsed = time.time() - t_start

    if latencies:
        ls = sorted(latencies)
        p50 = statistics.median(latencies)
        p95 = ls[int(len(ls) * 0.95)]
        p99 = ls[min(int(len(ls) * 0.99), len(ls) - 1)]
        throughput = len(latencies) / total_elapsed

        print(f"\n  {'Metric':<25} {'Value':>10}")
        print(f"  {'-'*37}")
        print(f"  {'Requests sent':<25} {n_requests:>10}")
        print(f"  {'Successful':<25} {len(latencies):>10}")
        print(f"  {'Errors':<25} {errors:>10}")
        print(f"  {'Throughput (req/s)':<25} {throughput:>10.2f}")
        print(f"  {'Latency p50 (ms)':<25} {p50:>10.0f}")
        print(f"  {'Latency p95 (ms)':<25} {p95:>10.0f}")
        print(f"  {'Latency p99 (ms)':<25} {p99:>10.0f}")
        print(f"  {'Latency min (ms)':<25} {min(latencies):>10.0f}")
        print(f"  {'Latency max (ms)':<25} {max(latencies):>10.0f}")
        print(f"  {'Total wall time (s)':<25} {total_elapsed:>10.1f}")

        check("Error rate < 5%",  errors / n_requests < 0.05,  f"{errors}/{n_requests}")
        check("p50 latency < 2s", p50 < 2000,                  f"{p50:.0f}ms")
        check("p95 latency < 5s", p95 < 5000,                  f"{p95:.0f}ms")
        return {"latencies": latencies, "seq_times": seq_times, "errors": errors,
                "n_requests": n_requests, "concurrency": concurrency,
                "throughput": throughput,
                "p50": p50, "p95": p95, "p99": p99, "total_elapsed": total_elapsed}
    else:
        check("Got responses", False, "all requests failed")
        return None


# ── Batch size variation test ─────────────────────────────────────────────────

def test_batch_sizes(url: str, golden_batch: list, top_k: int = 5) -> Optional[dict]:
    """
    Test /map/batch accuracy and latency at different batch sizes.
    Uses full golden batch groups, 50% of data, and sizes 1, 2, 5, 10, 20.
    """
    if not golden_batch:
        warn("No batch golden set — skipping batch size tests")
        return None

    section(f"Batch size variation  ({len(golden_batch)} groups available)")

    # Flatten all batch concepts for arbitrary re-grouping
    all_concepts = []
    all_acceptables = []
    for group in golden_batch:
        for q, acc in zip(group["queries"], group["acceptable_per_concept"]):
            all_concepts.append(q)
            all_acceptables.append(acc)

    total_concepts = len(all_concepts)
    info(f"Total concepts available: {total_concepts}")

    results: Dict[str, dict] = {}

    # Dataset subsets: 100%, 50%, 25% of available batch concepts
    subsets = {
        "full (100%)":    all_concepts,
        "half (50%)":     all_concepts[: total_concepts // 2],
        "quarter (25%)":  all_concepts[: max(1, total_concepts // 4)],
    }
    # Batch sizes to test (API hard-limit is 20 concepts per request)
    batch_sizes = [5, 10, 15, 20]

    for subset_name, concepts in subsets.items():
        acceptables = all_acceptables[: len(concepts)]
        n_concepts = len(concepts)
        info(f"\n  Dataset subset: {subset_name}  ({n_concepts} concepts)")
        print(f"  {'Batch size':<14} {'Batches':>8} {'Concepts':>10} {'Latency/req (ms)':>18} {'Hit@1':>8} {'Hit@5':>8}")
        print(f"  {'-'*70}")

        for bsize in batch_sizes:
            # Split concepts into batches of size bsize
            batches = [concepts[i : i + bsize] for i in range(0, n_concepts, bsize)]
            acc_batches = [acceptables[i : i + bsize] for i in range(0, n_concepts, bsize)]
            batch_latencies = []
            all_ranks = []

            for batch, acc_b in zip(batches, acc_batches):
                t0 = time.time()
                r = post(url, "/map/batch", {"text": batch, "max_results": top_k})
                elapsed_ms = (time.time() - t0) * 1000
                if r is None or r.status_code != 200:
                    all_ranks.extend([None] * len(batch))
                    continue
                batch_latencies.append(elapsed_ms)
                rdict = r.json().get("results", {})
                for q, acc in zip(batch, acc_b):
                    key = q["text"] if isinstance(q, dict) else q
                    concept_results = rdict.get(key, [])
                    rank = _matches(concept_results, acc) if concept_results else None
                    all_ranks.append(rank)

            if not batch_latencies:
                print(f"  {bsize:<14} {'—':>8} {'—':>10} {'—':>18} {'—':>8} {'—':>8}")
                continue

            avg_lat = statistics.mean(batch_latencies)
            stats_b = _endpoint_stats(all_ranks)
            h1  = f"{stats_b['hit_at'][1]/stats_b['n']*100:.1f}%" if stats_b["n"] else "—"
            h5  = f"{stats_b['hit_at'][5]/stats_b['n']*100:.1f}%" if stats_b["n"] else "—"
            print(f"  {bsize:<14} {len(batches):>8} {stats_b['n']:>10} {avg_lat:>18.0f} {h1:>8} {h5:>8}")

            key = f"{subset_name}|size={bsize}"
            results[key] = {
                "subset":      subset_name,
                "batch_size":  bsize,
                "n_batches":   len(batches),
                "n_concepts":  stats_b["n"],
                "avg_latency_ms": avg_lat,
                "latencies":   batch_latencies,
                **stats_b,
            }

    return results


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(accuracy_data: Optional[dict], perf_data: Optional[dict],
             batch_size_data: Optional[dict], out_dir: str, ts: str, server_config: dict):
    """Save all test results to CSV files in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    # ── Accuracy: per-query rows ─────────────────────────────────────────────
    if accuracy_data and accuracy_data.get("per_query"):
        path = os.path.join(out_dir, f"accuracy_per_query_{ts}.csv")
        with open(path, "w", newline="") as f:
            cols = ["query", "endpoint", "note", "rank", "not_found",
                    "hit_at_1", "hit_at_3", "hit_at_5", "reciprocal_rank", "top_label"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(accuracy_data["per_query"])
        saved.append(path)

    # ── Accuracy: summary by endpoint ────────────────────────────────────────
    if accuracy_data:
        path = os.path.join(out_dir, f"accuracy_summary_{ts}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["endpoint", "n", "hit_at_1_pct", "hit_at_3_pct", "hit_at_5_pct", "mrr"])
            for ep_key, ep_label in [("concept", "/map/concept"), ("search", "/map/search"),
                                     ("batch", "/map/batch"), ("overall", "overall")]:
                s = accuracy_data["ep_stats"].get(ep_key) if ep_key != "overall" else accuracy_data["overall"]
                if s and s["n"]:
                    n = s["n"]
                    w.writerow([
                        ep_label, n,
                        f"{s['hit_at'][1]/n*100:.2f}",
                        f"{s['hit_at'][3]/n*100:.2f}",
                        f"{s['hit_at'][5]/n*100:.2f}",
                        f"{s['mrr']:.4f}",
                    ])
        saved.append(path)

    # ── Config snapshot ───────────────────────────────────────────────────────
    if server_config:
        path = os.path.join(out_dir, f"config_{ts}.json")
        with open(path, "w") as f:
            json.dump(server_config, f, indent=2)
        saved.append(path)

    # ── Performance: per-request latencies ───────────────────────────────────
    if perf_data:
        path = os.path.join(out_dir, f"perf_latencies_{ts}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["request_index", "latency_ms"])
            for i, lat in enumerate(perf_data["latencies"]):
                w.writerow([i + 1, f"{lat:.2f}"])
        saved.append(path)

        path = os.path.join(out_dir, f"perf_summary_{ts}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in [
                ("n_requests",    perf_data["n_requests"]),
                ("successful",    len(perf_data["latencies"])),
                ("errors",        perf_data["errors"]),
                ("throughput_rps", f"{perf_data['throughput']:.4f}"),
                ("p50_ms",        f"{perf_data['p50']:.2f}"),
                ("p95_ms",        f"{perf_data['p95']:.2f}"),
                ("p99_ms",        f"{perf_data['p99']:.2f}"),
                ("min_ms",        f"{min(perf_data['latencies']):.2f}"),
                ("max_ms",        f"{max(perf_data['latencies']):.2f}"),
                ("wall_time_s",   f"{perf_data['total_elapsed']:.2f}"),
            ]:
                w.writerow([k, v])
        saved.append(path)

    # ── Batch size variation ──────────────────────────────────────────────────
    if batch_size_data:
        path = os.path.join(out_dir, f"batch_sizes_{ts}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subset", "batch_size", "n_batches", "n_concepts",
                        "avg_latency_ms", "hit_at_1_pct", "hit_at_5_pct", "mrr"])
            for row in batch_size_data.values():
                n = row["n_concepts"] or 1
                w.writerow([
                    row["subset"], row["batch_size"], row["n_batches"], row["n_concepts"],
                    f"{row['avg_latency_ms']:.2f}",
                    f"{row['hit_at'][1]/n*100:.2f}" if row["n_concepts"] else "",
                    f"{row['hit_at'][5]/n*100:.2f}" if row["n_concepts"] else "",
                    f"{row['mrr']:.4f}" if row["n_concepts"] else "",
                ])
        saved.append(path)

    if saved:
        section("CSV files saved")
        for p in saved:
            info(p)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _config_subtitle(cfg: dict) -> str:
    """One-line config string for figure subtitles."""
    if not cfg:
        return ""
    r = cfg.get("retrieval", {})
    rk = cfg.get("reranking", {})
    parts = []
    if rk.get("reranker_type"):
        parts.append(f"reranker={rk['reranker_type']}")
    if r.get("embedding_model"):
        m = r["embedding_model"].split("/")[-1]
        parts.append(f"model={m}")
    if r.get("vector_backend"):
        parts.append(f"backend={r['vector_backend']}")
    return "  |  ".join(parts)


def save_plots(accuracy_data: Optional[dict], perf_data: Optional[dict],
               batch_size_data: Optional[dict], out_dir: str, fmt: str,
               server_config: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        warn("matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    cfg_sub = _config_subtitle(server_config)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _rank_color(rk):
        if rk == 1:   return "#2ecc71"   # green
        if rk <= 3:   return "#f39c12"   # orange
        if rk <= 5:   return "#e67e22"   # amber
        return "#e74c3c"                  # red = not found

    def _rank_distribution(rows):
        """Return individual counts for rank 1, 2, 3, 4, 5, and not_found."""
        counts = {}
        for k in range(1, 6):
            counts[k] = sum(1 for r in rows if r["rank"] != "" and int(r["rank"]) == k)
        counts["nf"] = sum(1 for r in rows if r["not_found"] == "1")
        return counts

    # ── Accuracy figure ───────────────────────────────────────────────────────
    if accuracy_data:
        ep_stats = accuracy_data["ep_stats"]
        overall  = accuracy_data["overall"]
        n_total  = overall["n"]

        # Layout: 3 rows × 2 cols
        # [0,0]  Per-endpoint Hit@k grouped bars
        # [0,1]  Overall Hit@k + MRR (all N queries)
        # [1,0]  /map/concept — queries that missed rank-1
        # [1,1]  /map/search  — queries that missed rank-1
        # [2,0]  Full /map/concept rank distribution (individual ranks)
        # [2,1]  Full /map/search  rank distribution (individual ranks)
        fig = plt.figure(figsize=(16, 20))
        fig.suptitle(
            f"Ontology Mapping API — Accuracy  (N={n_total} queries evaluated)\n{cfg_sub}",
            fontsize=13, fontweight="bold"
        )
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.75, wspace=0.38)
        fig.subplots_adjust(top=0.94)

        # ── [0,0] Per-endpoint Hit@k grouped bars ────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0])
        ep_labels = [
            f"/map/concept\n(n={ep_stats['concept']['n']})",
            f"/map/search\n(n={ep_stats['search']['n']})",
            f"/map/batch\n(n={ep_stats['batch']['n']})",
        ]
        ep_keys  = ["concept", "search", "batch"]
        ks       = [1, 3, 5]
        k_labels = ["Hit@1", "Hit@3", "Hit@5"]
        k_colors = ["#2ecc71", "#3498db", "#9b59b6"]
        x        = np.arange(len(ep_labels))
        width    = 0.25

        for j, (k, lbl, col) in enumerate(zip(ks, k_labels, k_colors)):
            vals = [ep_stats[ep]["hit_at"][k] / ep_stats[ep]["n"] * 100
                    if ep_stats[ep]["n"] else 0 for ep in ep_keys]
            bars = ax0.bar(x + (j - 1) * width, vals, width, label=lbl,
                           color=col, edgecolor="white", linewidth=0.8)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                             f"{v:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax0.set_xticks(x); ax0.set_xticklabels(ep_labels, fontsize=8)
        ax0.set_ylim(0, 118); ax0.set_ylabel("Score (%)")
        ax0.set_title("Hit@k per Endpoint\n(dashed: 50% min threshold  /  80% target)", fontweight="bold")
        ax0.axhline(50, color="gray",      linestyle="--", linewidth=0.7, alpha=0.6)
        ax0.axhline(80, color="steelblue", linestyle="--", linewidth=0.7, alpha=0.6)
        # legend goes in the empty lower-center region (bars all ≥ 70%, so 0–55% is free)
        ax0.legend(fontsize=7, loc="lower center", ncol=3)
        ax0.spines["top"].set_visible(False); ax0.spines["right"].set_visible(False)

        # ── [0,1] Overall Hit@k + MRR ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1])
        metrics    = ["Hit@1", "Hit@3", "Hit@5", "MRR\n(×100 for scale)"]
        ov_vals    = [
            overall["hit_at"][1] / n_total * 100,
            overall["hit_at"][3] / n_total * 100,
            overall["hit_at"][5] / n_total * 100,
            overall["mrr"] * 100,  # scaled to % for visual comparison only
        ]
        thresholds = [50, 70, 80, 50]
        bar_colors = ["#2ecc71" if v >= th else "#e67e22" if v >= th * 0.85 else "#e74c3c"
                      for v, th in zip(ov_vals, thresholds)]
        bars = ax1.bar(metrics, ov_vals, color=bar_colors, edgecolor="white", linewidth=1.2, width=0.55)
        ax1.set_ylim(0, 118); ax1.set_ylabel("Score (%)")
        ax1.set_title(
            f"Overall Metrics  (all {n_total} queries)\n"
            "Pass: Hit@1≥50%  Hit@3≥70%  Hit@5≥80%  MRR≥0.50",
            fontweight="bold"
        )
        ax1.axhline(50, color="gray",      linestyle="--", linewidth=0.8, alpha=0.7)
        ax1.axhline(80, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.7)
        real_vals    = [overall["hit_at"][1]/n_total*100, overall["hit_at"][3]/n_total*100,
                        overall["hit_at"][5]/n_total*100, overall["mrr"]]
        display_fmts = ["{:.1f}%", "{:.1f}%", "{:.1f}%", "MRR={:.3f}"]
        for bar, val, real, dfmt in zip(bars, ov_vals, real_vals, display_fmts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                     dfmt.format(real), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax1.text(0.98, 0.04, "Green=pass  |  Amber=close  |  Red=fail",
                 ha="right", va="bottom", transform=ax1.transAxes, fontsize=7, color="gray", style="italic")
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

        # ── [1,0] /map/concept — queries that missed rank-1 (up to 40) ──────────
        # "missed rank-1" = correct answer was NOT the top result (rank≥2 or not found)
        ax2 = fig.add_subplot(gs[1, 0])
        all_concept_rows = [r for r in accuracy_data["per_query"] if r["endpoint"] == "/map/concept"]
        concept_missed = [r for r in all_concept_rows if r["hit_at_1"] != "1"]
        concept_missed_sorted = sorted(
            concept_missed,
            key=lambda r: int(r["rank"]) if r["rank"] != "" else 99,
            reverse=True
        )[:40]
        if concept_missed_sorted:
            q_labels = [r["query"][:26] for r in concept_missed_sorted]
            ranks    = [int(r["rank"]) if r["rank"] != "" else 6 for r in concept_missed_sorted]
            bar_col  = [_rank_color(rk) for rk in ranks]
            y_pos    = range(len(q_labels))
            ax2.barh(list(y_pos), ranks, color=bar_col, edgecolor="white", linewidth=0.4)
            ax2.set_yticks(list(y_pos))
            ax2.set_yticklabels(q_labels, fontsize=6.5)
            ax2.set_ylabel("Query text (truncated)", fontsize=8)
            ax2.set_xlabel("Position of correct answer in results  (N/F = not in top-5)", fontsize=8)
            n_missed = len(concept_missed)
            ax2.set_title(
                f"/map/concept — {min(40, n_missed)} queries where correct answer\nwas NOT the top result"
                f"  ({n_missed} of {len(all_concept_rows)} total, worst first)",
                fontweight="bold"
            )
            ax2.axvline(1, color="#2ecc71", linestyle="--", linewidth=0.8, alpha=0.6)
            ax2.invert_yaxis(); ax2.set_xlim(0, 7)
            ax2.set_xticks([1, 2, 3, 4, 5, 6])
            ax2.set_xticklabels(["1", "2", "3", "4", "5", "N/F"])
            ax2.text(0.99, 0.01,
                     "Green=rank 1  |  Orange=rank 2–3  |  Amber=rank 4–5  |  Red=not found",
                     ha="right", va="bottom", transform=ax2.transAxes, fontsize=6.5, color="gray", style="italic")
        else:
            ax2.text(0.5, 0.5, "All /map/concept queries returned rank-1 results ✓",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=10)

        # ── [1,1] /map/search — queries that missed rank-1 (up to 40) ─────────
        ax3 = fig.add_subplot(gs[1, 1])
        all_search_rows = [r for r in accuracy_data["per_query"] if r["endpoint"] == "/map/search"]
        search_missed = [r for r in all_search_rows if r["hit_at_1"] != "1"]
        search_missed_sorted = sorted(
            search_missed,
            key=lambda r: int(r["rank"]) if r["rank"] != "" else 99,
            reverse=True
        )[:40]
        if search_missed_sorted:
            q_labels = [r["query"][:26] for r in search_missed_sorted]
            ranks    = [int(r["rank"]) if r["rank"] != "" else 6 for r in search_missed_sorted]
            bar_col  = [_rank_color(rk) for rk in ranks]
            y_pos    = range(len(q_labels))
            ax3.barh(list(y_pos), ranks, color=bar_col, edgecolor="white", linewidth=0.4)
            ax3.set_yticks(list(y_pos))
            ax3.set_yticklabels(q_labels, fontsize=6.5)
            ax3.set_ylabel("Query text (truncated)", fontsize=8)
            ax3.set_xlabel("Position of correct answer in results  (N/F = not in top-5)", fontsize=8)
            n_missed = len(search_missed)
            ax3.set_title(
                f"/map/search — {min(40, n_missed)} queries where correct answer\nwas NOT the top result"
                f"  ({n_missed} of {len(all_search_rows)} total, worst first)",
                fontweight="bold"
            )
            ax3.axvline(1, color="#2ecc71", linestyle="--", linewidth=0.8, alpha=0.6)
            ax3.invert_yaxis(); ax3.set_xlim(0, 7)
            ax3.set_xticks([1, 2, 3, 4, 5, 6])
            ax3.set_xticklabels(["1", "2", "3", "4", "5", "N/F"])
            ax3.text(0.99, 0.01,
                     "Green=rank 1  |  Orange=rank 2–3  |  Amber=rank 4–5  |  Red=not found",
                     ha="right", va="bottom", transform=ax3.transAxes, fontsize=6.5, color="gray", style="italic")
        else:
            ax3.text(0.5, 0.5, "All /map/search queries returned rank-1 results ✓",
                     ha="center", va="center", transform=ax3.transAxes, fontsize=10)

        # ── [2,0] Full /map/concept rank distribution — individual ranks ────────
        ax4 = fig.add_subplot(gs[2, 0])
        if all_concept_rows:
            d = _rank_distribution(all_concept_rows)
            n_c    = len(all_concept_rows)
            cats   = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5", "Not Found"]
            counts = [d[1], d[2], d[3], d[4], d[5], d["nf"]]
            cols   = ["#2ecc71", "#a8d97f", "#f0c040", "#e67e22", "#c0392b", "#e74c3c"]
            bars   = ax4.bar(cats, counts, color=cols, edgecolor="white", linewidth=1.0, width=0.6)
            ax4.set_ylabel("Number of queries", fontsize=9)
            ax4.set_xlabel("Rank position of correct answer", fontsize=9)
            ax4.set_title(
                f"/map/concept — rank distribution  (all {n_c} queries)",
                fontweight="bold"
            )
            max_c = max(counts)
            for bar, cnt in zip(bars, counts):
                if cnt == 0:
                    continue
                pct = cnt / n_c * 100
                # Place label above bar; if bar is tiny, raise the label further
                label_y = bar.get_height() + max_c * 0.015
                ax4.text(bar.get_x() + bar.get_width() / 2, label_y,
                         f"{cnt}\n({pct:.1f}%)",
                         ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            ax4.set_ylim(0, max_c * 1.28)
            ax4.tick_params(axis="x", labelsize=8.5)
            ax4.spines["top"].set_visible(False)
            ax4.spines["right"].set_visible(False)
        else:
            ax4.text(0.5, 0.5, "No /map/concept entries", ha="center", va="center", transform=ax4.transAxes)

        # ── [2,1] Full /map/search rank distribution — individual ranks ─────────
        ax5 = fig.add_subplot(gs[2, 1])
        if all_search_rows:
            d = _rank_distribution(all_search_rows)
            n_s    = len(all_search_rows)
            cats   = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5", "Not Found"]
            counts = [d[1], d[2], d[3], d[4], d[5], d["nf"]]
            cols   = ["#2ecc71", "#a8d97f", "#f0c040", "#e67e22", "#c0392b", "#e74c3c"]
            bars   = ax5.bar(cats, counts, color=cols, edgecolor="white", linewidth=1.0, width=0.6)
            ax5.set_ylabel("Number of queries", fontsize=9)
            ax5.set_xlabel("Rank position of correct answer", fontsize=9)
            ax5.set_title(
                f"/map/search — rank distribution  (all {n_s} queries)",
                fontweight="bold"
            )
            max_s = max(counts)
            for bar, cnt in zip(bars, counts):
                if cnt == 0:
                    continue
                pct = cnt / n_s * 100
                label_y = bar.get_height() + max_s * 0.015
                ax5.text(bar.get_x() + bar.get_width() / 2, label_y,
                         f"{cnt}\n({pct:.1f}%)",
                         ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            ax5.set_ylim(0, max_s * 1.28)
            ax5.tick_params(axis="x", labelsize=8.5)
            ax5.spines["top"].set_visible(False)
            ax5.spines["right"].set_visible(False)
        else:
            ax5.text(0.5, 0.5, "No /map/search entries", ha="center", va="center", transform=ax5.transAxes)

        path = os.path.join(out_dir, f"accuracy_{ts}.{fmt}")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        saved.append(path)
        info(f"Accuracy plot → {path}")

    # ── Performance figure ────────────────────────────────────────────────────
    if perf_data:
        latencies = perf_data["latencies"]
        n_req     = len(latencies)
        n_sent    = perf_data["n_requests"]
        seq_times = perf_data.get("seq_times", [])

        fig = plt.figure(figsize=(14, 7.5))
        cfg_line = f"  [{cfg_sub}]" if cfg_sub else ""
        fig.suptitle(
            f"Ontology Mapping API — Performance Benchmark{cfg_line}",
            fontsize=12, fontweight="bold"
        )
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42)
        fig.subplots_adjust(top=0.88, bottom=0.14)

        # ── Left: latency histogram ───────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        # Bin count: sqrt rule gives ~10 bins for 100 requests, clean histogram
        n_bins = max(8, min(20, int(np.sqrt(n_req))))
        ax1.hist(latencies, bins=n_bins, color="#3498db", edgecolor="white", linewidth=0.8,
                 label=f"{n_req} concurrent requests")
        # Add mean line
        lat_mean = statistics.mean(latencies)
        ax1.axvline(lat_mean, color="#2c3e50", linestyle="-", linewidth=1.4,
                    label=f"mean = {lat_mean:.0f} ms")
        for pct, col, lbl in [
            (perf_data["p50"], "#2ecc71", f"p50 (median) = {perf_data['p50']:.0f} ms"),
            (perf_data["p95"], "#e67e22", f"p95 = {perf_data['p95']:.0f} ms"),
            (perf_data["p99"], "#e74c3c", f"p99 = {perf_data['p99']:.0f} ms"),
        ]:
            ax1.axvline(pct, color=col, linestyle="--", linewidth=1.8, label=lbl)
        if seq_times:
            seq_avg = statistics.mean(seq_times)
            ax1.axvline(seq_avg, color="#9b59b6", linestyle=":", linewidth=1.8,
                        label=f"sequential avg = {seq_avg:.0f} ms")
        ax1.set_xlabel("End-to-end latency (ms)", fontsize=10)
        ax1.set_ylabel("Number of requests", fontsize=10)
        ax1.set_title(
            f"Latency Distribution — POST /map/concept\n"
            f"{n_req} concurrent requests  |  concurrency = {perf_data.get('concurrency', 4)}  "
            f"|  {n_bins} bins",
            fontweight="bold", pad=10
        )
        ax1.legend(fontsize=8, loc="upper left")
        # Note at bottom — below the bars, not obscuring them
        note = ("Each bar = number of requests completing in that latency range.  "
                "Dotted purple = sequential (single-thread) baseline for comparison.")
        ax1.text(0.5, -0.14, note, ha="center", va="top",
                 transform=ax1.transAxes, fontsize=7, color="gray", style="italic")

        # ── Right: summary table ──────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
        seq_avg_str = f"{statistics.mean(seq_times):.0f} ms" if seq_times else "—"
        tdata = [
            ["Metric",           "Value"],
            ["Test endpoint",    "POST /map/concept"],
            ["Query type",       "Single-term (max_results=5)"],
            ["Requests sent",    str(n_sent)],
            ["Successful",       str(n_req)],
            ["Errors",           str(perf_data["errors"])],
            ["Concurrency",      str(perf_data.get("concurrency", 4))],
            ["Throughput",       f"{perf_data['throughput']:.2f} req/s"],
            ["Sequential avg",   seq_avg_str],
            ["Concurrent p50",   f"{perf_data['p50']:.0f} ms"],
            ["Concurrent p95",   f"{perf_data['p95']:.0f} ms"],
            ["Concurrent p99",   f"{perf_data['p99']:.0f} ms"],
            ["Min latency",      f"{min(latencies):.0f} ms"],
            ["Max latency",      f"{max(latencies):.0f} ms"],
            ["Wall time",        f"{perf_data['total_elapsed']:.1f} s"],
        ]
        tbl = ax2.table(cellText=tdata[1:], colLabels=tdata[0],
                        loc="center", cellLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.3, 1.55)
        # bold the header row
        for j in range(2):
            tbl[0, j].set_text_props(fontweight="bold")
        ax2.set_title("Benchmark Summary", fontweight="bold", pad=20)

        path = os.path.join(out_dir, f"performance_{ts}.{fmt}")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        saved.append(path)
        info(f"Performance plot → {path}")

    # ── Batch size figure ─────────────────────────────────────────────────────
    if batch_size_data:
        # Group by subset
        subsets_seen: Dict[str, list] = {}
        for row in batch_size_data.values():
            subsets_seen.setdefault(row["subset"], []).append(row)
        for rows in subsets_seen.values():
            rows.sort(key=lambda x: x["batch_size"])

        n_subsets = len(subsets_seen)
        fig, axes = plt.subplots(n_subsets, 2, figsize=(13, 5 * n_subsets + 1))
        if n_subsets == 1:
            axes = [axes]
        cfg_line = f"  [{cfg_sub}]" if cfg_sub else ""
        fig.suptitle(
            f"Ontology Mapping API — POST /map/batch  |  Batch Size Variation{cfg_line}",
            fontsize=12, fontweight="bold"
        )
        fig.subplots_adjust(top=0.93, hspace=0.55)

        subset_colors = ["#3498db", "#e67e22", "#2ecc71"]
        for row_idx, (subset_name, rows) in enumerate(subsets_seen.items()):
            sizes  = [r["batch_size"] for r in rows]
            lats   = [r["avg_latency_ms"] for r in rows]
            h1s    = [r["hit_at"][1] / r["n_concepts"] * 100 if r["n_concepts"] else 0 for r in rows]
            h5s    = [r["hit_at"][5] / r["n_concepts"] * 100 if r["n_concepts"] else 0 for r in rows]
            n_vals = [r["n_concepts"] for r in rows]
            nb_vals= [r["n_batches"]  for r in rows]
            col    = subset_colors[row_idx % len(subset_colors)]

            # Left: latency vs batch size
            ax_lat = axes[row_idx][0]
            ax_lat.plot(sizes, lats, "o-", color=col, linewidth=2.2, markersize=8)
            # Stagger annotation offsets to avoid overlap on closely-spaced points
            offsets = [(6, 8), (6, -18), (6, 8), (6, -18)]
            for i, (x_v, y_v, n_c, n_b) in enumerate(zip(sizes, lats, n_vals, nb_vals)):
                ox, oy = offsets[i % len(offsets)]
                ax_lat.annotate(
                    f"{y_v:.0f} ms  ({n_b} req × {n_c} concepts)",
                    (x_v, y_v), textcoords="offset points", xytext=(ox, oy),
                    fontsize=7, color="#444",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6)
                )
            ax_lat.set_xlabel("Batch size (concepts per /map/batch request)", fontsize=9)
            ax_lat.set_ylabel("Avg wall-time per request (ms)", fontsize=9)
            ax_lat.set_title(f"Latency vs Batch Size — {subset_name}", fontweight="bold")
            ax_lat.set_xticks(sizes)
            # Extend y-axis top by 20% so annotations don't clip
            ax_lat.set_ylim(0, max(lats) * 1.35)
            ax_lat.text(0.02, 0.97,
                        "Each data point = average over all batches of that size.",
                        ha="left", va="top", transform=ax_lat.transAxes,
                        fontsize=7, color="gray", style="italic")

            # Right: Hit@1 and Hit@5 vs batch size — zoom to data range
            ax_acc = axes[row_idx][1]
            ax_acc.plot(sizes, h1s, "o-", color="#2ca02c", linewidth=2.2, markersize=8, label="Hit@1")
            ax_acc.plot(sizes, h5s, "s-", color="#1f77b4", linewidth=2.2, markersize=8, label="Hit@5")
            all_acc = h1s + h5s
            y_lo = max(0,   min(all_acc) - 8)
            y_hi = min(100, max(all_acc) + 8)
            for x_v, h1, h5 in zip(sizes, h1s, h5s):
                ax_acc.text(x_v, h1 + (y_hi - y_lo) * 0.02,
                            f"{h1:.1f}%", ha="center", fontsize=7, color="#2ca02c", fontweight="bold")
                ax_acc.text(x_v, h5 - (y_hi - y_lo) * 0.05,
                            f"{h5:.1f}%", ha="center", fontsize=7, color="#1f77b4", fontweight="bold")
            ax_acc.set_xlabel("Batch size", fontsize=9)
            ax_acc.set_ylabel("Accuracy (%)", fontsize=9)
            ax_acc.set_title(f"Hit@1 and Hit@5 vs Batch Size — {subset_name}", fontweight="bold")
            ax_acc.set_ylim(y_lo, y_hi); ax_acc.set_xticks(sizes)
            ax_acc.legend(fontsize=9, loc="lower right")
            if 50 > y_lo:
                ax_acc.axhline(50, color="gray",      linestyle="--", linewidth=0.7, alpha=0.6, label="50%")
            if 80 > y_lo:
                ax_acc.axhline(80, color="steelblue", linestyle="--", linewidth=0.7, alpha=0.6, label="80%")
            ax_acc.text(0.02, 0.03,
                        "Accuracy should stay flat — same concepts, different batch grouping.\n"
                        "Visible variation may indicate ordering or edge effects.",
                        ha="left", va="bottom", transform=ax_acc.transAxes,
                        fontsize=7, color="gray", style="italic")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        path = os.path.join(out_dir, f"batch_sizes_{ts}.{fmt}")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        saved.append(path)
        info(f"Batch size plot → {path}")

    if saved:
        print(f"\n  Plots saved to: {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test the Ontology Mapping API")
    parser.add_argument("--url",           default="http://localhost:8000")
    parser.add_argument("--verbose",       action="store_true", help="Print per-query accuracy results")
    parser.add_argument("--wait",          action="store_true", help="Poll /health until indexing is complete")
    parser.add_argument("--accuracy",      action="store_true", help="Run accuracy evaluation (Hit@k, MRR) per endpoint")
    parser.add_argument("--perf",          action="store_true", help="Run performance benchmark")
    parser.add_argument("--batch-sizes",   action="store_true", help="Run batch size variation test")
    parser.add_argument("--all",           action="store_true", help="Run all tests")
    parser.add_argument("--max-accuracy",  type=int, default=None, help="Limit accuracy to N single queries (smoke-test)")
    parser.add_argument("--n-requests",    type=int, default=100,  help="Requests for perf benchmark (default 100)")
    parser.add_argument("--concurrency",   type=int, default=4,    help="Concurrent threads for perf benchmark (default 4)")
    parser.add_argument("--plot",          action="store_true",    help="Save plots")
    parser.add_argument("--csv",           action="store_true",    help="Save CSV files with all results")
    parser.add_argument("--plot-format",   default="png", choices=["png", "pdf"])
    parser.add_argument("--plot-dir",      default="test_results", help="Output directory for plots/CSV (default: test_results/)")
    args = parser.parse_args()

    if args.all:
        args.accuracy = args.perf = args.batch_sizes = True

    url = args.url.rstrip("/")
    print(f"Testing API: {url}")

    # Fetch server config early for plot/CSV metadata
    server_config = fetch_config(url)
    if server_config:
        rc = server_config.get("reranking", {})
        rv = server_config.get("retrieval", {})
        info(f"Server config: reranker={rc.get('reranker_type','?')}  "
             f"model={rv.get('embedding_model','?')}  backend={rv.get('vector_backend','?')}")

    indexing_ready = test_health(url, args.verbose)

    if args.wait and not indexing_ready:
        print(f"\n  Waiting for indexing (polling every 30s)...")
        while not indexing_ready:
            time.sleep(30)
            r = get(url, "/health")
            if r and r.status_code == 200:
                indexing_ready = r.json().get("indexing_complete", False)
                print(f"  [{INFO}] indexing_complete={indexing_ready}")
        print(f"  [{PASS}] Indexing complete")

    test_root(url, args.verbose)
    test_ontologies(url, args.verbose)
    test_stats(url, args.verbose)

    if indexing_ready:
        test_single_concept(url, args.verbose)
        test_contextual_search(url, args.verbose)
        test_batch(url, args.verbose)
        test_edge_cases(url, args.verbose)

        accuracy_data   = None
        perf_data       = None
        batch_size_data = None

        if args.accuracy:
            accuracy_data = test_accuracy(url, args.verbose, max_queries=args.max_accuracy)

        if args.perf:
            perf_data = test_performance(url, args.n_requests, args.concurrency)

        if args.batch_sizes:
            batch_size_data = test_batch_sizes(url, GOLDEN.get("batch", []))

        if args.plot and (accuracy_data or perf_data or batch_size_data):
            section("Saving plots")
            save_plots(accuracy_data, perf_data, batch_size_data,
                       args.plot_dir, args.plot_format, server_config)

        if args.csv and (accuracy_data or perf_data or batch_size_data):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv(accuracy_data, perf_data, batch_size_data,
                     args.plot_dir, ts, server_config)
    else:
        print(f"\n  [{SKIP}] Search tests skipped — indexing not complete (use --wait)")

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"RESULT: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  ALL PASSED")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
