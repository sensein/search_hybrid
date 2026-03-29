# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DISCLAIMER: This software is provided "as is" without any warranty,
# express or implied, including but not limited to the warranties of
# merchantability, fitness for a particular purpose, and non-infringement.
#
# In no event shall the authors or copyright holders be liable for any
# claim, damages, or other liability, whether in an action of contract,
# tort, or otherwise, arising from, out of, or in connection with the
# software or the use or other dealings in the software.
# -----------------------------------------------------------------------------

# @Author  : Tek Raj Chhetri
# @Email   : tekraj@mit.edu
# @Web     : https://tekrajchhetri.com/
# @File    : evaluate_performance.py
# @Software: PyCharm

#!/usr/bin/env python3
"""
Performance benchmark (plots enabled by default).

All queries from the golden set are flattened into a single list.
Single requests (/map/concept, /map/search) sample from that list.
Batch requests (/map/batch) chunk the list at --batch-chunk-size.

Timeout scales automatically with batch chunk size:
    timeout = max(--timeout, batch_chunk_size * 25 + 30)

Defaults:
  concurrency:       1,2,4
  single requests:   200  (split evenly: 100 concept + 100 search)
  batch requests:    50
  batch chunk size:  5
  base timeout:      60 s
  warmup requests:   3
  seed:              42

Throughput is measured as isolated endpoint throughput:
    requests / (last_finish - first_start) for each endpoint group.
This is more accurate than dividing by total round wall-time when
endpoint groups finish at very different times.

Error breakdown distinguishes:
  http_error        — server responded 4xx/5xx
  timeout           — request exceeded the configured timeout
  connection_error  — TCP connection refused / reset
  other             — any other exception

Install orjson for faster JSON I/O (optional):
    pip install orjson
"""

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

# orjson is ~3-5x faster than stdlib json for both serialisation and
# deserialisation.  Use it when available; fall back to stdlib silently.
try:
    import orjson as _json_lib

    def _dumps_pretty(obj: Any) -> str:
        return _json_lib.dumps(obj, option=_json_lib.OPT_INDENT_2).decode()

    def _dumps_bytes(obj: Any) -> bytes:
        return _json_lib.dumps(obj)

    _loads = _json_lib.loads
    _ORJSON = True
except ImportError:
    def _dumps_pretty(obj: Any) -> str:  # type: ignore[misc]
        return json.dumps(obj, indent=2)

    def _dumps_bytes(obj: Any) -> bytes:  # type: ignore[misc]
        return json.dumps(obj).encode()

    _loads = json.loads  # type: ignore[assignment]
    _ORJSON = False


# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Result:
    endpoint:     str
    concurrency:  int
    ok:           bool
    latency_ms:   float
    started_at:   float   # perf_counter timestamp — used for isolated throughput
    finished_at:  float   # perf_counter timestamp — used for isolated throughput
    status_code:  int     # HTTP status, or 0 for non-HTTP failures
    error_type:   str     # "", "http_error", "timeout", "connection_error", "other"


# ──────────────────────────────────────────────────────────────────────────────
# GOLDEN SET LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_golden(path: str) -> dict[str, Any]:
    return _loads(Path(path).read_bytes())


def flatten_queries(golden: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every query text from both the 'single' and 'batch' sections."""
    queries: list[dict[str, Any]] = []

    for entry in golden.get("single", []):
        queries.append({"text": entry["query"], "context": entry.get("context")})

    for group in golden.get("batch", []):
        for q in group.get("queries", []):
            queries.append({"text": q["text"], "context": q.get("context")})

    if not queries:
        raise ValueError("Golden set is empty — no queries to benchmark.")

    return queries


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST BUILDING
# ──────────────────────────────────────────────────────────────────────────────

def build_requests(
    queries: list[dict[str, Any]],
    seed: int,
    single_requests: int,
    batch_requests: int,
    batch_chunk_size: int,
) -> list[tuple[str, bytes, str]]:
    """
    Return a shuffled list of (route, payload_bytes, endpoint_label) tuples.

    Payloads are pre-serialised to bytes so the hot path (call()) does no JSON
    encoding at dispatch time — particularly valuable under high concurrency.

    Args:
        seed: RNG seed; use a fixed value for reproducible request sequences.
    """
    rng = random.Random(seed)
    reqs: list[tuple[str, bytes, str]] = []

    # Single requests: alternate concept / search
    single_routes = [("/map/concept", "concept"), ("/map/search", "search")]
    for i in range(single_requests):
        q = rng.choice(queries)
        route, ep_label = single_routes[i % 2]
        payload: dict[str, Any] = {"text": q["text"], "max_results": 10}
        if q.get("context"):
            payload["context"] = q["context"]
        reqs.append((route, _dumps_bytes(payload), ep_label))

    # Batch requests: fixed-size chunks cycled as needed
    shuffled = list(queries)
    rng.shuffle(shuffled)
    chunks = [shuffled[i:i + batch_chunk_size] for i in range(0, len(shuffled), batch_chunk_size)]
    for i in range(batch_requests):
        chunk = chunks[i % len(chunks)]
        payload = {
            "text": [{"text": q["text"], "context": q.get("context")} for q in chunk],
            "max_results": 10,
        }
        reqs.append(("/map/batch", _dumps_bytes(payload), "batch"))

    rng.shuffle(reqs)
    return reqs


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

_JSON_HEADERS = {"Content-Type": "application/json"}


async def call(
    session: aiohttp.ClientSession,
    route: str,
    payload_bytes: bytes,
    endpoint: str,
    concurrency: int,
    semaphore: asyncio.Semaphore,
) -> Result:
    """
    Fire one HTTP POST and return a Result.

    started_at is recorded inside the semaphore so it reflects the time the
    request actually left the client (not the time it spent waiting to acquire
    the semaphore slot), giving accurate latency figures under concurrency.
    """
    async with semaphore:
        started_at = time.perf_counter()
        status_code = 0
        error_type  = ""
        ok          = False

        try:
            async with session.post(route, data=payload_bytes, headers=_JSON_HEADERS) as resp:
                await resp.read()               # consume body so connection is reusable
                status_code = resp.status
                ok          = 200 <= status_code < 300
                if not ok:
                    error_type = "http_error"

        except asyncio.TimeoutError:
            error_type = "timeout"
        except aiohttp.ClientConnectionError:
            error_type = "connection_error"
        except Exception:
            error_type = "other"

        finished_at = time.perf_counter()

    return Result(
        endpoint    = endpoint,
        concurrency = concurrency,
        ok          = ok,
        latency_ms  = (finished_at - started_at) * 1000.0,
        started_at  = started_at,
        finished_at = finished_at,
        status_code = status_code,
        error_type  = error_type,
    )


async def run_round(
    base_url: str,
    reqs: list[tuple[str, bytes, str]],
    concurrency: int,
    timeout_s: int,
) -> list[Result]:
    semaphore = asyncio.Semaphore(concurrency)
    timeout   = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(
        limit          = concurrency,
        limit_per_host = concurrency,
        ttl_dns_cache  = 300,
    )

    async with aiohttp.ClientSession(
        base_url  = base_url,
        timeout   = timeout,
        connector = connector,
    ) as session:
        tasks = [
            asyncio.create_task(call(session, route, payload_bytes, endpoint, concurrency, semaphore))
            for route, payload_bytes, endpoint in reqs
        ]
        return await asyncio.gather(*tasks)


async def warmup(
    base_url: str,
    queries: list[dict[str, Any]],
    timeout_s: int,
    n: int,
    seed: int,
) -> None:
    """
    Fire n un-timed requests to warm up the service before measuring.

    Covers all three endpoint types so caches, thread pools, and JIT paths
    are primed before the benchmark starts.
    """
    if n <= 0:
        return

    reqs = build_requests(
        queries           = queries,
        seed              = seed,
        single_requests   = max(2, n - 1),   # at least one concept + one search
        batch_requests    = 1,
        batch_chunk_size  = 3,
    )
    reqs = reqs[:n]

    print(f"  Warming up with {len(reqs)} requests …", flush=True)
    semaphore = asyncio.Semaphore(len(reqs))   # all at once — we don't care about results
    connector = aiohttp.TCPConnector(limit=len(reqs), ttl_dns_cache=300)
    async with aiohttp.ClientSession(
        base_url  = base_url,
        timeout   = aiohttp.ClientTimeout(total=timeout_s),
        connector = connector,
    ) as session:
        tasks = [
            asyncio.create_task(call(session, route, payload_bytes, endpoint, 0, semaphore))
            for route, payload_bytes, endpoint in reqs
        ]
        await asyncio.gather(*tasks)
    print("  Warmup complete.\n", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

def _percentiles(vals: list[float]) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0

    def interp(p: float) -> float:
        k = (len(vals) - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, len(vals) - 1)
        return vals[f] + (vals[c] - vals[f]) * (k - f)

    return interp(50), interp(95), interp(99)


def aggregate(results: list[Result], round_duration_s: float) -> list[dict[str, Any]]:
    """
    Compute per-endpoint statistics.

    Throughput is calculated as isolated endpoint throughput:
        len(items) / (last_finish - first_start)

    This avoids the misleading inflation that occurs when a fast endpoint
    (concept) shares the round wall-time with a slow one (batch).

    The round_duration_s is retained for the "overall" row only, which
    measures total system throughput across all endpoints.
    """
    grouped: dict[tuple[str, int], list[Result]] = defaultdict(list)
    for r in results:
        grouped[(r.endpoint, r.concurrency)].append(r)

    output: list[dict[str, Any]] = []

    for (endpoint, conc), items in sorted(grouped.items()):
        latencies = sorted(r.latency_ms for r in items)
        p50, p95, p99 = _percentiles(latencies)
        success = sum(1 for r in items if r.ok)

        # Isolated throughput: span from first request dispatch to last response
        first_start  = min(r.started_at  for r in items)
        last_finish  = max(r.finished_at for r in items)
        ep_duration  = max(last_finish - first_start, 1e-9)

        # Error breakdown
        error_counts: dict[str, int] = defaultdict(int)
        for r in items:
            if r.error_type:
                error_counts[r.error_type] += 1

        output.append({
            "endpoint":    endpoint,
            "concurrency": conc,
            "requests":    len(items),
            "throughput":  len(items) / ep_duration,      # isolated endpoint req/s
            "error_rate":  1.0 - (success / len(items)),
            "errors":      dict(error_counts),            # breakdown by failure mode
            "p50_ms":      p50,
            "p95_ms":      p95,
            "p99_ms":      p99,
        })

    # Overall row: total requests / total round wall-time
    if results:
        conc = results[0].concurrency
        total_success = sum(1 for r in results if r.ok)
        all_latencies = sorted(r.latency_ms for r in results)
        p50, p95, p99 = _percentiles(all_latencies)
        all_errors: dict[str, int] = defaultdict(int)
        for r in results:
            if r.error_type:
                all_errors[r.error_type] += 1
        output.append({
            "endpoint":    "overall",
            "concurrency": conc,
            "requests":    len(results),
            "throughput":  len(results) / round_duration_s,
            "error_rate":  1.0 - (total_success / len(results)),
            "errors":      dict(all_errors),
            "p50_ms":      p50,
            "p95_ms":      p95,
            "p99_ms":      p99,
        })

    return output


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot(results: list[dict[str, Any]], metric: str, ylabel: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    endpoints = sorted({r["endpoint"] for r in results})
    for ep in endpoints:
        data = sorted(
            (r for r in results if r["endpoint"] == ep),
            key=lambda x: x["concurrency"],
        )
        plt.plot(
            [d["concurrency"] for d in data],
            [d[metric] for d in data],
            marker="o",
            label=ep,
        )
    plt.xlabel("Concurrency")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

async def main(
    base_url:          str,
    golden_path:       str,
    out_dir:           str,
    concurrency_levels: list[int],
    single_requests:   int,
    batch_requests:    int,
    batch_chunk_size:  int,
    base_timeout:      int,
    warmup_requests:   int,
    seed:              int,
    make_plots:        bool,
) -> None:
    timeout_s = max(base_timeout, batch_chunk_size * 25 + 30)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    golden  = load_golden(golden_path)
    queries = flatten_queries(golden)

    print(
        f"Loaded {len(queries)} queries | "
        f"batch_chunk_size={batch_chunk_size} | "
        f"timeout={timeout_s}s | "
        f"concurrency={concurrency_levels} | "
        f"seed={seed} | "
        f"orjson={'yes' if _ORJSON else 'no (pip install orjson for faster I/O)'}"
    )

    await warmup(base_url, queries, timeout_s, warmup_requests, seed=seed)

    all_results: list[dict[str, Any]] = []

    for c in concurrency_levels:
        # Use seed + c so each concurrency level gets a distinct but reproducible
        # request sequence, making cross-run comparisons fair.
        reqs = build_requests(
            queries          = queries,
            seed             = seed + c,
            single_requests  = single_requests,
            batch_requests   = batch_requests,
            batch_chunk_size = batch_chunk_size,
        )

        print(f"\n=== concurrency {c} | {len(reqs)} requests ===")
        start    = time.perf_counter()
        results  = await run_round(base_url, reqs, c, timeout_s)
        duration = time.perf_counter() - start

        agg = aggregate(results, duration)
        all_results.extend(agg)

        for row in agg:
            ep   = row["endpoint"]
            rps  = row["throughput"]
            p95  = row["p95_ms"]
            err  = row["error_rate"] * 100
            errs = row["errors"]
            print(
                f"  {ep:<12}  rps={rps:6.1f}  p95={p95:8.1f}ms  "
                f"err={err:5.1f}%  {errs if errs else ''}"
            )

    (out / "results.json").write_text(_dumps_pretty(all_results))

    if make_plots:
        plot(all_results, "throughput", "Throughput (req/sec)",  out / "throughput.png")
        plot(all_results, "p95_ms",     "P95 Latency (ms)",      out / "latency_p95.png")
        plot(all_results, "error_rate", "Error Rate",            out / "error_rate.png")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HTTP performance benchmark for the ontology mapping API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url",         default="http://localhost:8000")
    parser.add_argument("--golden",           default="golden_set.json")
    parser.add_argument("--out-dir",          default="perf_results")
    parser.add_argument("--concurrency",      default="1,2,4",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--single-requests",  type=int, default=200,
                        help="Single-endpoint requests per round (split evenly concept/search)")
    parser.add_argument("--batch-requests",   type=int, default=50,
                        help="Batch requests per round")
    parser.add_argument("--batch-chunk-size", type=int, default=5,
                        help="Concepts per /map/batch request")
    parser.add_argument("--timeout",          type=int, default=60,
                        help="Base timeout (s); auto-scaled for large batch chunks")
    parser.add_argument("--warmup",           type=int, default=3,
                        help="Un-timed warmup requests before measuring (0 to skip)")
    parser.add_argument("--seed",             type=int, default=42,
                        help="RNG seed for reproducible request sequences")
    parser.add_argument("--no-plots",         action="store_true")
    args = parser.parse_args()

    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]

    asyncio.run(
        main(
            base_url          = args.base_url,
            golden_path       = args.golden,
            out_dir           = args.out_dir,
            concurrency_levels= concurrency_levels,
            single_requests   = args.single_requests,
            batch_requests    = args.batch_requests,
            batch_chunk_size  = args.batch_chunk_size,
            base_timeout      = args.timeout,
            warmup_requests   = args.warmup,
            seed              = args.seed,
            make_plots        = not args.no_plots,
        )
    )
