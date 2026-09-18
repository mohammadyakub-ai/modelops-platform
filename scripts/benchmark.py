"""Reproducible serving benchmark. Run against a live uvicorn instance:

    uvicorn src.serving.api:app --port 8000 &
    python scripts/benchmark.py --n 2000 --concurrency 8

Prints (and optionally writes) a JSON summary: latency percentiles over the
full HTTP round-trip, sequential throughput, and concurrent throughput.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time

import requests

FEATURES = ["age", "income", "credit_score", "loan_amount", "employment_years", "num_defaults", "has_collateral"]
ROW = {"age": 45, "income": 72_000, "credit_score": 680, "loan_amount": 24_000,
       "employment_years": 8, "num_defaults": 1, "has_collateral": 1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=None, help="path to write the JSON summary")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"
    payload = {f: ROW[f] for f in FEATURES}

    def one(_: int) -> float:
        t0 = time.perf_counter()
        requests.post(f"{base}/predict", json=payload, timeout=10).raise_for_status()
        return time.perf_counter() - t0

    for _ in range(args.warmup):
        one(0)

    lats = []
    t0 = time.perf_counter()
    for i in range(args.n):
        lats.append(one(i))
    seq_dur = time.perf_counter() - t0

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(one, range(args.n)))
    conc_dur = time.perf_counter() - t0

    lats.sort()
    summary = {
        "model": "modelops_logistic_regression",
        "requests": args.n,
        "concurrency": args.concurrency,
        "readiness_checks": {"executed": args.warmup},
        "latency_ms": {
            "avg": round(statistics.fmean(lats) * 1000, 3),
            "p50": round(lats[int(len(lats) * 0.50)] * 1000, 3),
            "p95": round(lats[int(len(lats) * 0.95)] * 1000, 3),
            "p99": round(lats[int(len(lats) * 0.99)] * 1000, 3),
        },
        "throughput_sequential_req_per_s": round(args.n / seq_dur, 1),
        "throughput_concurrent_req_per_s": round(args.n / conc_dur, 1),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()