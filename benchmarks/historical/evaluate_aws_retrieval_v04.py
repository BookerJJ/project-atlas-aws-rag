#!/usr/bin/env python3
"""Run the frozen Atlas v0.4 retrieval evaluation from the project root."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

EVAL_FILE = "data/evaluation/frozen/eval_cases_v01_full.jsonl"
BASELINE_FILE = "results/retrieval_eval_v01_full_selective_intent_rerank_v04_databricks.json"
OUTPUT_STEM = "retrieval_eval_v01_full_selective_intent_rerank_v04_aws"
RESULT_FIELDS = (
    "chunk_id", "document_id", "section", "status", "date", "score",
    "source_rank", "dense_rank", "lexical_rank", "dense_score", "lexical_score",
    "coverage_identifiers", "coverage_injected",
)
SUMMARY_METRICS = ("recall_at_1", "recall_at_5", "mrr", "acceptable_hit_at_5", "acceptable_mrr")
CASE_COUNT = 50
DIM = 1024


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def load_inputs(eval_path, baseline_path):
    cases = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline = load_json(baseline_path)
    if baseline.get("dataset_version") != "v0.1-full" or baseline.get("dense_backend") != "Databricks AI Search ANN":
        raise ValueError("Baseline must be the frozen Databricks Track A retrieval result")
    reference = {case["query_id"]: case for case in baseline["cases"]}
    ids = [case["query_id"] for case in cases]
    if (len(ids) != CASE_COUNT or len(set(ids)) != CASE_COUNT
            or baseline.get("case_count") != CASE_COUNT or len(baseline["cases"]) != CASE_COUNT
            or set(ids) != set(reference)):
        raise ValueError("Frozen evaluation and Track A baseline must contain the same 50 unique query IDs")
    for case in cases:
        before = reference[case["query_id"]]
        for key in ("query", "expected_chunk_ids", "acceptable_supporting_chunk_ids"):
            if case.get(key, []) != before.get(key, []):
                raise ValueError(f"Frozen case differs from Track A: {case['query_id']} ({key})")
        if not case["expected_chunk_ids"] or len(before["ranked_chunk_ids"]) != 5:
            raise ValueError(f"Missing labels or baseline Top 5 for {case['query_id']}")
    return cases, baseline, reference


def score_query(ranked_ids, case, metrics_module):
    expected = case["expected_chunk_ids"]
    acceptable = list(dict.fromkeys(expected + case.get("acceptable_supporting_chunk_ids", [])))
    scores = {}
    for k in (1, 3, 5):
        scores[f"recall_at_{k}"] = metrics_module.recall_at_k(ranked_ids, expected, k)
        scores[f"hit_at_{k}"] = metrics_module.hit_at_k(ranked_ids, expected, k)
        scores[f"acceptable_hit_at_{k}"] = metrics_module.hit_at_k(ranked_ids, acceptable, k)
    scores["mrr"] = metrics_module.reciprocal_rank(ranked_ids, expected)
    scores["acceptable_mrr"] = metrics_module.reciprocal_rank(ranked_ids, acceptable)
    return {key: float(value) for key, value in scores.items()}


def invoke(cli, args, directory, query, vector):
    payload_path = directory / "request.json"
    response_path = directory / "response.json"
    payload_path.write_text(json.dumps({
        "mode": "v04", "query": query, "top_k": 5, "query_embedding": vector,
    }, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    response_path.unlink(missing_ok=True)
    cmd = [
        cli, "lambda", "invoke", "--region", args.region, "--function-name", args.function,
        "--invocation-type", "RequestResponse", "--output", "json", "--no-cli-pager",
        "--no-cli-auto-prompt", "--payload", f"fileb://{payload_path}", str(response_path),
    ]
    if args.profile:
        cmd += ["--profile", args.profile]
    start = time.perf_counter()
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=100)
    cli_ms = (time.perf_counter() - start) * 1000
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"AWS CLI exit {completed.returncode}")
    meta = json.loads(completed.stdout)
    if meta.get("StatusCode") != 200 or meta.get("FunctionError"):
        raise RuntimeError(f"Lambda invoke failed (status={meta.get('StatusCode')}, FunctionError={meta.get('FunctionError')})")
    response = load_json(response_path)
    raw = response["body"]
    body = json.loads(raw) if isinstance(raw, str) else raw
    if response.get("statusCode") != 200 or body.get("status") != "success":
        raise RuntimeError(f"Lambda v0.4 response failed: HTTP {response.get('statusCode')} {body.get('error_code', '')}")
    if (body.get("mode") != "v04" or body.get("chunks_loaded") != 128
            or body.get("embedding_dimension") != DIM or body.get("route") not in ("dense", "identifier_hybrid")):
        raise ValueError("Unexpected Lambda mode, route, or snapshot metadata")
    return body, cli_ms, meta.get("ExecutedVersion")


def mean_metrics(rows):
    return {key: statistics.fmean(row["metrics"][key] for row in rows) for key in rows[0]["metrics"]}


def summary(records, reference):
    ref_rows = [reference[row["query_id"]] for row in records]
    aws_scores = mean_metrics(records)
    ref_scores = mean_metrics(ref_rows)
    differences = []
    exact = same_set = 0
    for row, ref in zip(records, ref_rows):
        aws, dbx = row["ranked_chunk_ids"], ref["ranked_chunk_ids"]
        exact += aws == dbx
        same_set += set(aws) == set(dbx)
        if aws != dbx or row["route"] != ref["route"]:
            differences.append({
                "query_id": row["query_id"], "aws_route": row["route"], "databricks_route": ref["route"],
                "aws_top5": aws, "databricks_top5": dbx,
                "recall_at_5_delta": row["metrics"]["recall_at_5"] - ref["metrics"]["recall_at_5"],
            })
    timings = {}
    for key in ("aws_retrieval_total", "aws_cli_invoke_wall", "local_query_embedding"):
        values = [r["latency_ms"][key] for r in records]
        timings[key] = {"mean": statistics.fmean(values), "median": statistics.median(values)}
    return {
        "metrics": {key: {"aws": value, "databricks": ref_scores[key], "delta": value-ref_scores[key]}
                    for key, value in aws_scores.items()},
        "exact_top5_order_cases": exact, "same_top5_set_cases": same_set,
        "route_counts": dict(Counter(row["route"] for row in records)),
        "reranked_cases": sum(row["reranked"] for row in records),
        "coverage_summary": {
            "applied_cases": sum(row["query_signals"].get("multi_identifier_coverage_applied") is True for row in records),
            "complete_cases": sum(row["query_signals"].get("coverage_complete") is True for row in records),
            "injected_cases": sum(bool(row["query_signals"].get("coverage_injected_chunk_ids")) for row in records),
        },
        "latency_ms_aws": timings,
        "latency_note": "AWS retrieval timing is in-handler search/policy only. CLI wall time also includes process launch, network and any initialization. Neither is a direct comparison to Databricks server ANN time.",
        "changed_cases": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--eval-file", type=Path, default=Path(EVAL_FILE))
    parser.add_argument("--baseline", type=Path, default=Path(BASELINE_FILE))
    parser.add_argument("--region", default="eu-north-1")
    parser.add_argument("--function", default="project-atlas-aws21-retrieval")
    parser.add_argument("--profile")
    parser.add_argument("--limit", type=int, help="Run the first N frozen cases as a separate smoke run")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit < CASE_COUNT:
        parser.error("--limit must be 1–49; omit it to run all 50")
    root = args.root.resolve()
    eval_path, baseline_path = root / args.eval_file, root / args.baseline
    cli = shutil.which("aws")
    if cli is None:
        raise RuntimeError("AWS CLI not found on PATH")
    cases, baseline, reference = load_inputs(eval_path, baseline_path)
    cases = cases[:args.limit] if args.limit is not None else cases
    sys.path.insert(0, str(root))
    import yaml
    from src.evaluation import retrieval_metrics
    from src.embeddings.bge_m3 import BGEM3Embedder

    config = yaml.safe_load((root / "config/baseline.yaml").read_text(encoding="utf-8"))
    model = config["embedding"]["model"]
    if model != baseline["embedding_model"]:
        raise ValueError(f"Configured embedding model ({model}) differs from baseline ({baseline['embedding_model']})")
    for case in cases:
        ref = reference[case["query_id"]]
        rescored = score_query(ref["ranked_chunk_ids"], case, retrieval_metrics)
        if any(not math.isclose(value, float(ref["metrics"][key]), abs_tol=1e-9)
               for key, value in rescored.items()):
            raise ValueError(f"Project metric definitions don't reproduce Track A scoring on {case['query_id']}")

    tag = f"_smoke_{args.limit}" if args.limit is not None else ""
    output = root / "results" / f"{OUTPUT_STEM}{tag}.json"
    checkpoint = output.with_suffix(".partial.json")
    identity = {
        "runner_version": 1, "function": args.function, "region": args.region,
        "profile": args.profile, "embedding_model": model, "retrieval_policy": "v04",
        "eval_sha256": sha256(eval_path), "baseline_sha256": sha256(baseline_path),
        "query_ids": [case["query_id"] for case in cases],
    }
    records = []
    if checkpoint.exists():
        prior = load_json(checkpoint)
        if prior["run"] != identity:
            raise ValueError(f"Run inputs changed. Move the old checkpoint before restarting: {checkpoint}")
        records = prior["cases"]
        if [r["query_id"] for r in records] != identity["query_ids"][:len(records)]:
            raise ValueError(f"Checkpoint does not match case order: {checkpoint}")
        print(f"Resuming at {len(records)}/{len(cases)} completed cases", flush=True)

    if len(records) < len(cases):
        print(f"Loading {model} and running {len(cases)-len(records)} Lambda calls", flush=True)
        embedder = BGEM3Embedder(model_name=model)
        with tempfile.TemporaryDirectory(prefix="atlas-aws-eval-") as tmp:
            for case in cases[len(records):]:
                query_id = case["query_id"]
                try:
                    start = time.perf_counter()
                    vector = [float(x) for x in embedder.embed_query(case["query"])]
                    embedding_ms = (time.perf_counter() - start) * 1000
                    if len(vector) != DIM or not all(math.isfinite(x) for x in vector) or not any(vector):
                        raise ValueError(f"Expected a nonzero, finite {DIM}d embedding")
                    body, wall, version = invoke(cli, args, Path(tmp), case["query"], vector)
                    hits = body["results"]
                    ids = [hit["chunk_id"] for hit in hits]
                    if len(ids) != 5 or len(set(ids)) != 5 or not all(isinstance(i, str) and i for i in ids):
                        raise ValueError("Expected exactly five distinct chunk IDs")
                    metrics = score_query(ids, case, retrieval_metrics)
                    records.append({
                        "query_id": query_id,
                        "category": case.get("category"), "language_direction": case.get("language_direction"),
                        "route": body["route"], "identifiers": body["identifiers"],
                        "reranked": body["reranked"], "query_signals": body["query_signals"],
                        "executed_version": version, "ranked_chunk_ids": ids,
                        "retrieved": [{key: hit.get(key) for key in RESULT_FIELDS} for hit in hits],
                        "metrics": metrics,
                        "latency_ms": {
                            **body["latency_ms"],
                            "aws_retrieval_total": float(body["latency_ms"]["retrieval_total"]),
                            "aws_cli_invoke_wall": round(wall, 2),
                            "local_query_embedding": round(embedding_ms, 2),
                        },
                    })
                    write_json_atomic(checkpoint, {"run": identity, "cases": records})
                except (ValueError, OSError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
                    raise RuntimeError(f"{query_id}: {error}") from error
                print(f"[{len(records):02d}/{len(cases):02d}] {query_id} | {body['route']} | Recall@5 {metrics['recall_at_5']:.3f}", flush=True)

    compared = summary(records, reference)
    write_json_atomic(output, {
        "run": identity, "dataset_version": baseline["dataset_version"],
        "evaluation": "smoke_subset" if args.limit is not None else "frozen_full",
        "case_count": len(records), "databricks_comparison": compared, "cases": records,
    })
    checkpoint.unlink(missing_ok=True)
    print(f"\nMatched-case comparison ({len(records)} queries)")
    for key in SUMMARY_METRICS:
        values = compared["metrics"][key]
        print(f"{key:23s} AWS {values['aws']:.4f} | Databricks {values['databricks']:.4f} | delta {values['delta']:+.4f}")
    print(f"Exact Top 5 order: {compared['exact_top5_order_cases']}/{len(records)}")
    print(f"Same Top 5 set:    {compared['same_top5_set_cases']}/{len(records)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped. Re-run the same command to resume completed cases.")
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as error:
        sys.exit(f"Evaluation stopped: {error}")
