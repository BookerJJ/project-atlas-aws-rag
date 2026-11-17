"""Evaluate AWS-backed Retrieval v0.4 with the existing Track A generation scorer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
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
from types import SimpleNamespace

import yaml


ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))


from src.embeddings.bge_m3 import BGEM3Embedder
from src.generation.context import build_context
from src.generation.prompt_v021 import (
    PROMPT_VERSION,
    build_messages,
)
from src.generation.ollama_client import (
    OllamaClient,
    ns_to_ms,
)
from src.generation.citations import (
    validate_citations,
)
from src.evaluation.generation_metrics import (
    evaluate_generation,
    review_flags,
)


REFERENCE = "results/generation_eval_v04_v021_databricks.json"
THRESHOLDS = {
    "correctness": 0.65, "groundedness": 0.58, "citation_support": 0.58,
    "groundedness_sentence": 0.50, "citation_support_sentence": 0.50,
}
METRICS_TO_COMPARE = (
    "citation_label_valid_rate", "historical_scope_violation_rate",
    "current_superseded_citation_rate", "manual_review_rate",
    "avg_correctness_similarity", "avg_groundedness_similarity",
    "avg_citation_semantic_support",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        staged = Path(f.name)
        try:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.close()
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)


def validated_reference(path, cases, baseline):
    published = read_json(path)
    expectations = {
        "evaluation_name": "project_atlas_generation_eval_v04_v021_databricks_track_a",
        "case_count": 50, "dataset_version": baseline["project"]["dataset_version"],
        "embedding_model": baseline["embedding"]["model"],
        "generation_model": baseline["generation"]["model"],
        "prompt_version": PROMPT_VERSION, "heuristic_thresholds": THRESHOLDS,
    }
    for key, value in expectations.items():
        if published.get(key) != value:
            raise ValueError(f"The Databricks reference does not match {key}: {path}")
    reference = {row["query_id"]: row for row in published["cases"]}
    if len(cases) != 50 or len(reference) != 50 or set(reference) != {row["query_id"] for row in cases}:
        raise ValueError("Expected the same 50 unique question IDs as the Databricks evaluation")
    for case in cases:
        for field in ("query", "category", "language_direction", "expected_answer_summary"):
            if case[field] != reference[case["query_id"]][field]:
                raise ValueError(f"Databricks reference differs for {case['query_id']} ({field})")
    return reference


class AWSRetriever:
    def __init__(self, *, embedder, args):
        self.cli = shutil.which("aws")
        if self.cli is None:
            raise RuntimeError("AWS CLI is not on PATH")
        self.embedder = embedder
        self.args = args

    def retrieve(self, query):
        start = time.perf_counter()
        vector = [float(value) for value in self.embedder.embed_query(query)]
        embed_ms = (time.perf_counter() - start) * 1000
        if len(vector) != 1024 or not any(vector) or not all(math.isfinite(x) for x in vector):
            raise ValueError("Expected a nonzero, finite 1024-dimensional query vector")
        with tempfile.TemporaryDirectory(prefix="atlas-aws-generation-") as d:
            request = Path(d) / "request.json"
            output = Path(d) / "response.json"
            request.write_text(json.dumps({
                "mode": "v04", "query": query, "query_embedding": vector, "top_k": 5,
            }, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            cmd = [
                self.cli, "lambda", "invoke", "--region", self.args.region,
                "--function-name", self.args.function, "--invocation-type", "RequestResponse",
                "--output", "json", "--no-cli-pager", "--no-cli-auto-prompt",
                "--payload", f"fileb://{request}", str(output),
            ]
            if self.args.profile:
                cmd += ["--profile", self.args.profile]
            invoke_start = time.perf_counter()
            call = subprocess.run(cmd, text=True, capture_output=True, timeout=100)
            invoke_ms = (time.perf_counter() - invoke_start) * 1000
            if call.returncode:
                raise RuntimeError(call.stderr.strip() or f"AWS CLI exit {call.returncode}")
            meta = json.loads(call.stdout)
            if meta.get("StatusCode") != 200 or meta.get("FunctionError"):
                raise RuntimeError(f"Lambda invoke failed: {meta.get('StatusCode')} {meta.get('FunctionError')}")
            response = read_json(output)
            body = response["body"]
            body = json.loads(body) if isinstance(body, str) else body
            if response.get("statusCode") != 200 or body.get("status") != "success":
                raise RuntimeError(f"AWS retrieval failed: {response.get('statusCode')} {body.get('error_code', '')}")
        if (body.get("mode") != "v04" or body.get("chunks_loaded") != 128
                or body.get("embedding_dimension") != 1024
                or body.get("route") not in {"dense", "identifier_hybrid"}):
            raise ValueError("Unexpected AWS response mode, route, or snapshot metadata")
        hits = body["results"]
        ids = [hit.get("chunk_id") for hit in hits]
        if (len(ids) != 5 or len(set(ids)) != 5
                or not all(isinstance(i, str) and i for i in ids)
                or any(not isinstance(hit.get("text"), str) for hit in hits)):
            raise ValueError("Expected five distinct AWS evidence chunks with text")
        timings = dict(body["latency_ms"])
        timings["aws_search_policy"] = timings.pop("retrieval_total")
        timings["query_embedding"] = round(embed_ms, 2)
        timings["aws_cli_invoke_wall"] = round(invoke_ms, 2)
        timings["retrieval_total"] = round((time.perf_counter() - start) * 1000, 2)
        return SimpleNamespace(
            route=body["route"], identifiers=body["identifiers"], reranked=body["reranked"],
            query_signals=body["query_signals"], latency_ms=timings,
            hits=[SimpleNamespace(payload=hit, score=hit["score"]) for hit in hits],
            executed_version=meta.get("ExecutedVersion"),
        )


def compare_tracks(results, reference, overall):
    published = [reference[row["query_id"]] for row in results]
    ref_agg = aggregate(published)
    comparisons = {}
    for key in METRICS_TO_COMPARE:
        aws, dbx = overall[key], ref_agg[key]
        comparisons[key] = {
            "aws": aws, "databricks": dbx, "delta": (aws - dbx if aws is not None and dbx is not None else None),
        }
    same = sum(a["retrieval"]["ranked_chunk_ids"] == b["retrieval"]["ranked_chunk_ids"]
               for a, b in zip(results, published))
    transitions = []
    for a, b in zip(results, published):
        if (a["citation_label_validity"]["valid"] != b["citation_label_validity"]["valid"]
                or set(a["review_flags"]) != set(b["review_flags"])
                or a["behavior"]["post_reference_date_citation_count"] != b["behavior"]["post_reference_date_citation_count"]
                or a["behavior"]["current_query_superseded_citation_count"] != b["behavior"]["current_query_superseded_citation_count"]):
            transitions.append({
                "query_id": a["query_id"],
                "aws_flags": a["review_flags"], "databricks_flags": b["review_flags"],
                "aws_valid": a["citation_label_validity"]["valid"],
                "databricks_valid": b["citation_label_validity"]["valid"],
                "aws_post_reference_date_citation_count": a["behavior"]["post_reference_date_citation_count"],
                "databricks_post_reference_date_citation_count": b["behavior"]["post_reference_date_citation_count"],
                "aws_current_query_superseded_citation_count": a["behavior"]["current_query_superseded_citation_count"],
                "databricks_current_query_superseded_citation_count": b["behavior"]["current_query_superseded_citation_count"],
            })
    return {
        "matched_case_count": len(results), "exact_top5_order_cases": same,
        "aws_historical_case_count": overall["historical_case_count"],
        "databricks_historical_case_count": ref_agg["historical_case_count"],
        "aws_current_case_count": overall["current_case_count"],
        "databricks_current_case_count": ref_agg["current_case_count"],
        "metrics": comparisons, "changed_review_or_citation_behavior": transitions,
        "note": "The comparison uses same-case historical Databricks results. Similarity is a heuristic, not a factual-accuracy rate. Local model runtime, generation variability and evidence differences can affect results.",
    }


def review_entry(row):
    return {
        "query_id": row["query_id"], "query": row["query"],
        "expected": row["expected_answer_summary"], "answer": row["answer"],
        "flags": row["review_flags"], "behavior": row["behavior"],
        "retrieved_chunk_ids": row["retrieval"]["ranked_chunk_ids"],
    }


def load_jsonl(
    path: Path,
) -> list[dict]:
    rows: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}: "
                    f"{exc}"
                ) from exc

    return rows


def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return sum(values) / len(
        values
    )


def nullable_mean(
    values: list[float | None],
) -> float | None:
    usable = [
        value
        for value in values
        if value is not None
    ]

    if not usable:
        return None

    return mean(usable)


def parse_date(
    value: str | None,
) -> date | None:
    if not value:
        return None

    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()
    except ValueError:
        return None


def behavior_diagnostics(
    *,
    citation_result: dict,
    sources: list,
    query_signals: dict,
    answer: str,
    output_tokens: int | None,
    answer_sentence_count: int,
) -> dict:
    cited_labels = set(
        citation_result.get(
            "cited_labels",
            [],
        )
    )

    cited_sources = [
        source
        for source in sources
        if source.label in cited_labels
    ]

    reference_date = parse_date(
        query_signals.get(
            "reference_date"
        )
    )

    post_reference = []

    if reference_date is not None:
        for source in cited_sources:
            source_date = parse_date(
                source.date
            )

            if (
                source_date is not None
                and source_date
                > reference_date
            ):
                post_reference.append(
                    source.label
                )

    superseded = [
        source.label
        for source in cited_sources
        if str(
            source.status
        ).lower()
        == "superseded"
    ]

    current_superseded = (
        superseded
        if query_signals.get(
            "current_intent",
            False,
        )
        else []
    )

    return {
        "answer_char_count": len(
            answer
        ),
        "answer_sentence_count": (
            answer_sentence_count
        ),
        "output_tokens": output_tokens,
        "cited_source_count": len(
            cited_sources
        ),
        "post_reference_date_citations": (
            post_reference
        ),
        "post_reference_date_citation_count": len(
            post_reference
        ),
        "current_query_superseded_citations": (
            current_superseded
        ),
        "current_query_superseded_citation_count": len(
            current_superseded
        ),
    }


def aggregate(
    cases: list[dict],
) -> dict:
    if not cases:
        return {}

    citation_support = nullable_mean(
        [
            case[
                "generation_metrics"
            ][
                "citation_semantic_support"
            ]
            for case in cases
        ]
    )

    historical_cases = [
        case
        for case in cases
        if case[
            "retrieval"
        ][
            "query_signals"
        ].get(
            "reference_date"
        )
    ]

    current_cases = [
        case
        for case in cases
        if case[
            "retrieval"
        ][
            "query_signals"
        ].get(
            "current_intent",
            False,
        )
    ]

    return {
        "case_count": len(cases),

        "avg_correctness_similarity": mean(
            [
                case[
                    "generation_metrics"
                ][
                    "correctness_similarity"
                ]
                for case in cases
            ]
        ),

        "avg_answer_relevance_similarity": mean(
            [
                case[
                    "generation_metrics"
                ][
                    "answer_relevance_similarity"
                ]
                for case in cases
            ]
        ),

        "avg_groundedness_similarity": mean(
            [
                case[
                    "generation_metrics"
                ][
                    "groundedness_similarity"
                ]
                for case in cases
            ]
        ),

        "avg_citation_semantic_support": (
            citation_support
        ),

        "avg_citation_coverage": mean(
            [
                case[
                    "generation_metrics"
                ][
                    "citation_coverage"
                ]
                for case in cases
            ]
        ),

        "avg_number_recall": mean(
            [
                case[
                    "generation_metrics"
                ][
                    "number_recall"
                ]
                for case in cases
            ]
        ),

        "avg_technical_id_recall": mean(
            [
                case[
                    "generation_metrics"
                ][
                    "technical_id_recall"
                ]
                for case in cases
            ]
        ),

        "citation_label_valid_rate": mean(
            [
                1.0
                if case[
                    "citation_label_validity"
                ][
                    "valid"
                ]
                else 0.0
                for case in cases
            ]
        ),

        "manual_review_rate": mean(
            [
                1.0
                if case[
                    "review_flags"
                ]
                else 0.0
                for case in cases
            ]
        ),

        "avg_retrieval_latency_ms": mean(
            [
                case[
                    "latency_ms"
                ][
                    "retrieval_total"
                ]
                for case in cases
            ]
        ),

        "median_generation_wall_ms": (
            statistics.median(
                [
                    case[
                        "latency_ms"
                    ][
                        "generation_wall"
                    ]
                    for case in cases
                ]
            )
        ),

        "avg_generation_wall_ms": mean(
            [
                case[
                    "latency_ms"
                ][
                    "generation_wall"
                ]
                for case in cases
            ]
        ),

        "avg_end_to_end_ms": mean(
            [
                case[
                    "latency_ms"
                ][
                    "end_to_end"
                ]
                for case in cases
            ]
        ),

        "avg_output_tokens": mean(
            [
                float(
                    case[
                        "behavior"
                    ][
                        "output_tokens"
                    ]
                    or 0
                )
                for case in cases
            ]
        ),

        "avg_answer_sentence_count": mean(
            [
                float(
                    case[
                        "behavior"
                    ][
                        "answer_sentence_count"
                    ]
                )
                for case in cases
            ]
        ),

        "historical_case_count": len(
            historical_cases
        ),

        "historical_scope_violation_rate": (
            mean(
                [
                    1.0
                    if case[
                        "behavior"
                    ][
                        "post_reference_date_citation_count"
                    ]
                    > 0
                    else 0.0
                    for case in historical_cases
                ]
            )
            if historical_cases
            else 0.0
        ),

        "current_case_count": len(
            current_cases
        ),

        "current_superseded_citation_rate": (
            mean(
                [
                    1.0
                    if case[
                        "behavior"
                    ][
                        "current_query_superseded_citation_count"
                    ]
                    > 0
                    else 0.0
                    for case in current_cases
                ]
            )
            if current_cases
            else 0.0
        ),
    }


def grouped(
    cases: list[dict],
    key: str,
) -> dict:
    groups: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for case in cases:
        groups[
            str(
                case.get(
                    key,
                    "unknown",
                )
            )
        ].append(case)

    return {
        group_name: aggregate(
            group_cases
        )
        for (
            group_name,
            group_cases,
        ) in sorted(
            groups.items()
        )
    }


def derive_smoke_path(
    path: Path,
    limit: int,
) -> Path:
    return path.with_name(
        f"{path.stem}_smoke{limit}"
        f"{path.suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--region", default="eu-north-1")
    parser.add_argument("--function", default="project-atlas-aws21-retrieval")
    parser.add_argument("--profile")

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. "
            "Smoke output is saved to a "
            "separate file automatically."
        ),
    )

    args = parser.parse_args()

    if args.limit is not None and not 1 <= args.limit < 50:
        parser.error("--limit must be between 1 and 49; omit it to run all 50")
    config = {
        "evaluation": {
            "cases_file": "data/evaluation/frozen/eval_cases_v01_full.jsonl",
            "output_file": "results/generation_eval_v04_v021_aws.json",
            "manual_review_file": "results/generation_eval_v04_v021_aws_review_queue.jsonl",
        },
        "retrieval": {
            "version": "selective_intent_rerank_v0.4_aws",
            "config_file": "atlas_policy.json (AWS Lambda package)",
        },
        "heuristic_thresholds": THRESHOLDS,
    }

    baseline_path = (
        ROOT
        / "config"
        / "baseline.yaml"
    )

    with baseline_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        baseline = yaml.safe_load(f)

    eval_path = (
        ROOT
        / config[
            "evaluation"
        ][
            "cases_file"
        ]
    )

    output_path = (
        ROOT
        / config[
            "evaluation"
        ][
            "output_file"
        ]
    )

    review_path = (
        ROOT
        / config[
            "evaluation"
        ][
            "manual_review_file"
        ]
    )

    cases = load_jsonl(eval_path)
    ref_path = ROOT / args.reference
    reference = validated_reference(ref_path, cases, baseline)

    if args.limit is not None:
        cases = cases[
            : args.limit
        ]

        output_path = (
            derive_smoke_path(
                output_path,
                args.limit,
            )
        )

        review_path = (
            derive_smoke_path(
                review_path,
                args.limit,
            )
        )

    print(
        f"Generation evaluation cases: "
        f"{len(cases)}"
    )
    print(
        f"Dataset: "
        f"{baseline['project']['dataset_version']}"
    )
    print(
        f"Retrieval: "
        f"{config['retrieval']['version']}"
    )
    print(
        f"Generation model: "
        f"{baseline['generation']['model']}"
    )
    print(
        f"Prompt: {PROMPT_VERSION}"
    )
    print()

    thresholds = config["heuristic_thresholds"]
    run = {
        "revision": 1, "region": args.region, "function": args.function, "profile": args.profile,
        "query_ids": [case["query_id"] for case in cases],
        "frozen_cases_sha256": fingerprint(eval_path),
        "databricks_generation_reference_sha256": fingerprint(ref_path),
        "baseline_config_sha256": fingerprint(baseline_path),
        "prompt_sha256": fingerprint(ROOT / "src/generation/prompt_v021.py"),
        "metrics_sha256": fingerprint(ROOT / "src/evaluation/generation_metrics.py"),
    }
    checkpoint = output_path.with_suffix(".partial.json")
    results = []
    if checkpoint.exists():
        prior = read_json(checkpoint)
        if prior["run"] != run:
            raise ValueError(f"Run settings changed; move the old checkpoint before restarting: {checkpoint}")
        results = prior["cases"]
        if [row["query_id"] for row in results] != run["query_ids"][:len(results)]:
            raise ValueError("Checkpoint order differs from the frozen evaluation")
        print(f"Resuming at {len(results)}/{len(cases)} completed cases", flush=True)

    if len(results) < len(cases):
        embedder = BGEM3Embedder(model_name=baseline["embedding"]["model"])
        retriever = AWSRetriever(embedder=embedder, args=args)
        client = OllamaClient()

    flag_counter = Counter(flag for row in results for flag in row["review_flags"])

    for index, case in enumerate(cases[len(results):], start=len(results)+1):
        print("=" * 76)
        print(
            f"[{index}/{len(cases)}] "
            f"{case['query_id']} "
            f"[{case['language_direction']}] "
            f"[{case['category']}]"
        )
        print(case["query"])

        total_start = (
            time.perf_counter()
        )

        retrieval = retriever.retrieve(
            case["query"]
        )

        context, sources = (
            build_context(
                retrieval.hits
            )
        )

        messages = build_messages(
            query=case["query"],
            context=context,
        )

        generation_start = (
            time.perf_counter()
        )

        response = client.chat(
            model=baseline[
                "generation"
            ][
                "model"
            ],
            messages=messages,
            temperature=baseline[
                "generation"
            ][
                "temperature"
            ],
        )

        generation_wall_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000

        if not response.content.strip():
            raise RuntimeError(f"{case['query_id']}: Ollama returned an empty answer")
        if response.model != baseline["generation"]["model"]:
            raise RuntimeError(f"{case['query_id']}: unexpected Ollama model {response.model}")

        valid_labels = {
            source.label
            for source in sources
        }

        citation_result = (
            validate_citations(
                response.content,
                valid_labels,
            )
        )

        metric_result = (
            evaluate_generation(
                embedder=embedder,
                query=case["query"],
                answer=response.content,
                expected_answer=case[
                    "expected_answer_summary"
                ],
                sources=sources,
                groundedness_threshold=thresholds[
                    "groundedness_sentence"
                ],
                citation_support_threshold=thresholds[
                    "citation_support_sentence"
                ],
            )
        )

        flags = review_flags(
            metrics=metric_result,
            citation_label_valid=(
                citation_result["valid"]
            ),
            correctness_threshold=thresholds[
                "correctness"
            ],
            groundedness_threshold=thresholds[
                "groundedness"
            ],
            citation_support_threshold=thresholds[
                "citation_support"
            ],
        )

        for flag in flags:
            flag_counter[flag] += 1

        behavior = behavior_diagnostics(
            citation_result=citation_result,
            sources=sources,
            query_signals=(
                retrieval.query_signals
            ),
            answer=response.content,
            output_tokens=(
                response.eval_count
            ),
            answer_sentence_count=(
                metric_result.answer_sentence_count
            ),
        )

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        result = {
            "query_id": case["query_id"],
            "query": case["query"],
            "language_direction": case[
                "language_direction"
            ],
            "category": case["category"],
            "expected_answer_summary": (
                case[
                    "expected_answer_summary"
                ]
            ),

            "retrieval": {
                "executed_lambda_version": retrieval.executed_version,
                "version": config[
                    "retrieval"
                ][
                    "version"
                ],
                "route": retrieval.route,
                "identifiers": (
                    retrieval.identifiers
                ),
                "reranked": (
                    retrieval.reranked
                ),
                "query_signals": (
                    retrieval.query_signals
                ),
                "ranked_chunk_ids": [
                    hit.payload.get(
                        "chunk_id"
                    )
                    for hit in retrieval.hits
                ],
            },

            "sources": [
                {
                    "label": source.label,
                    "chunk_id": (
                        source.chunk_id
                    ),
                    "document_id": (
                        source.document_id
                    ),
                    "section": (
                        source.section
                    ),
                    "status": source.status,
                    "date": source.date,
                    "language": (
                        source.language
                    ),
                    "score": source.score,
                }
                for source in sources
            ],

            "answer": response.content,

            "citation_label_validity": (
                citation_result
            ),

            "generation_metrics": {
                "correctness_similarity": round(
                    metric_result.correctness_similarity,
                    4,
                ),
                "answer_relevance_similarity": round(
                    metric_result.answer_relevance_similarity,
                    4,
                ),
                "groundedness_similarity": round(
                    metric_result.groundedness_similarity,
                    4,
                ),
                "citation_semantic_support": (
                    round(
                        metric_result.citation_semantic_support,
                        4,
                    )
                    if (
                        metric_result.citation_semantic_support
                        is not None
                    )
                    else None
                ),
                "citation_coverage": round(
                    metric_result.citation_coverage,
                    4,
                ),
                "number_recall": round(
                    metric_result.number_recall,
                    4,
                ),
                "technical_id_recall": round(
                    metric_result.technical_id_recall,
                    4,
                ),
                "unsupported_sentence_count": (
                    metric_result.unsupported_sentence_count
                ),
                "citation_supported_sentence_count": (
                    metric_result.citation_supported_sentence_count
                ),
                "cited_sentence_count": (
                    metric_result.cited_sentence_count
                ),
                "answer_sentence_count": (
                    metric_result.answer_sentence_count
                ),
            },

            "behavior": behavior,

            "review_flags": flags,

            "latency_ms": {
                **retrieval.latency_ms,
                "generation_wall": round(
                    generation_wall_ms,
                    2,
                ),
                "ollama_total": round(
                    ns_to_ms(
                        response.total_duration_ns
                    ),
                    2,
                ),
                "ollama_model_load": round(
                    ns_to_ms(
                        response.load_duration_ns
                    ),
                    2,
                ),
                "end_to_end": round(
                    total_ms,
                    2,
                ),
            },

            "generation_metrics_raw": {
                "prompt_tokens": (
                    response.prompt_eval_count
                ),
                "output_tokens": (
                    response.eval_count
                ),
            },
        }

        results.append(result)
        write_json_atomic(checkpoint, {"run": run, "cases": results})

        print(
            f"Answer: "
            f"{response.content}"
        )
        print(
            "Metrics: "
            f"correctness="
            f"{metric_result.correctness_similarity:.3f}, "
            f"groundedness="
            f"{metric_result.groundedness_similarity:.3f}, "
            f"relevance="
            f"{metric_result.answer_relevance_similarity:.3f}, "
            f"citation_support="
            f"{metric_result.citation_semantic_support}"
        )
        print(
            "Behavior: "
            f"tokens="
            f"{behavior['output_tokens']}, "
            f"future_citations="
            f"{behavior['post_reference_date_citation_count']}, "
            f"current_superseded="
            f"{behavior['current_query_superseded_citation_count']}"
        )
        print(
            f"Flags: "
            f"{flags or 'none'}"
        )
        print()

    overall = aggregate(results)
    comparison = compare_tracks(results, reference, overall)
    review_rows = [review_entry(row) for row in results if row["review_flags"]]

    by_language = grouped(
        results,
        "language_direction",
    )

    by_category = grouped(
        results,
        "category",
    )

    output = {
        "evaluation_name": (
            "project_atlas_generation_eval_v04_v021_aws"
        ),
        "run": run,
        "databricks_generation_comparison": comparison,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "case_count": len(results),
        "dataset_version": baseline[
            "project"
        ][
            "dataset_version"
        ],
        "retrieval_version": config[
            "retrieval"
        ][
            "version"
        ],
        "retrieval_config": config[
            "retrieval"
        ][
            "config_file"
        ],
        "embedding_model": baseline[
            "embedding"
        ][
            "model"
        ],
        "generation_model": baseline[
            "generation"
        ][
            "model"
        ],
        "prompt_version": (
            PROMPT_VERSION
        ),
        "heuristic_thresholds": (
            thresholds
        ),
        "metric_note": (
            "Embedding-based correctness, relevance, groundedness, "
            "and citation-support scores are deterministic semantic "
            "similarity heuristics, not entailment or human judgments."
        ),
        "behavior_note": (
            "Temporal-scope and superseded-source diagnostics measure "
            "citation behavior only. They are not general factual "
            "correctness scores."
        ),
        "overall": overall,
        "by_language_direction": (
            by_language
        ),
        "by_category": by_category,
        "review_flag_counts": dict(
            flag_counter
        ),
        "cases": results,
    }

    write_json_atomic(output_path, output)

    review_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with review_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in review_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 76)
    print("AWS LAMBDA — GENERATION v0.4 + PROMPT v0.2.1 AGGREGATE")
    print()

    print(
        f"Correctness similarity: "
        f"{overall['avg_correctness_similarity']:.3f}"
    )
    print(
        f"Answer relevance: "
        f"{overall['avg_answer_relevance_similarity']:.3f}"
    )
    print(
        f"Groundedness similarity: "
        f"{overall['avg_groundedness_similarity']:.3f}"
    )
    print(
        f"Citation semantic support: "
        f"{overall['avg_citation_semantic_support']}"
    )
    print(
        f"Citation label valid rate: "
        f"{overall['citation_label_valid_rate']:.3f}"
    )
    print(
        f"Citation coverage: "
        f"{overall['avg_citation_coverage']:.3f}"
    )
    print(
        f"Numeric recall: "
        f"{overall['avg_number_recall']:.3f}"
    )
    print(
        f"Manual-review rate: "
        f"{overall['manual_review_rate']:.3f}"
    )
    print(
        f"Avg output tokens: "
        f"{overall['avg_output_tokens']:.1f}"
    )
    print(
        f"Historical scope violation rate: "
        f"{overall['historical_scope_violation_rate']:.3f}"
    )
    print(
        f"Current superseded citation rate: "
        f"{overall['current_superseded_citation_rate']:.3f}"
    )
    checkpoint.unlink(missing_ok=True)
    print(f"\nMatched Databricks generation comparison ({len(results)} questions)")
    print(f"Top 5 order match: {comparison['exact_top5_order_cases']}/{len(results)}")
    print(f"Historical cases (AWS / Databricks): {comparison['aws_historical_case_count']} / {comparison['databricks_historical_case_count']}")
    print(f"Current cases (AWS / Databricks): {comparison['aws_current_case_count']} / {comparison['databricks_current_case_count']}")
    for key, values in comparison["metrics"].items():
        view = lambda v: "n/a" if v is None else f"{v:.4f}"
        print(f"{key:34s} AWS {view(values['aws'])} | Databricks {view(values['databricks'])} | delta {view(values['delta'])}")
    print()
    print(
        f"Saved: "
        f"{output_path.relative_to(ROOT)}"
    )
    print(
        f"Review queue: "
        f"{review_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped. Re-run the same command to resume completed questions.")
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(f"Generation evaluation stopped: {error}")
