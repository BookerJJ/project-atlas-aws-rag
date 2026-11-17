#!/usr/bin/env python3
"""Connect Project Atlas AWS retrieval v0.4 to the existing local RAG generator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


def retrieve(cli, args, vector):
    with tempfile.TemporaryDirectory(prefix="atlas-aws-rag-") as directory:
        request = Path(directory) / "request.json"
        output = Path(directory) / "response.json"
        request.write_text(json.dumps({
            "mode": "v04", "query": args.query,
            "top_k": 5, "query_embedding": vector,
        }, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        command = [
            cli, "lambda", "invoke", "--region", args.region, "--function-name", args.function,
            "--invocation-type", "RequestResponse", "--output", "json", "--no-cli-pager",
            "--no-cli-auto-prompt", "--payload", f"fileb://{request}", str(output),
        ]
        if args.profile:
            command += ["--profile", args.profile]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=100)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or f"AWS CLI exit {completed.returncode}")
        meta = json.loads(completed.stdout)
        if meta.get("StatusCode") != 200 or meta.get("FunctionError"):
            raise RuntimeError(f"Lambda invocation failed: {meta.get('StatusCode')} {meta.get('FunctionError')}")
        response = json.loads(output.read_text(encoding="utf-8"))
        body = json.loads(response["body"]) if isinstance(response["body"], str) else response["body"]
        if response.get("statusCode") != 200 or body.get("status") != "success":
            raise RuntimeError(f"AWS retrieval failed: {response.get('statusCode')} {body.get('error_code', '')}")
        if body.get("mode") != "v04" or body.get("embedding_dimension") != 1024:
            raise ValueError("Unexpected AWS retrieval response")
        hits = body["results"]
        if len(hits) != 5 or any(not hit.get("chunk_id") or "text" not in hit for hit in hits):
            raise ValueError("AWS retrieval did not return five evidence chunks")
        return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--region", default="eu-north-1")
    parser.add_argument("--function", default="project-atlas-aws21-retrieval")
    parser.add_argument("--profile")
    args = parser.parse_args()

    root = Path.cwd()
    config_path = root / "config/baseline.yaml"
    if not config_path.is_file():
        raise RuntimeError("Run this script from the Project Atlas project directory")
    cli = shutil.which("aws")
    if cli is None:
        raise RuntimeError("AWS CLI is not on PATH")
    sys.path.insert(0, str(root))
    import yaml
    from src.embeddings.bge_m3 import BGEM3Embedder
    from src.generation.citations import validate_citations
    from src.generation.context import build_context
    from src.generation.ollama_client import OllamaClient
    from src.generation.prompt_v021 import PROMPT_VERSION, build_messages

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generator = config["generation"]
    embedder = BGEM3Embedder(model_name=config["embedding"]["model"])
    vector = [float(value) for value in embedder.embed_query(args.query)]
    if len(vector) != 1024 or not any(vector) or not all(math.isfinite(value) for value in vector):
        raise ValueError("Expected a nonzero, finite 1024-dimensional query embedding")

    print("Retrieving evidence from AWS Lambda...", flush=True)
    start = time.perf_counter()
    result = retrieve(cli, args, vector)
    invoke_ms = (time.perf_counter() - start) * 1000
    hits = [SimpleNamespace(payload=hit, score=hit["score"]) for hit in result["results"]]
    context, sources = build_context(hits)
    messages = build_messages(query=args.query, context=context)
    print(f"Generating locally with Ollama ({generator['model']})...", flush=True)
    try:
        response = OllamaClient().chat(
            model=generator["model"], messages=messages,
            temperature=generator["temperature"],
        )
    except OSError as error:
        raise RuntimeError(f"Local Ollama request failed for {generator['model']}: {error}") from error
    if not response.content.strip():
        raise ValueError("Ollama returned an empty answer")
    validation = validate_citations(response.content, {source.label for source in sources})
    print(f"\nRetrieval: AWS Lambda v0.4 ({result['route']}, reranked={result['reranked']})")
    print(f"Generation: local Ollama {response.model} / {PROMPT_VERSION}")
    if result["identifiers"]:
        print(f"Identifiers: {', '.join(result['identifiers'])}")
    coverage = result["query_signals"].get("coverage_complete")
    if coverage is not None:
        print(f"Coverage complete: {coverage}")
    print(f"AWS CLI round trip: {invoke_ms:.1f} ms")
    print(f"\nANSWER\n{response.content}")
    print("\nRETRIEVED SOURCES")
    for source in sources:
        print(f"[{source.label}] {source.chunk_id} | {source.status} | {source.date}")
    print(f"\nCitations valid: {validation['valid']}")
    print(f"Cited labels: {', '.join(validation['cited_labels']) or '(none)'}")
    if validation["invalid_labels"]:
        print(f"Unknown source labels: {', '.join(validation['invalid_labels'])}")
    return 0 if validation["valid"] else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(f"AWS RAG test stopped: {error}")
