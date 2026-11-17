"""Run the actual Lambda handler with synthetic data and a fake S3 client."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from synthetic_corpus import synthetic_chunks

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "aws/atlas-aws21/lambda"
QUERY = "DEMO-101과 DEMO-202의 현재 상태는 각각 무엇인가?"


class FakeBody(io.BytesIO):
    def iter_lines(self):
        yield from self


class DemoS3:
    def get_object(self, *, Bucket, Key):
        if (Bucket, Key) != ("demo-fixture", "synthetic.jsonl"):
            raise RuntimeError("Offline demo attempted an unexpected snapshot")
        rows = synthetic_chunks()
        weights = [.95, .30, .97, .70, .60, .94] + [.55 - i/1000 for i in range(122)]
        for row, score in zip(rows, weights):
            row["embedding"] = [score, math.sqrt(1-score**2)] + [0.0] * 1022
        data = "".join(json.dumps(row) + "\n" for row in rows)
        return {"Body": FakeBody(data.encode("utf-8"))}


class FakeClientError(Exception):
    pass


def main():
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location("atlas_offline_handler", SOURCE / "lambda_function.py")
    handler = importlib.util.module_from_spec(spec)
    modules = {
        spec.name: handler,
        "boto3": SimpleNamespace(client=lambda _: DemoS3()),
        "botocore.exceptions": SimpleNamespace(ClientError=FakeClientError),
    }
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {
        "ATLAS_S3_BUCKET": "demo-fixture", "ATLAS_S3_KEY": "synthetic.jsonl",
    }):
        spec.loader.exec_module(handler)
        response = handler.lambda_handler({
            "mode": "v04", "query": QUERY, "top_k": 5,
            "query_embedding": [1.0] + [0.0] * 1023,
        }, None)
    body = json.loads(response["body"])
    if response["statusCode"] != 200 or body["route"] != "identifier_hybrid":
        raise RuntimeError(f"Unexpected offline demo result: {body}")
    print("Synthetic retrieval demo. No AWS calls, BGE inference, or answer generation.")
    print(QUERY)
    print(f"Route: {body['route']}")
    print(f"Coverage complete: {body['query_signals']['coverage_complete']}")
    for i, hit in enumerate(body["results"], start=1):
        print(f"[S{i}] {hit['chunk_id']} | {hit['text']}")
    if not body["query_signals"]["coverage_complete"]:
        raise RuntimeError("Expected direct evidence for both fixture issues")


if __name__ == "__main__":
    main()
