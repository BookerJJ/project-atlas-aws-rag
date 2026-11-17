"""Offline handler checks using a fake S3 client and a 128-chunk fixture."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "lambda"
BUCKET = "atlas-test-snapshot"
KEY = "snapshots/test/fixture.jsonl"


class ClientError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class StreamingBody:
    def __init__(self, records):
        data = "".join(json.dumps(record) + "\n" for record in records)
        self.stream = io.BytesIO(data.encode())

    def iter_lines(self):
        yield from self.stream

    def close(self):
        self.stream.close()


class FakeS3:
    def __init__(self, records):
        self.records = records
        self.calls = []
        self.error = None
        self.body = None

    def get_object(self, *, Bucket, Key):
        self.calls.append((Bucket, Key))
        if self.error:
            raise self.error
        self.body = StreamingBody(self.records)
        return {"Body": self.body}


def vector(value):
    return [value, math.sqrt(1-value**2)] + [0.0] * 1022


def snapshot():
    def row(name, score, section, status, date, text):
        return {
            "chunk_id": f"{name}::chunk::000", "document_id": name,
            "document_type": "engineering_issue", "section": section,
            "status": status, "date": date, "version": "1.0",
            "language": "en", "point_id": name, "text": text, "embedding": vector(score),
        }
    special = [
        row("issue_net217", .98, "Current State", "open", "2026-09-02", "Production validation"),
        row("release_target", .91, "Release Target", "active", "2026-09-12", "Latest target"),
        row("issue_auth342", .87, "Dependency", "open", "2026-09-01", "AUTH-342 approval pending"),
        row("issue_obs119", .23, "Current State", "open", "2026-09-05", "OBS-119 validation"),
        row("security_decision", .96, "Decision", "superseded", "2026-08-02", "Archived policy"),
    ]
    return special + [
        row(f"fixture_{i:03}", .4 - i/1000, "Notes", "active", "2026-08-01", "Fixture text")
        for i in range(123)
    ]


class LambdaContract(unittest.TestCase):
    def setUp(self):
        self.s3 = FakeS3(snapshot())
        sys.path.insert(0, str(SOURCE))
        self.addCleanup(sys.path.pop, 0)
        modules = {
            "boto3": types.SimpleNamespace(client=lambda _: self.s3),
            "botocore.exceptions": types.SimpleNamespace(ClientError=ClientError),
        }
        spec = importlib.util.spec_from_file_location("atlas_test_lambda", SOURCE / "lambda_function.py")
        self.handler = importlib.util.module_from_spec(spec)
        modules[spec.name] = self.handler
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {
            "ATLAS_S3_BUCKET": BUCKET, "ATLAS_S3_KEY": KEY,
        }):
            spec.loader.exec_module(self.handler)

    def invoke(self, mode, **overrides):
        request = {"mode": mode, "top_k": 5}
        if mode != "smoke":
            request["query_embedding"] = vector(1)
        request.update(overrides)
        response = self.handler.lambda_handler(request, None)
        return response["statusCode"], json.loads(response["body"])

    def test_smoke_query_and_warm_reuse(self):
        code, smoke = self.invoke("smoke")
        self.assertEqual((code, smoke["chunks_loaded"], len(smoke["results"])), (200, 128, 5))
        self.assertEqual(smoke["results"][0]["score"], 1.0)
        self.assertNotIn("text", smoke["results"][0])
        code, body = self.invoke("query")
        self.assertEqual(code, 200)
        self.assertIn("text", body["results"][0])
        self.assertEqual(self.s3.calls, [(BUCKET, KEY)])
        self.assertTrue(self.s3.body.stream.closed)

    def test_dense_with_reranking(self):
        code, body = self.invoke("v04", query="현재 production release를 막고 있는 문제가 무엇인가?")
        self.assertEqual((code, body["route"], body["reranked"]), (200, "dense", True))
        self.assertTrue(all(hit["lexical_rank"] is None for hit in body["results"]))
        self.assertTrue(all("embedding" not in hit and "point_id" not in hit for hit in body["results"]))

    def test_hybrid_multi_id_coverage(self):
        code, body = self.invoke("v04", query="AUTH-342와 OBS-119의 현재 상태는?")
        self.assertEqual((code, body["route"], body["reranked"]), (200, "identifier_hybrid", True))
        signals = body["query_signals"]
        self.assertTrue(signals["coverage_complete"])
        self.assertTrue(all(signals["identifier_coverage"][i] for i in ("AUTH-342", "OBS-119")))
        self.assertTrue(any(hit["lexical_rank"] is not None for hit in body["results"]))

    def test_historical_rerank_penalty(self):
        code, body = self.invoke("v04", query="As of September 1, 2026, what is the release status?")
        self.assertEqual(code, 200)
        self.assertEqual(body["query_signals"]["reference_date"], "2026-09-01")
        self.assertTrue(any(hit["date_bonus"] < 0 for hit in body["results"]))

    def test_invalid_inputs_fail_without_s3(self):
        for bad in ([], [0.0]*1024, [True]*1024, [math.inf]*1024):
            code, _ = self.invoke("v04", query="What changed?", query_embedding=bad)
            self.assertEqual(code, 400)
        self.assertEqual(self.s3.calls, [])

    def test_missing_bucket_is_rejected(self):
        self.handler.S3_BUCKET = ""
        code, body = self.invoke("smoke")
        self.assertEqual((code, body["error_code"]), (500, "InvalidConfiguration"))
        self.assertEqual(self.s3.calls, [])

    def test_duplicate_snapshot_does_not_enter_cache(self):
        self.s3.records[-1]["chunk_id"] = self.s3.records[0]["chunk_id"]
        code, body = self.invoke("smoke")
        self.assertEqual((code, body["error_code"]), (500, "InvalidSnapshot"))
        self.assertTrue(self.s3.body.stream.closed)
        self.assertIsNone(self.handler._INDEX)

    def test_s3_denial_is_reported(self):
        self.s3.error = ClientError("AccessDenied")
        code, body = self.invoke("smoke")
        self.assertEqual((code, body["error_code"]), (500, "AccessDenied"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
