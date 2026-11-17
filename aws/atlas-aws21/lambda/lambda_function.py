"""Project Atlas AWS 2-1. Existing dense API and opt-in v0.4 policy."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.retrieval.lexical_bm25 import LexicalBM25Index
from src.retrieval.retrieval_v2 import RetrievalV2

S3_BUCKET = os.environ.get("ATLAS_S3_BUCKET", "").strip()
S3_KEY = os.environ.get("ATLAS_S3_KEY", "snapshots/v01/project_atlas_chunks_bge_m3.jsonl").strip()
VECTOR_DIM = 1024
EXPECTED_CHUNKS = 128

_POLICY = json.loads(Path(__file__).with_name("atlas_policy.json").read_text(encoding="utf-8"))
_S3 = boto3.client("s3")
_INDEX = None
_RETRIEVER = None


class SnapshotError(Exception):
    pass


@dataclass(frozen=True)
class DenseHit:
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    payload: dict[str, Any]
    vector: list[float]
    norm: float


def _parse_vector(value: Any) -> tuple[list[float], float]:
    if not isinstance(value, list) or len(value) != VECTOR_DIM:
        raise ValueError(f"Expected a {VECTOR_DIM}-dimensional numeric vector")
    try:
        if not all(type(x) in (int, float) and math.isfinite(x) for x in value):
            raise ValueError("Vector must contain only finite numbers")
        vector = [float(x) for x in value]
        norm = math.sqrt(sum(x * x for x in vector))
    except OverflowError:
        raise ValueError("Vector contains a number outside the supported range") from None
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Vector must have a finite, nonzero norm")
    return vector, norm


class DenseIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def search(self, *, collection_name: str, query_vector: list[float], top_k: int) -> list[DenseHit]:
        _, norm = _parse_vector(query_vector)
        hits = [
            DenseHit(
                sum(a * b for a, b in zip(query_vector, row.vector)) / (norm * row.norm),
                row.payload,
            )
            for row in self.chunks
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.payload["chunk_id"]))
        return hits[:top_k]


def _load_index() -> DenseIndex:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    body = _S3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)["Body"]
    chunks: list[Chunk] = []
    seen: set[str] = set()
    try:
        for line_number, line in enumerate(body.iter_lines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each JSONL row must be an object")
                chunk_id = row.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in seen:
                    raise ValueError("Missing or duplicate chunk_id")
                vector, norm = _parse_vector(row.pop("embedding", None))
                row.pop("point_id", None)
            except (ValueError, UnicodeError) as error:
                raise SnapshotError(f"Invalid snapshot row {line_number}") from error
            seen.add(chunk_id)
            chunks.append(Chunk(payload=row, vector=vector, norm=norm))
    finally:
        body.close()
    if len(chunks) != EXPECTED_CHUNKS:
        raise SnapshotError(f"Expected {EXPECTED_CHUNKS} snapshot records, found {len(chunks)}")
    _INDEX = DenseIndex(chunks)
    return _INDEX


def _get_retriever(index: DenseIndex) -> RetrievalV2:
    global _RETRIEVER
    if _RETRIEVER is None:
        cfg = _POLICY["lexical"]
        weights = cfg["field_weights"]
        payloads = sorted(
            (chunk.payload for chunk in index.chunks),
            key=lambda row: row["chunk_id"],
        )
        lexical = LexicalBM25Index(
            payloads,
            k1=cfg["k1"], b=cfg["b"],
            document_id_weight=weights["document_id"],
            title_weight=weights["title"],
            section_weight=weights["section"],
            text_weight=weights["text"],
        )
        _RETRIEVER = RetrievalV2(experiment=_POLICY, store=index, lexical_index=lexical)
    return _RETRIEVER


def _response(code: int, body: dict[str, Any]) -> dict[str, Any]:
    return {"statusCode": code, "body": json.dumps(body, ensure_ascii=False, allow_nan=False)}


def _error(code: int, error_code: str, message: str | None = None) -> dict[str, Any]:
    body = {"status": "error", "error_code": error_code}
    if message is not None:
        body["message"] = message
    return _response(code, body)


def _prepare_request(event: Any):
    if not isinstance(event, dict):
        raise ValueError("Event must be an object")
    mode = event.get("mode", "query")
    if mode not in ("smoke", "query", "v04"):
        raise ValueError("mode must be smoke, query, or v04")
    top_k = event.get("top_k", 5)
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be 1 to 10")
    query = None
    if mode == "v04":
        if top_k != _POLICY["hybrid"]["final_top_k"]:
            raise ValueError("v04 requires the frozen final Top 5")
        query = event.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise ValueError("v04 requires nonempty query text of at most 4096 characters")
    vector = None if mode == "smoke" else _parse_vector(event.get("query_embedding"))[0]
    return mode, top_k, query, vector


def lambda_handler(event, context):
    try:
        mode, top_k, query, vector = _prepare_request(event)
    except ValueError as error:
        return _error(400, "InvalidRequest", str(error))

    if not S3_BUCKET or not S3_KEY:
        return _error(500, "InvalidConfiguration", "Set ATLAS_S3_BUCKET and a nonempty ATLAS_S3_KEY")

    try:
        index = _load_index()
        base = {
            "status": "success",
            "mode": mode,
            "chunks_loaded": len(index.chunks),
            "embedding_dimension": VECTOR_DIM,
        }
        if mode == "v04":
            result = _get_retriever(index).retrieve(query, vector)
            hits = [
                {
                    **hit.payload,
                    "score": round(hit.score, 6),
                    "source_rank": hit.source_rank,
                    "dense_rank": hit.dense_rank,
                    "lexical_rank": hit.lexical_rank,
                    "dense_score": round(hit.dense_score, 6) if hit.dense_score is not None else None,
                    "lexical_score": round(hit.lexical_score, 6) if hit.lexical_score is not None else None,
                    "identifier_bonus": hit.identifier_bonus,
                    "section_bonus": hit.section_bonus,
                    "status_bonus": hit.status_bonus,
                    "date_bonus": hit.date_bonus,
                    "coverage_identifiers": hit.coverage_identifiers,
                    "coverage_injected": hit.coverage_injected,
                }
                for hit in result.hits
            ]
            score_kind = (
                "rerank" if result.reranked else
                "weighted_rrf" if result.route == "identifier_hybrid" else "cosine"
            )
            return _response(200, {
                **base,
                "route": result.route,
                "identifiers": result.identifiers,
                "reranked": result.reranked,
                "score_kind": score_kind,
                "query_signals": result.query_signals,
                "latency_ms": result.latency_ms,
                "results": hits,
            })

        if mode == "smoke":
            vector = index.chunks[0].vector
        matches = index.search(collection_name="", query_vector=vector, top_k=top_k)
        fields = ("chunk_id", "document_id", "document_type", "date", "status", "version")
        hits = []
        for match in matches:
            item = {field: match.payload.get(field) for field in fields}
            item["score"] = round(match.score, 6)
            if mode == "query":
                item["text"] = match.payload.get("text", "")
            hits.append(item)
        return _response(200, {**base, "results": hits})

    except ClientError as error:
        return _error(500, error.response.get("Error", {}).get("Code", "S3ClientError"))
    except SnapshotError as error:
        print(str(error))
        return _error(500, "InvalidSnapshot")
