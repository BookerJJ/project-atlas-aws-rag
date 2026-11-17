# Project Atlas 2-1 · AWS retrieval deployment

[Portfolio case study](../../docs/aws-portfolio.md) · [한국어 포트폴리오](../../docs/aws-portfolio.ko.md) · [Benchmark evidence](../../benchmarks/atlas-aws21/summary.json)

This directory contains the AWS retrieval source, not the complete bilingual RAG project. The live runner in `scripts/` uses the local embedding and generation modules in this repository. The original frozen corpus, questions, and raw historical results are separate from the public synthetic example.

The publishable handler differs from the benchmarked deployment in one respect. **The snapshot bucket and key are configured through environment variables.** The retrieval policy, candidate selection, response modes, and v0.4 settings are unchanged. The publishable handler was verified with offline tests. The benchmark data comes from the original live deployment.

## Offline test and ZIP build

Run from the Git repository root. The offline test uses a generated fixture and a fake S3 client. It does not call AWS, download models, or need the benchmark corpus.

```bash
python3 aws/atlas-aws21/tests/test_lambda_contract.py
python3 aws/atlas-aws21/build_lambda.py
```

The builder creates `dist/atlas-aws21-lambda.zip` with the Lambda entry point, policy configuration, and source modules at the archive root. Keep the generated ZIP and `dist/` out of Git.

## Build a standalone synthetic snapshot

If you are not reusing the original deployment, generate a new JSONL snapshot from the public synthetic corpus. This step requires the [local dependencies](../../README.md#live-aws-to-ollama-demo) and can download the BGE-M3 model. It writes a local file only.

```bash
./.venv/bin/python scripts/build_demo_snapshot.py
```

Upload `dist/atlas-synthetic-snapshot.jsonl` to your private S3 bucket and set `ATLAS_S3_KEY` to its object key. Adjust the execution role's example policy to the same key. Do not use this snapshot to interpret the historical 50-query results.

## Configure your own Lambda

The existing deployment does not need to be replaced to publish the portfolio. For a fresh deployment, use the settings below and upload the built ZIP.

| Setting | Value |
| --- | --- |
| Runtime used by this project | Python 3.12 |
| Handler | `lambda_function.lambda_handler` |
| Starting memory and timeout used by this project | 512 MB and 30 seconds |
| `ATLAS_S3_BUCKET` | Your private S3 bucket name, without `s3://` |
| `ATLAS_S3_KEY` | Snapshot object key. Default is `snapshots/v01/project_atlas_chunks_bge_m3.jsonl`. |

The benchmark handler expects a JSONL snapshot with **128 unique `chunk_id` values** and **1,024 finite numeric embedding values per row**, with a nonzero vector norm. Each row needs source text and the project metadata used by the retrieval policy, including document ID, section, date, status, and version. Use a corpus approved for your deployment. The local query embedding must use the same BGE-M3 model and settings as the embeddings in the snapshot.

Give the Lambda execution role read access to the snapshot object. [The example S3 policy](iam/s3-read-policy.example.json) grants only `s3:GetObject` on the specified object. Replace the bucket placeholder and adjust the key if it differs from the default. The execution role also needs CloudWatch Logs permissions if you want execution logs. The caller's AWS identity needs `lambda:InvokeFunction` on your function. No public function URL is used by this project.

## Event contract

| Mode | Required event fields | Purpose |
| --- | --- | --- |
| `smoke` | `mode` | Searches with an existing snapshot vector and omits text in the response |
| `query` | `mode`, `query_embedding` | Runs dense-only search and returns evidence text |
| `v04` | `mode`, `query`, `query_embedding` | Executes v0.4 routing, fusion, reranking, and coverage as applicable |

`top_k` defaults to 5. The legacy modes accept 1–10 and `v04` requires 5 to preserve the frozen evaluation setting. The model remains local. Lambda returns evidence and diagnostics, not a generated answer.

A Lambda console smoke test can use this event without a local model.

```json
{"mode": "smoke", "top_k": 1}
```

Look for the function response's `statusCode: 200`, `mode: smoke`, 128 loaded chunks, and a top score of 1.0. To run a real question from the Mac, see the [live AWS-to-Ollama demo](../../README.md#live-aws-to-ollama-demo). The client scripts default to the original function name and region. For your own deployment, pass `--function` and `--region` when they differ.
