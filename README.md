# Project Atlas · AWS RAG

[한국어](README.ko.md) · [Migration case study](docs/aws-portfolio.md) · [Benchmark evidence](benchmarks/atlas-aws21/summary.json) · [GitHub setup](docs/github-setup.ko.md)

**A bilingual technical-program copilot with AWS retrieval and local answer generation.** The migration from Databricks Track A to AWS reproduced the exact Top 5 evidence order and the answer text on all **50 frozen evaluation questions**.

| Historical benchmark | Verified result |
| --- | --- |
| Exact Top 5 order / exact answer text | 50/50 each |
| Search quality | Recall@5 **0.9800** and MRR **0.8557**, unchanged |
| Citations | 50/50 valid source labels, not a proof of factual accuracy |
| Architecture | Lambda and S3 for retrieval. BGE-M3 and Ollama `qwen3:8b` on the local machine. |

Lambda reads a private S3 snapshot and applies dense search, ID-based BM25 and RRF, selective reranking, and multi-ID evidence coverage. The local model uses the returned evidence to generate cited answers in Korean or English. This is not a fully cloud-hosted model.

## Quick start

**Run the public retrieval demo without AWS credentials, dependencies, or model downloads.** Use Python 3.12 and run from this repository's root.

```bash
python3 examples/offline_demo.py
python3 aws/atlas-aws21/tests/test_lambda_contract.py
```

The demo executes the actual handler with a fake S3 client, synthetic documents, and fixed test vectors. It prints `identifier_hybrid`, direct evidence for both demo issue IDs, and `Coverage complete: True`. It does not call AWS, compute model embeddings, or generate an answer.

## Live AWS-to-Ollama demo

For a live run, use a deployed Atlas Lambda you can invoke through the AWS CLI. Install the local dependencies and run Ollama with `qwen3:8b`.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-local.txt
```

To test a deployment built with the **public synthetic snapshot**, run the following. If you use a different AWS function or region, pass `--function` and `--region`.

```bash
./.venv/bin/python scripts/query_rag_aws_v04_v021.py \
  --query "DEMO-101과 DEMO-202의 현재 상태는 각각 무엇인가?"
```

See the [AWS deployment guide](aws/atlas-aws21/README.md) to build the Lambda ZIP and generate the public synthetic snapshot with BGE-M3. If you already have a deployment using the original benchmark snapshot, you can run the original `AUTH-342` and `OBS-119` question without redeploying.

## Repository map

| Path | Contents |
| --- | --- |
| `aws/atlas-aws21/` | Lambda source, ZIP builder, IAM example, and offline tests |
| `examples/` | Public synthetic corpus and offline demo |
| `src/`, `config/`, `scripts/` | Local embedding, prompting, Ollama generation, citation checks, and live demo |
| `benchmarks/atlas-aws21/` | Published-safe results, with 50 per-question agreement rows |
| `benchmarks/historical/` | Original evaluation runners and instructions for the separate legacy environment |
| `docs/` | Bilingual case study and GitHub setup walkthrough |

The published metrics were measured on the **original 128-chunk snapshot and 50 frozen queries**. The synthetic demo is separate and does not reproduce those scores. The private corpus, embeddings, raw evaluation questions, and answer logs are not bundled. The [case study](docs/aws-portfolio.md) explains a shared retrieval miss and why equal results do not imply perfect answers.
