# Project Atlas AWS 2-1 · Benchmark evidence

[summary.json](summary.json) contains public-safe aggregate metrics and per-case agreement flags derived from the retained AWS run and the saved Databricks Track A baseline. It contains no source documents, embeddings, raw questions, ranked chunk IDs, credentials, or generated answer text.

An exact Top 5 match requires the same five chunk IDs in the same order. An exact answer match compares the saved answer strings. Valid citation labels confirm that the cited labels exist in the supplied context, not that every cited document supports its claim. Review flags are heuristic triage, not an error rate.

The published result documents agreement on **50 frozen questions and a 128-chunk snapshot**. It is not an estimate of parity or accuracy on unseen questions. Read the [portfolio case study](../../docs/aws-portfolio.md#evaluation) for the benchmark scope and the shared ownership-boundary failure.
