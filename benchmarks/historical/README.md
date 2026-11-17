# Historical benchmark reruns

These two archived runners are the scripts used to evaluate the original AWS migration. **They are not required for the new repository's synthetic demo, offline tests, or live AWS-to-Ollama client.**

They import scoring modules and use evaluation inputs from the original Portfolio 2 environment. The new public repository intentionally omits the original frozen questions, embeddings, private snapshot, raw Databricks references, and legacy scoring modules. No scores in the [public summary](../atlas-aws21/summary.json) come from the synthetic demo.

To rerun the historical comparison, use the original Portfolio 2 project root with the original Python environment, AWS deployment, and local Ollama model. Run these archived files by absolute path while keeping the original project as your working directory. The files will read the expected local inputs under that project and write new AWS reports to its `results/`. Save existing results separately if you want to retain them.

| Required input in the original environment | Purpose |
| --- | --- |
| `data/evaluation/frozen/eval_cases_v01_full.jsonl` | Fixed 50-query evaluation |
| `results/retrieval_eval_v01_full_selective_intent_rerank_v04_databricks.json` | Retrieval reference |
| `results/generation_eval_v04_v021_databricks.json` | Generation reference |
| Original S3 snapshot and v0.4 Lambda | Same retrieval inputs and policy |
| Original `src/evaluation/` and `config/baseline.yaml` | Same scoring and model settings |

For a publishable, independently runnable example, start with the [public demo](../../README.md#quick-start).
