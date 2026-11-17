"""Embed the public synthetic corpus into a Lambda-compatible local JSONL file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.synthetic_corpus import synthetic_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/atlas-synthetic-snapshot.jsonl"))
    args = parser.parse_args()

    import yaml
    from src.embeddings.bge_m3 import BGEM3Embedder

    config = yaml.safe_load((ROOT / "config/baseline.yaml").read_text(encoding="utf-8"))
    chunks = synthetic_chunks()
    if len(chunks) != 128 or len({row["chunk_id"] for row in chunks}) != 128:
        raise ValueError("Expected 128 unique synthetic chunk IDs")
    embedder = BGEM3Embedder(model_name=config["embedding"]["model"])
    vectors = embedder.embed_documents([row["text"] for row in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count differs from chunk count")
    for row, vector in zip(chunks, vectors):
        values = [float(x) for x in vector]
        if len(values) != 1024 or not any(values) or not all(math.isfinite(x) for x in values):
            raise ValueError(f"Invalid embedding for {row['chunk_id']}")
        row["embedding"] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as f:
        staged = Path(f.name)
        try:
            for row in chunks:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.close()
            staged.replace(args.output)
        finally:
            staged.unlink(missing_ok=True)
    print(f"Wrote {len(chunks)} synthetic chunks to {args.output}. No AWS resources changed.")


if __name__ == "__main__":
    main()
