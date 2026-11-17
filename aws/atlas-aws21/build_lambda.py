"""Build the Lambda ZIP with source files only."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

SOURCE = Path(__file__).resolve().parent / "lambda"
FILES = (
    "lambda_function.py",
    "atlas_policy.json",
    "src/__init__.py",
    "src/retrieval/__init__.py",
    "src/retrieval/retrieval_v2.py",
    "src/retrieval/lexical_bm25.py",
    "src/retrieval/hybrid_rrf.py",
    "src/retrieval/intent_reranker.py",
    "src/retrieval/query_router.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/atlas-aws21-lambda.zip"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in FILES:
            member = ZipInfo(name)
            member.compress_type = ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, (SOURCE / name).read_bytes())
    with ZipFile(args.output) as archive:
        if archive.testzip() is not None or tuple(archive.namelist()) != FILES:
            raise RuntimeError("Invalid Lambda ZIP")
    print(f"Built {args.output} ({len(FILES)} deployment files)")


if __name__ == "__main__":
    main()
