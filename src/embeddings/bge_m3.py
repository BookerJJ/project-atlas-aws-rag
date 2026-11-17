from __future__ import annotations

from sentence_transformers import SentenceTransformer


class BGEM3Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def embed_documents(
        self,
        texts: list[str],
    ):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(
        self,
        query: str,
    ):
        return self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
