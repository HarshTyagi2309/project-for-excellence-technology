import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, TOP_K


class VectorStore:
    def __init__(self):
        # Load the embedding model only when needed.
        self.model = None
        self.index = None
        self.chunks = []

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        return self.model

    def build(self, chunks: list[str]) -> None:
        if not chunks:
            raise ValueError("No text chunks were created from this document.")

        model = self._load_model()
        embeddings = model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.chunks = chunks
        self.index = index

    def search(self, question: str) -> list[dict]:
        if self.index is None or not self.chunks:
            return []

        model = self._load_model()
        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        count = min(TOP_K, len(self.chunks))
        scores, indices = self.index.search(query_embedding, count)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            results.append({
                "chunk_id": int(idx) + 1,
                "text": self.chunks[int(idx)],
                "score": float(score),
            })

        return results
