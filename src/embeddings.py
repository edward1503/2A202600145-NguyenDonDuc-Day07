from __future__ import annotations

import hashlib
import math
from typing import Union

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def _embed_single(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def __call__(self, text: Union[str, list[str]]) -> Union[list[float], list[list[float]]]:
        if isinstance(text, str):
            return self._embed_single(text)
        return [self._embed_single(t) for t in text]


_local_model_instance = None

class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        global _local_model_instance
        self.model_name = model_name
        self._backend_name = model_name
        
        if _local_model_instance is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Explicitly setting device to 'cpu' to avoid 'meta tensor' errors
                _local_model_instance = SentenceTransformer(model_name, device="cpu")
            except Exception as e:
                print(f"[FATAL] Failed to load local embedding model: {e}")
                _local_model_instance = MockEmbedder()
            
        self.model = _local_model_instance

    def __call__(self, text: Union[str, list[str]]) -> Union[list[float], list[list[float]]]:
        if isinstance(self.model, MockEmbedder):
            return self.model(text)
            
        # Natively supports both single strings and lists
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return embedding


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI
        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: Union[str, list[str]]) -> Union[list[float], list[list[float]]]:
        # OpenAI supports batch input natively
        response = self.client.embeddings.create(model=self.model_name, input=text)
        if isinstance(text, str):
            return [float(value) for value in response.data[0].embedding]
        return [[float(v) for v in r.embedding] for r in response.data]


_mock_embed = MockEmbedder()
