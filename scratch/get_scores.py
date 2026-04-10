import math
import re

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def compute_similarity(vec_a, vec_b):
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)

import hashlib
class MockEmbedder:
    def __init__(self, dim=64):
        self.dim = dim
    def _embed_single(self, text):
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
    def __call__(self, text):
        if isinstance(text, str):
            return self._embed_single(text)
        return [self._embed_single(t) for t in text]

embedder = MockEmbedder()

pairs = [
    ("The cat sits on the mat", "A feline is resting on the rug"),
    ("I love programming in Python", "Python is a great language for coding"),
    ("The weather is sunny today", "It is a very bright and clear day"),
    ("Machine learning is a subset of AI", "Deep learning uses neural networks"),
    ("Apple is a tech company", "I like eating red apples")
]

for a, b in pairs:
    sim = compute_similarity(embedder(a), embedder(b))
    print(f"A: {a}")
    print(f"B: {b}")
    print(f"Sim: {sim:.4f}")
    print("-" * 20)
