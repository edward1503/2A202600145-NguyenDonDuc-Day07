import math
import os
import shutil
from src.store import EmbeddingStore
from src.models import Document

# Clean up any existing chroma_db for a clean test
if os.path.exists("./chroma_test_db"):
    shutil.rmtree("./chroma_test_db")

def mock_embedding(text):
    # Dummy embedding for testing
    if isinstance(text, list):
        return [[0.1] * 64 for _ in text]
    return [0.1] * 64

# --- TEST 1: Write to Chroma ---
print("--- TEST 1: Writing data ---")
store = EmbeddingStore(collection_name="test_collection", embedding_fn=mock_embedding)
# Forcing path for test (need to modify store.py slightly or just rely on default if I can control cwd)
# For this test, I'll assume the implementation uses ./chroma_db if not otherwise specified.

doc = Document(id="doc1", content="Hello persistence world", metadata={"topic": "testing"})
store.add_documents([doc])
print(f"Added doc. Collection size: {store.get_collection_size()}")

# --- TEST 2: Re-init and Read ---
print("\n--- TEST 2: Re-initializing store ---")
# Re-initializing should pick up the data from the disk
store2 = EmbeddingStore(collection_name="test_collection", embedding_fn=mock_embedding)
size = store2.get_collection_size()
print(f"Collection size after re-init: {size}")

results = store2.search("hello", top_k=1)
if results:
    print(f"Found content: {results[0]['content']}")
    print(f"Metadata: {results[0]['metadata']}")

if size > 0 and results and results[0]['content'] == "Hello persistence world":
    print("\n✅ PERSISTENCE SUCCESSFUL!")
else:
    print("\n❌ PERSISTENCE FAILED or data mismatch.")

# Optional: cleanup
# shutil.rmtree("./chroma_db")
