from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        question = (question or "").strip()
        if not question:
            return "Please provide a question."

        retrieved = self.store.search(question, top_k=top_k)
        context_lines = []
        for idx, item in enumerate(retrieved, start=1):
            context_lines.append(f"[{idx}] {item.get('content', '').strip()}")

        if context_lines:
            context = "\n".join(context_lines)
        else:
            context = "(no relevant context found)"

        prompt = (
            "You are a helpful assistant answering from the provided context.\n"
            "If the context is insufficient, say that clearly.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
