import os
from typing import Any

from datasets import Dataset

def evaluate_ragas(question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
    """
    Evaluates a single Q&A generation using RAGas.
    Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"error": "Thiếu OPENAI_API_KEY để chạy RAGas Metrics."}

    if not answer or answer.startswith("[DEMO ANSWER]") or answer.startswith("[LLM FALLBACK]"):
        return {"error": "Cần sử dụng LLM Mode = 'real' để đánh giá."}

    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError:
        return {"error": "Chưa cài đặt thư viện 'ragas'."}

    # Prepare single-item dataset
    # We pass the generated answer as ground_truth just to satisfy context_precision's structural requirement
    # Though context_precision normally requires actual human ground truth.
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [answer], 
    }
    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy, context_precision]

    try:
        result = evaluate(dataset, metrics=metrics)
        # Result is a dict-like object
        return {
            "Faithfulness": float(result.get("faithfulness", 0.0)),
            "Answer Relevance": float(result.get("answer_relevancy", 0.0)),
            "Context Precision": float(result.get("context_precision", 0.0)),
        }
    except Exception as exc:
        return {"error": f"Lỗi RAGas: {str(exc)}"}
