"""Turn retrieved chunks into a grounded answer using a local Ollama model."""

from typing import List, Dict
import ollama


SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.

Rules:
1. Every factual claim must come from the context. Do not use outside knowledge.
2. Cite the passage number(s) you used, like [2] or [1][3], after each claim.
3. If the context does not contain enough information to answer, reply exactly:
   INSUFFICIENT_CONTEXT
   followed by one sentence explaining what is missing.
4. Do not speculate. Do not fill gaps with plausible-sounding detail.
5. Be concise. Two to four sentences unless the question requires more."""


def format_context(hits: List[Dict]) -> str:
    """Render retrieved chunks as a numbered block."""
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] (source: {h['source']})\n{h['text']}")
    return "\n\n".join(parts)


def generate(question: str,
             hits: List[Dict],
             model: str = "llama3.1:8b",
             temperature: float = 0.0) -> Dict:
    """Generate a grounded answer from retrieved chunks."""
    context = format_context(hits)
    user_msg = f"Context passages:\n\n{context}\n\nQuestion: {question}"

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": temperature},
    )

    answer = resp["message"]["content"].strip()

    return {
        "question": question,
        "answer": answer,
        "abstained": answer.startswith("INSUFFICIENT_CONTEXT"),
        "sources": [{"n": i, "id": h["id"], "source": h["source"],
                     "similarity": h["similarity"]}
                    for i, h in enumerate(hits, start=1)],
        "model": model,
    }


if __name__ == "__main__":
    from src.store import VectorStore

    store = VectorStore()

    questions = [
        "How does U-Net handle limited training data?",
        "What is the purpose of residual connections in ResNet?",
        "What learning rate schedule did the DreamGaussian paper use?",
    ]

    for q in questions:
        hits = store.search(q, k=5)
        result = generate(q, hits)

        print("=" * 70)
        print(f"Q: {q}")
        print(f"abstained: {result['abstained']}")
        print(f"\n{result['answer']}\n")
        print("retrieved:")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['similarity']:.3f}  {s['source']}")
        print()