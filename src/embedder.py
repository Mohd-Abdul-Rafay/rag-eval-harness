"""Turn text chunks into vectors."""

from typing import List
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32,
               show_progress: bool = True) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # makes cosine == dot product
        )
        return vecs.tolist()


if __name__ == "__main__":
    import numpy as np

    emb = Embedder()
    print(f"model: {emb.model_name}")
    print(f"device: {emb.device}")
    print(f"dim: {emb.dim}\n")

    sentences = [
        "Data augmentation is essential when only few training samples are available.",
        "How does U-Net handle limited training data?",
        "The transformer uses multi-head self-attention.",
        "Residual connections allow training of very deep networks.",
    ]

    vecs = np.array(emb.encode(sentences, show_progress=False))

    print("cosine similarity matrix:\n")
    print(f"{'':>4}", end="")
    for i in range(len(sentences)):
        print(f"{'s'+str(i):>8}", end="")
    print()
    for i in range(len(sentences)):
        print(f"{'s'+str(i):>4}", end="")
        for j in range(len(sentences)):
            print(f"{float(vecs[i] @ vecs[j]):>8.3f}", end="")
        print()

    print("\ns0: augmentation with few samples")
    print("s1: how does U-Net handle limited data")
    print("s2: transformer self-attention")
    print("s3: residual connections")