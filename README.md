# RAG Evaluation Harness

A retrieval-augmented generation pipeline over a corpus of computer vision papers, built around an evaluation harness that measures **retrieval quality and generation quality separately** — because when a RAG system returns a wrong answer, "the system is wrong" is not a diagnosis.

Runs entirely locally. No API keys, no paid services.

> **Status:** in progress. Ingestion and embedding complete; vector store, generation, and evaluation in development. Findings below come from real runs, not expectations.

---

## Why separate the metrics

A single end-to-end accuracy number tells you a RAG system failed without telling you *where*. There are two distinct failure modes and they need different fixes:

- **Retrieval failed** — the answer was never in the context. Fix chunking, embeddings, or `k`.
- **Generation failed** — the answer was in the context and the model ignored it, or blended it with parametric knowledge. Fix the prompt or the model.

Conflating them means guessing. So retrieval is measured with `recall@k` and MRR against known source chunks, and generation is measured for faithfulness and abstention correctness given the retrieved context.

---

## Stack

| Component | Choice | Why |
|---|---|---|
| Generation | Ollama + Llama 3.1 8B | Local, free, fast enough for hundreds of eval calls |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | 384-dim, runs on Apple MPS |
| Vector store | ChromaDB | Local persistence, no server |
| Serving | FastAPI | |
| Corpus | 10 arXiv CV papers | ResNet, Transformer, ViT, Swin, SimCLR, EfficientNet, U-Net, YOLO, Focal Loss, DETR |

The corpus is deliberately in a domain I know well, because the eval set is hand-written and correctness has to be judged rather than assumed.

---

## Setup

```bash
git clone https://github.com/Mohd-Abdul-Rafay/rag-eval-harness
cd rag-eval-harness

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com/download) and pull the model:

```bash
ollama pull llama3.1:8b
```

Download the corpus:

```bash
cd data/papers
for id in 1512.03385 1706.03762 2010.11929 2103.14030 2002.05709 \
          1905.11946 1505.04597 1506.02640 1708.02002 2005.12872; do
  curl -L -o "${id}.pdf" "https://arxiv.org/pdf/${id}"; sleep 2
done
cd ../..
```

---

## Build log

### Chunking

Recursive character splitting on paragraph, line, sentence, then word boundaries, with configurable overlap. Chunk size is a **variable in the comparison below**, not a value copied from a tutorial, because the tradeoff is real: small chunks retrieve precisely but sever context; large chunks preserve context but produce embeddings averaged over too many ideas.

```bash
python src/chunker.py
```

Current corpus at `chunk_size=512, overlap=50`:

```
papers: 10
chunks: 224
avg words/chunk: 421
min: 63   max: 562
```

The average falls below target because the splitter respects paragraph boundaries rather than cutting at an exact word count. The max exceeds target because overlap is prepended after splitting — worth noting, since 562 words is roughly 730 tokens and the embedder truncates at 512.

### Embedding

`all-MiniLM-L6-v2`, 384 dimensions, running on Apple MPS. Vectors are normalised so cosine similarity reduces to a dot product.

```bash
python src/embedder.py
```

**Finding — the baseline embedder does not separate in-domain topics well.**

Cosine similarity across four probe sentences:

|  | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| **s0** augmentation with few samples | 1.000 | **0.463** | 0.168 | 0.413 |
| **s1** how does U-Net handle limited data | 0.463 | 1.000 | 0.070 | **0.453** |
| **s2** transformer self-attention | 0.168 | 0.070 | 1.000 | 0.258 |
| **s3** residual connections | 0.413 | 0.453 | 0.258 | 1.000 |

The intended result is visible: **s0 to s1 scores 0.463** despite near-zero vocabulary overlap, which is the case for dense retrieval over keyword search.

The unintended result is more useful. **s1 to s3 scores 0.453** — "residual connections allow training of very deep networks" is judged nearly as relevant to the U-Net question as the actually-correct answer, despite being unrelated. A 6-layer general-purpose embedder appears to encode *topic* ("this is a deep learning sentence") more strongly than the specific semantic relation being asked about.

This motivates three things that follow: comparing `all-mpnet-base-v2` against the baseline, cross-encoder reranking (which scores query and candidate jointly rather than independently), and hybrid retrieval with BM25 to recover exact-term matching.

### Vector store — *in progress*

### Generation — *in progress*

### Evaluation harness — *in progress*

---

## Planned comparison

One variable at a time, everything else fixed, so any difference is attributable to a single change.

| Variant | Chunk size | Top-k | Retrieval | recall@5 | MRR | Faithfulness | Abstention |
|---|---|---|---|---|---|---|---|
| Baseline | 512 | 5 | dense (MiniLM) | — | — | — | — |
| Chunk 256 | 256 | 5 | dense (MiniLM) | — | — | — | — |
| Chunk 1024 | 1024 | 5 | dense (MiniLM) | — | — | — | — |
| Top-k 3 | 512 | 3 | dense (MiniLM) | — | — | — | — |
| Top-k 10 | 512 | 10 | dense (MiniLM) | — | — | — | — |
| Larger embedder | 512 | 5 | dense (mpnet) | — | — | — | — |
| BM25 only | 512 | 5 | sparse | — | — | — | — |
| Hybrid (RRF) | 512 | 5 | dense + BM25 | — | — | — | — |
| Hybrid + rerank | 512 | 5 | dense + BM25 + cross-encoder | — | — | — | — |

---

## Metrics

**Retrieval**
- `recall@k` — is the known source chunk in the top *k*?
- `MRR` — mean reciprocal rank of the correct chunk.

**Generation**
- **Faithfulness** — is every claim supported by the retrieved context?
- **Answer relevance** — does it address the question asked?
- **Abstention correctness** — when the context genuinely lacks the answer, does the model say so instead of inventing one?

Abstention is measured deliberately. Llama 3.1 8B abstains correctly on unknown-entity questions *without* retrieval — asked about a nonexistent paper, it replied that the work "may not exist at all." The failure mode that matters in RAG is different: plausible-but-wrong retrieved context making the model confident when it should not be. The eval set includes questions targeting exactly that.

---

## Repository layout

```
src/
  chunker.py       PDF to cleaned text to overlapping chunks
  embedder.py      text to normalised vectors (MPS-accelerated)
  store.py         ChromaDB persistence and k-NN search
  generate.py      retrieved context + question to grounded answer
  app.py           FastAPI service
eval/
  questions.jsonl  hand-written eval set with known source chunks
  metrics.py       retrieval + generation metrics
  run.py           runs one config, logs per-variant results
  configs/         one YAML per variant
data/papers/       source PDFs (not committed)
data/chroma/       vector store (not committed, regenerable)
```

---

## Limitations

- Corpus is 10 papers in one domain; results may not transfer to heterogeneous corpora.
- The eval set is hand-written and small, so differences of a few points are not meaningful.
- Faithfulness scoring uses LLM-as-judge, which is imperfect and correlated with the generator.
- Chunks exceeding the embedder's 512-token limit are silently truncated.

---

**Abdul Rafay Mohd** — M.S. Artificial Intelligence, University of North Texas
[GitHub](https://github.com/Mohd-Abdul-Rafay) · [LinkedIn](https://linkedin.com/in/mohd-abdul-rafay)
