# RAG Evaluation Harness

A retrieval-augmented generation pipeline over a corpus of computer vision papers, built around an evaluation harness that measures **retrieval quality and generation quality separately** — because when a RAG system returns a wrong answer, "the system is wrong" is not a diagnosis.

Runs entirely locally. No API keys, no paid services.

## Headline result

Seven retrieval configurations were tested against a hand-verified 45-question eval set. **No configuration beat plain dense retrieval on recall@1.** Every added component — a larger embedding model, a sparse index, rank fusion, a cross-encoder reranker — improved recall@5 while making the top result *worse*.

| Variant | recall@1 | recall@5 | MRR |
|---|---|---|---|
| **Dense (MiniLM, k=5)** | **25.0%** | 50.0% | 0.322 |
| Dense (mpnet) | 19.4% | 44.4% | 0.279 |
| BM25 only | 13.9% | 30.6% | 0.187 |
| Hybrid RRF (MiniLM + BM25) | 16.7% | 58.3% | 0.315 |
| Hybrid RRF (mpnet + BM25) | 13.9% | 47.2% | 0.260 |
| Dense + cross-encoder rerank | 19.4% | 63.9% | **0.350** |
| Hybrid + rerank | 13.9% | **66.7%** | 0.324 |

There is no configuration that wins on both metrics. Which one is correct depends on what consumes the output: a human scanning five results should get hybrid+rerank; a model receiving only the top chunk should get plain dense.

> **Status:** retrieval comparison complete. Generation metrics measured for the baseline. Per-variant generation runs pending.

---

## Why separate the metrics

A single end-to-end accuracy number tells you a RAG system failed without telling you *where*. There are two distinct failure modes needing different fixes:

- **Retrieval failed** — the answer was never in the context. Fix chunking, embeddings, or `k`.
- **Generation failed** — the answer was in the context and the model ignored it or blended it with parametric knowledge. Fix the prompt or the model.

The baseline settles this empirically. Retrieval puts the correct chunk first 25% of the time, while 97.7% of answers are judged faithful to whatever context they received. An aggregate score would have looked respectable and pointed at nothing.

---

## Stack

| Component | Choice |
|---|---|
| Generation | Ollama + Llama 3.1 8B, temperature 0 |
| Embeddings | sentence-transformers, Apple MPS |
| Sparse retrieval | `rank_bm25` (BM25Okapi) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector store | ChromaDB, cosine space |
| Judge | Llama 3.1 8B — same model as the generator, a limitation |
| Corpus | 10 arXiv CV papers, 188 chunks |

Corpus: ResNet, Transformer, ViT, Swin, SimCLR, EfficientNet, U-Net, YOLO, Focal Loss, DETR. Deliberately a domain I know well, because the eval set is hand-written and correctness must be judged rather than assumed.

---

## Setup

```bash
git clone https://github.com/Mohd-Abdul-Rafay/rag-eval-harness
cd rag-eval-harness
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b

cd data/papers
for id in 1512.03385 1706.03762 2010.11929 2103.14030 2002.05709 \
          1905.11946 1505.04597 1506.02640 1708.02002 2005.12872; do
  curl -L -o "${id}.pdf" "https://arxiv.org/pdf/${id}"; sleep 2
done
cd ../..
```

Reproduce the comparison:

```bash
python -m src.ingest --reset
python -m eval.verify

python -m eval.run --name r_dense         --retriever dense         --skip-generation
python -m eval.run --name r_bm25          --retriever bm25          --skip-generation
python -m eval.run --name r_hybrid        --retriever hybrid        --skip-generation
python -m eval.run --name r_rerank        --retriever rerank        --skip-generation
python -m eval.run --name r_hybrid_rerank --retriever hybrid_rerank --skip-generation

python -m src.ingest --reset --collection papers_mpnet --embed-model all-mpnet-base-v2
python -m eval.run --name r_mpnet --retriever dense \
  --collection papers_mpnet --embed-model all-mpnet-base-v2 --skip-generation

python -m eval.run --name baseline    # full run with generation, ~13 min
```

---

## The eval set

45 hand-written questions in `eval/questions.jsonl`, each with a gold chunk, reference answer, and type.

| Type | Count | Purpose |
|---|---|---|
| `answerable` | 28 | The answer sits in one identifiable chunk |
| `distractor` | 8 | Answer exists, but other chunks look more relevant |
| `unanswerable` | 9 | Tests abstention, including near-misses where the right paper is present but the fact is not |

Questions are phrased as a user would ask them, never copied from the papers — copying source wording makes retrieval trivially easy and the metrics meaningless.

**Gold IDs were verified, not assumed.** An initial pass flagged 17 questions whose gold chunk did not appear in the top 20. Inspection showed 13 were mislabelled by me and 4 were genuine retrieval failures. Unverified ground truth would have made every number here wrong.

---

## Findings

### 1 — The baseline embedder does not separate in-domain topics

|  | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| **s0** augmentation with few samples | 1.000 | **0.463** | 0.168 | 0.413 |
| **s1** how does U-Net handle limited data | 0.463 | 1.000 | 0.070 | **0.453** |
| **s2** transformer self-attention | 0.168 | 0.070 | 1.000 | 0.258 |
| **s3** residual connections | 0.413 | 0.453 | 0.258 | 1.000 |

Intended: **s0–s1 = 0.463** despite near-zero vocabulary overlap. Unintended: **s1–s3 = 0.453**, an unrelated sentence scoring nearly as relevant as the correct answer. The embedder encodes *topic* more strongly than semantic relation.

### 2 — The failure reproduces at corpus scale

Top-5 for *"How does U-Net handle limited training data?"*: the correct chunk ranks **second, behind an unrelated ResNet chunk, by 0.002**. Total spread across the top five is 0.053.

### 3 — Generation compensated for a retrieval failure

On that query the model ignored the top-ranked ResNet passage and cited the correct one. End-to-end it looks like success while `recall@1` is 0.

### 4 — Abstention held under plausible-but-wrong context

| Question type | Top-1 similarity |
|---|---|
| Answerable from corpus | 0.50 |
| Not in corpus | 0.45 |

A score threshold looked promising as a cheap filter. Finding 8 kills it.

### 5 — Removing 16% of the corpus as noise changed nothing

Reference sections were being embedded as content. Stripping them cut 224 chunks to 188. Ranks 1–4 for the test query were **identical to three decimal places** afterwards; the bibliography chunk was evicted from position 5 and replaced by an equally irrelevant results table. Retained for the smaller index, but it ruled out the cheap explanation.

### 6 — The embedder repeatedly fails on exact terminology

| Query | Target contains | Dense result |
|---|---|---|
| "layer normalization before every block" | *"Layernorm (LN) is applied before every block"* verbatim | Absent from top 5 |
| "hierarchical feature maps various scales backbone" | *"constructs hierarchical feature maps"* verbatim | Zero Swin chunks returned |
| "elastic deformation random displacement" | Section 3.1 Data Augmentation | Rank 1 — but the natural-language form drops it out of the top 20 |

The layernorm chunk additionally ranked **3rd for an unrelated projection-head query**. Dense retrieval fails on near-identical text and succeeds on unrelated text.

### 7 — Augmentation questions retrieve each other's papers

q002 (U-Net augmentation) returns SimCLR chunks. q017 (SimCLR augmentation) returns *other* SimCLR chunks, not its own ablation.

### 8 — The one hallucination fabricated a relation, not a fact

Asked how ViT compares to Swin on COCO — a comparison neither paper makes:

> The Vision Transformer (ViT) and Swin Transformer are compared on COCO object detection, with Swin achieving a box AP of 58.7 and mask AP of 51.1 [4].

Every number is correct and correctly cited. The fabrication is *"are compared"* — a relationship the context never establishes. This was the only abstention failure in 9 unanswerable questions, and its top-1 similarity was **0.661**, the highest of any unanswerable question and well above the 0.50 answerable average. The Finding 4 threshold would have passed it confidently.

### 9 — Faithfulness scoring gave that hallucination a perfect score

After fixing the harness to judge every non-abstained answer:

```
faithfulness_failed_abstention: 1.0
```

The judge marked the fabricated comparison fully faithful — correctly, by its own criteria, since every claim traces to context. What does not trace is the *relation* between the claims.

Claim-level faithfulness scoring is structurally blind to this. It verifies that facts appear in context; it cannot verify that the asserted relationship between them appears there. Detecting it would require entailment checking over the whole answer. This was only found by inspecting an individual failure rather than trusting the 96.4% aggregate.

### 10 — Hybrid retrieval fixed the exact-term failures and cost more than it gained

Per-question ranks for the Finding 6 case (q032, layernorm):

| Retriever | Rank |
|---|---|
| Dense | not found |
| BM25 | 3 |
| Hybrid | 2 |
| Rerank | **1** |
| Hybrid + rerank | **1** |

Exactly as predicted. But comparing hybrid+rerank against dense across all questions at rank 1: **2 gained, 6 lost.**

| | Questions |
|---|---|
| Dense solves, hybrid+rerank does not | q011, q013, q014, q020, q021, q029 |
| Hybrid+rerank solves, dense does not | q009, q032 |

The gains are exact-term queries; the losses are conceptual paraphrase. The two methods fail in opposite directions and fusion inherits both failure modes at rank 1 while recovering more candidates by rank 5.

The cross-encoder is trained on MS MARCO web passages. Dense CV paper prose is out of distribution, so it pulls relevant candidates into the pool without reliably ordering them.

### 11 — A 5× larger embedding model performed worse

| Model | Params | Dim | recall@1 | recall@5 | MRR |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 22M | 384 | **25.0%** | **50.0%** | **0.322** |
| `all-mpnet-base-v2` | 110M | 768 | 19.4% | 44.4% | 0.279 |

mpnet ranks above MiniLM on general sentence-similarity benchmarks and lost by 5.6 points on both recall metrics here.

Finding 1 diagnosed the problem as insufficient embedder capacity. This falsifies that. A better hypothesis: both models are trained on general web and QA text, and CV paper prose — dense with notation, citations, and jargon — is out of distribution for both. **Capacity does not compensate for domain mismatch.** The indicated fix is a domain-adapted embedder (SPECTER, SciNCL) or fine-tuning on the corpus, not a larger general-purpose model.

### 12 — Three questions are unreachable by any configuration

q002, q017, and q031 return no gold chunk across all seven retrieval configurations, two embedding models, sparse, hybrid, and reranked. The failure is not in the retrieval strategy; it is upstream in chunking or in the general-purpose embedder itself. Left as an open problem rather than tuned away.

### 13 — Increasing k improves MRR without improving retrieval

| Variant | recall@1 | recall@5 | MRR |
|---|---|---|---|
| Dense k=5 | 25.0% | 50.0% | 0.322 |
| Dense k=10 | 25.0% | 50.0% | 0.356 |
| Rerank k=5 | 19.4% | 63.9% | 0.350 |
| Rerank k=10 | 19.4% | 63.9% | 0.374 |

recall@1 and recall@5 are necessarily unchanged. Only MRR moves, because gold chunks at ranks 6–10 now contribute a small reciprocal instead of zero. Raising k improves the *appearance* of retrieval quality on a cutoff-sensitive metric while retrieving nothing better.

### 14 — RRF's published constant is wrong at this scale

`rrf_k=60`, the value from the original RRF paper, produced fused results containing neither list's top hits. With 20-item candidate lists over 188 chunks, `1/(60+rank)` flattens rank differences so severely that a document ranked mid-list in *both* runs outscores one ranked first in a single run. Lowered to 5, chosen for a defensible curve shape rather than tuned per query.

---

## Generation metrics (baseline)

| Metric | Result |
|---|---|
| Abstention correct | 89% (8/9) |
| Faithfulness, answerable | 97.7% (22 judged) |
| Faithfulness, all non-abstained | 96.4% (28 judged) |
| Faithfulness on the one hallucination | **1.0** |

---

## Repository layout

```
src/
  chunker.py       PDF to cleaned text to overlapping chunks, reference stripping
  embedder.py      text to normalised vectors (MPS-accelerated)
  store.py         ChromaDB persistence and k-NN search
  retrievers.py    BM25, RRF hybrid, cross-encoder reranking
  ingest.py        chunk, embed, and load the corpus
  generate.py      retrieved context + question to grounded answer
eval/
  questions.jsonl  45 hand-written questions with verified gold chunks
  verify.py        checks gold ids exist and are retrievable
  fix_gold.py      records gold id corrections applied after verification
  inspect.py       browse and search chunks
  metrics.py       recall@k, MRR, LLM-as-judge faithfulness, abstention
  run.py           runs one variant end to end, writes results JSON
  results/         one JSON per variant (not committed)
```

---

## Limitations

- Corpus is 10 papers in one domain; results may not transfer.
- 36 scoreable questions is small — differences of a few points are not meaningful, and several results here are within that range.
- **The judge is the same model as the generator**, so it marks its own work and is biased toward fluent answers.
- Faithfulness measures whether claims trace to context, not whether the answer is *correct*.
- Claim-level faithfulness cannot detect fabricated relations between true facts — Finding 9.
- Chunks exceeding the embedder's 512-token limit are silently truncated.
- Generation metrics were measured for the baseline only; per-variant generation runs are pending.
- Gold chunks were assigned by hand. Where an answer spans adjacent chunks, a single gold id understates recall.

---

**Abdul Rafay Mohd** — M.S. Artificial Intelligence, University of North Texas
[GitHub](https://github.com/Mohd-Abdul-Rafay) · [LinkedIn](https://linkedin.com/in/mohd-abdul-rafay)
