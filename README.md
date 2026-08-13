# RAG Evaluation Harness

A retrieval-augmented generation pipeline over a corpus of computer vision papers, built around an evaluation harness that measures **retrieval quality and generation quality separately** — because when a RAG system returns a wrong answer, "the system is wrong" is not a diagnosis.

Runs entirely locally. No API keys, no paid services.

**Measured baseline: retrieval fails half the time; generation is near-perfect on whatever it's given.** That split is the point of the project.

| | recall@1 | recall@5 | MRR | Abstention | Faithfulness |
|---|---|---|---|---|---|
| Baseline (dense, MiniLM, k=5) | **25%** | **50%** | **0.322** | 89% | 96.4% |

> **Status:** in progress. Pipeline, verified 45-question eval set, and scoring harness complete. Retrieval variants (mpnet, BM25, hybrid, reranking) in development.

---

## Why separate the metrics

A single end-to-end accuracy number tells you a RAG system failed without telling you *where*. There are two distinct failure modes needing different fixes:

- **Retrieval failed** — the answer was never in the context. Fix chunking, embeddings, or `k`.
- **Generation failed** — the answer was in the context and the model ignored it or blended it with parametric knowledge. Fix the prompt or the model.

The baseline shows why this matters. Retrieval puts the correct chunk first only 25% of the time, yet 97.7% of answers are judged faithful to whatever context they received. An aggregate score would have looked acceptable and pointed at nothing. Measured separately, the bottleneck is unambiguous and entirely upstream.

---

## Stack

| Component | Choice | Why |
|---|---|---|
| Generation | Ollama + Llama 3.1 8B | Local, free, fast enough for hundreds of eval calls |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | 384-dim, runs on Apple MPS |
| Vector store | ChromaDB | Local persistence, cosine space, no server |
| Judge | Llama 3.1 8B, temperature 0 | Same model as the generator — a limitation, see Finding 9 |
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

Install [Ollama](https://ollama.com/download), pull the model, download the corpus:

```bash
ollama pull llama3.1:8b

cd data/papers
for id in 1512.03385 1706.03762 2010.11929 2103.14030 2002.05709 \
          1905.11946 1505.04597 1506.02640 1708.02002 2005.12872; do
  curl -L -o "${id}.pdf" "https://arxiv.org/pdf/${id}"; sleep 2
done
cd ../..
```

Run:

```bash
python -m src.ingest --reset                        # chunk, embed, store
python -m eval.verify                               # check gold ids
python -m eval.run --name baseline --skip-generation # retrieval only (seconds)
python -m eval.run --name baseline                  # full eval (~13 min)
python -m eval.inspect --search "..."               # browse the corpus
```

---

## The eval set

45 hand-written questions in `eval/questions.jsonl`, each with a gold chunk, a reference answer, and a type.

| Type | Count | Purpose |
|---|---|---|
| `answerable` | 28 | The answer sits in one identifiable chunk |
| `distractor` | 8 | Answer exists, but other chunks look more relevant |
| `unanswerable` | 9 | Tests abstention, including near-misses where the right paper is present but the fact is not |

Questions are phrased as a user would ask them, never copied from the papers. Copying source wording makes retrieval trivially easy and the resulting metrics meaningless.

**Gold IDs were verified, not assumed.** An initial pass flagged 17 questions whose gold chunk did not appear in the top 20. Inspection showed 13 were mislabelled by me and 4 were genuine retrieval failures. Unverified ground truth would have made every number below wrong.

---

## Baseline results

Dense retrieval, `all-MiniLM-L6-v2`, chunk 512, overlap 50, k=5, references stripped. 36 scoreable questions, 45 total.

| Metric | Result |
|---|---|
| **recall@1** | **25%** (9/36) |
| **recall@5** | **50%** (18/36) |
| **MRR** | **0.322** |
| recall@20 | 100% of found golds |
| Abstention correct | 89% (8/9) |
| Faithfulness, answerable | 97.7% (22 judged) |
| Faithfulness, all non-abstained | 96.4% (28 judged) |

Every gold chunk that is found at all appears within the top 20 — the information is retrievable and the **ranking** is the problem. That profile is what reranking exists to fix, and it is the central hypothesis the comparison tests.

---

## Findings

### 1 — The baseline embedder does not separate in-domain topics

Cosine similarity across four probe sentences:

|  | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| **s0** augmentation with few samples | 1.000 | **0.463** | 0.168 | 0.413 |
| **s1** how does U-Net handle limited data | 0.463 | 1.000 | 0.070 | **0.453** |
| **s2** transformer self-attention | 0.168 | 0.070 | 1.000 | 0.258 |
| **s3** residual connections | 0.413 | 0.453 | 0.258 | 1.000 |

The intended result: **s0–s1 = 0.463** despite near-zero vocabulary overlap. The unintended one: **s1–s3 = 0.453**, meaning an unrelated sentence about residual connections scores nearly as relevant to the U-Net question as the correct answer. The embedder encodes *topic* more strongly than semantic relation.

### 2 — The probe-level failure reproduces at corpus scale

Top-5 for *"How does U-Net handle limited training data?"*:

| Rank | Similarity | Source | Relevant |
|---|---|---|---|
| 1 | 0.504 | ResNet #9 | No |
| 2 | 0.502 | **U-Net #1** | **Yes** |
| 3 | 0.489 | SimCLR #13 | No |
| 4 | 0.486 | ResNet #14 | No |
| 5 | 0.451 | U-Net #7 | Partial |

Correct chunk second by 0.002; total spread across the top five is 0.053.

### 3 — Generation compensated for a retrieval failure

On that query the model ignored the top-ranked ResNet passage and cited the correct one. End-to-end it looks like success while `recall@1` is 0. The answer was also incomplete — it named the outcome but not the mechanism, which sat in a lower-ranked chunk and went uncited.

### 4 — Abstention held under plausible-but-wrong context

Asked about a paper not in the corpus, retrieval returned five confident-looking CV passages. The model emitted `INSUFFICIENT_CONTEXT` and named what was missing.

| Question type | Top-1 similarity |
|---|---|
| Answerable from corpus | 0.50 |
| Not in corpus | 0.45 |

A score threshold looked promising as a cheap pre-generation filter. Finding 8 substantially weakens that.

### 5 — Removing 16% of the corpus as noise did not change retrieval

Reference sections were being embedded as content; one U-Net chunk was 433 words of bibliography and had placed 5th.

| | Chunks | Avg words |
|---|---|---|
| References kept | 224 | 421 |
| References stripped | **188** | 415 |

Ranks 1–4 identical to three decimals afterwards. The bibliography chunk was evicted from position 5 and replaced by a ResNet results table — equally irrelevant. The intervention did exactly what it was designed to do and produced no measurable improvement, because removing noise does not help when the remaining pool is equally irrelevant. Retained for the 16% index reduction, but it rules out the cheap explanation.

### 6 — The embedder repeatedly fails to match exact terminology

Three independent cases surfaced while locating gold chunks:

| Query | Target contains | Result |
|---|---|---|
| "layer normalization before every block" | *"Layernorm (LN) is applied before every block"* verbatim | Gold absent from top 5 |
| "hierarchical feature maps various scales backbone" | *"constructs hierarchical feature maps"* verbatim | Zero Swin chunks; five EfficientNet chunks |
| "elastic deformation random displacement" | Section 3.1 Data Augmentation | Gold ranked #1 — but the natural-language form of the same question drops it out of the top 20 |

The layernorm chunk additionally ranked **3rd for an unrelated projection-head query**. The embedder fails on near-identical text and succeeds on unrelated text. This is the strongest argument in the project for hybrid retrieval — BM25 would match these terms trivially.

### 7 — Augmentation questions retrieve each other's papers

q002 (U-Net augmentation) returns three SimCLR chunks. q017 (SimCLR augmentation) returns three *other* SimCLR chunks, not its own ablation. The embedder holds "augmentation" as a topic direction without distinguishing whose.

### 8 — The one hallucination fabricated a relation, not a fact

Asked *"How does ViT compare against Swin on COCO detection?"* — a comparison neither paper makes — the model answered:

> The Vision Transformer (ViT) and Swin Transformer are compared on COCO object detection, with Swin achieving a box AP of 58.7 and mask AP of 51.1 on the test-dev set, surpassing previous state-of-the-art by +2.7 box AP and +2.6 mask AP [4].

Every number is correct and correctly cited to a real Swin passage. The fabrication is the phrase *"are compared"* — a relationship asserted between two papers that the context never establishes.

This was the **only** abstention failure in 9 unanswerable questions, and its top-1 similarity was **0.661**, the highest of any unanswerable question — well above the 0.50 answerable average. The score-threshold filter proposed in Finding 4 would have passed it confidently, because both papers genuinely are in the corpus and the retrieved content genuinely is about COCO detection. The threshold catches out-of-corpus questions; it does not catch this.

### 9 — Faithfulness scoring gave that hallucination a perfect score

Initially the harness only judged faithfulness on `answerable` questions, so the single clearest hallucination in the run went unscored. After fixing the harness to judge every non-abstained answer:

```
faithfulness_failed_abstention: 1.0
```

**The judge marked the fabricated comparison fully faithful.** Correctly, by its own criteria — every claim traces to the retrieved context. What does not trace is the *relation* asserted between those claims.

Claim-level faithfulness scoring is structurally blind to this failure mode. It verifies that facts appear in the context; it cannot verify that the relationship asserted between them appears there. This is the failure that matters most in production, because it produces confident, well-cited, plausible answers that are wrong.

Detecting it would require entailment checking over the whole answer rather than claim by claim. That is out of scope here, but the limitation is measured rather than assumed — and it was only found by inspecting an individual failure instead of trusting the 96.4% aggregate.

---

## Planned comparison

One variable at a time, everything else fixed. Baseline measured; the rest pending.

| Variant | Chunk | k | Retrieval | recall@1 | recall@5 | MRR | Faithfulness | Abstention |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | 512 | 5 | dense (MiniLM) | **25%** | **50%** | **0.322** | **96.4%** | **89%** |
| Chunk 256 | 256 | 5 | dense (MiniLM) | — | — | — | — | — |
| Chunk 1024 | 1024 | 5 | dense (MiniLM) | — | — | — | — | — |
| Top-k 3 | 512 | 3 | dense (MiniLM) | — | — | — | — | — |
| Top-k 10 | 512 | 10 | dense (MiniLM) | — | — | — | — | — |
| Larger embedder | 512 | 5 | dense (mpnet) | — | — | — | — | — |
| BM25 only | 512 | 5 | sparse | — | — | — | — | — |
| Hybrid (RRF) | 512 | 5 | dense + BM25 | — | — | — | — | — |
| Hybrid + rerank | 512 | 5 | dense + BM25 + cross-encoder | — | — | — | — | — |
| Score threshold | 512 | 5 | dense + similarity cutoff | — | — | — | — | — |
| References kept | 512 | 5 | dense, no ref stripping | — | — | — | — | — |

---

## Metrics

**Retrieval**
- `recall@k` — is a gold chunk in the top *k*? Reported at k=1 and k=5, since these diverge by 25 points.
- `MRR` — mean reciprocal rank of the highest-ranked gold chunk.

**Generation**
- **Faithfulness** — does every claim trace to the retrieved context? Judged by LLM at temperature 0. See Finding 9 for what this misses.
- **Abstention correctness** — measured on unanswerable questions (did it decline?) and answerable ones (did it decline when it shouldn't have?).

---

## Repository layout

```
src/
  chunker.py       PDF to cleaned text to overlapping chunks, reference stripping
  embedder.py      text to normalised vectors (MPS-accelerated)
  store.py         ChromaDB persistence and k-NN search
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
data/papers/       source PDFs (not committed)
data/chroma/       vector store (not committed, regenerable)
```

---

## Limitations

- Corpus is 10 papers in one domain; results may not transfer.
- 36 scoreable questions is small — differences of a few points are not meaningful.
- **The judge is the same model as the generator**, so it is marking its own work and is biased toward accepting fluent answers.
- Faithfulness measures whether claims trace to context, not whether the answer is *correct*. A model can faithfully summarise a chunk that does not answer the question.
- Claim-level faithfulness cannot detect fabricated relations between true facts — see Finding 9.
- Chunks exceeding the embedder's 512-token limit are silently truncated.
- At 188 chunks Chroma uses exact search; results may shift under approximate indexing at scale.
- Gold chunks were assigned by hand. Where an answer spans adjacent chunks, a single gold id understates recall.

---

**Abdul Rafay Mohd** — M.S. Artificial Intelligence, University of North Texas
[GitHub](https://github.com/Mohd-Abdul-Rafay) · [LinkedIn](https://linkedin.com/in/mohd-abdul-rafay)
