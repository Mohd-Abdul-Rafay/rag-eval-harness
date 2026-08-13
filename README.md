# RAG Evaluation Harness

A retrieval-augmented generation pipeline over a corpus of computer vision papers, built around an evaluation harness that measures **retrieval quality and generation quality separately** — because when a RAG system returns a wrong answer, "the system is wrong" is not a diagnosis.

Runs entirely locally. No API keys, no paid services.

**Measured baseline: recall@1 is 28%.** Details below.

> **Status:** in progress. Pipeline works end to end and the 45-question eval set is verified against the corpus. The scoring harness and controlled comparison are in development. Findings below come from real runs, including the ones that didn't work.

---

## Why separate the metrics

A single end-to-end accuracy number tells you a RAG system failed without telling you *where*. There are two distinct failure modes and they need different fixes:

- **Retrieval failed** — the answer was never in the context. Fix chunking, embeddings, or `k`.
- **Generation failed** — the answer was in the context and the model ignored it, or blended it with parametric knowledge. Fix the prompt or the model.

This is not theoretical here. **Finding 3 is a case where retrieval ranked the correct chunk second behind an unrelated one and generation recovered anyway** — end-to-end accuracy 1.0, `recall@1` 0. A single aggregate number would have reported success and hidden a real retrieval problem.

---

## Stack

| Component | Choice | Why |
|---|---|---|
| Generation | Ollama + Llama 3.1 8B | Local, free, fast enough for hundreds of eval calls |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | 384-dim, runs on Apple MPS |
| Vector store | ChromaDB | Local persistence, cosine space, no server |
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

Run:

```bash
python -m src.ingest --reset          # chunk, embed, store
python -m src.generate                # end-to-end demo queries
python -m eval.verify                 # check gold ids and preview recall
python -m eval.inspect --search "..." # browse the corpus
```

---

## The eval set

45 hand-written questions in `eval/questions.jsonl`, each with a known gold chunk, a reference answer, and a type.

| Type | Count | Purpose |
|---|---|---|
| `answerable` | 28 | Normal case — the answer sits in one identifiable chunk |
| `distractor` | 8 | Answer exists, but other chunks look more relevant |
| `unanswerable` | 9 | Tests abstention; includes near-misses where the right paper is present but the fact is not |

Questions are phrased as a user would ask them, never copied from the papers. Copying source wording makes retrieval trivially easy and the resulting metrics meaningless.

**Gold IDs were verified, not assumed.** An initial pass flagged 17 questions whose gold chunk did not appear in the top 20. Manual inspection showed 13 were mislabelled by me and 4 were genuine retrieval failures. `eval/verify.py` re-runs this check; `eval/fix_gold.py` records the corrections that were applied. Unverified ground truth would have made every number below wrong.

---

## Baseline results

Dense retrieval, `all-MiniLM-L6-v2`, chunk size 512, overlap 50, references stripped. 32 questions with verified gold chunks:

| Metric | Result |
|---|---|
| **recall@1** | **9/32 (28%)** |
| **recall@5** | **18/32 (56%)** |
| recall@20 | 32/32 (100%) |
| Gold absent from top 20 | 4/36 (11%) |

Nearly three-quarters of questions fail to put the correct chunk first, and almost half fail at *k*=5. But every gold chunk that is found at all appears within the top 20 — the information is retrievable and the **ranking** is the problem. That profile is what reranking exists to fix, and it is the central hypothesis the comparison below tests.

---

## Build log

### Chunking

Recursive character splitting on paragraph, line, sentence, then word boundaries, with configurable overlap. Chunk size is a **variable in the comparison**, not a value copied from a tutorial, because the tradeoff is real: small chunks retrieve precisely but sever context; large chunks preserve context but produce embeddings averaged over too many ideas.

```
papers: 10
chunks: 188
avg words/chunk: 415
min: 63   max: 562
```

The average falls below target because the splitter respects paragraph boundaries. The max exceeds target because overlap is prepended after splitting — 562 words is roughly 730 tokens and the embedder truncates at 512.

### Embedding

`all-MiniLM-L6-v2`, 384 dimensions, on Apple MPS. Vectors normalised so cosine similarity reduces to a dot product.

**Finding 1 — the baseline embedder does not separate in-domain topics well.**

|  | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| **s0** augmentation with few samples | 1.000 | **0.463** | 0.168 | 0.413 |
| **s1** how does U-Net handle limited data | 0.463 | 1.000 | 0.070 | **0.453** |
| **s2** transformer self-attention | 0.168 | 0.070 | 1.000 | 0.258 |
| **s3** residual connections | 0.413 | 0.453 | 0.258 | 1.000 |

The intended result is visible: **s0–s1 = 0.463** despite near-zero vocabulary overlap. The unintended result is more useful: **s1–s3 = 0.453**, meaning "residual connections allow training of very deep networks" is judged nearly as relevant to the U-Net question as the correct answer. The embedder encodes *topic* more strongly than the specific semantic relation.

### Vector store and retrieval

ChromaDB, persistent local client, cosine space. Chroma returns cosine *distance*; the store converts to similarity explicitly, since inverting that would silently reverse every ranking.

**Finding 2 — the probe-level failure reproduces in the full pipeline.**

Top-5 for *"How does U-Net handle limited training data?"*:

| Rank | Similarity | Source | Relevant |
|---|---|---|---|
| 1 | 0.504 | ResNet #9 | No — convergence rates |
| 2 | 0.502 | **U-Net #1** | **Yes** |
| 3 | 0.489 | SimCLR #13 | No — results table |
| 4 | 0.486 | ResNet #14 | No — 110-layer convergence |
| 5 | 0.451 | U-Net #7 | Partial |

The correct chunk ranks second by 0.002. The spread across the whole top five is 0.053 — the embedder is barely discriminating.

### Grounded generation

Llama 3.1 8B at `temperature=0`. The system prompt requires answering only from numbered passages, citing passage numbers per claim, and emitting `INSUFFICIENT_CONTEXT` when the context cannot support an answer. Citations make unsupported claims visible to the faithfulness metric.

**Finding 3 — generation compensated for a retrieval failure.**

On the U-Net query, the model ignored the top-ranked ResNet passage and cited the correct one:

> According to [2], the U-Net architecture can handle very few training images...

End-to-end this looks like success while `recall@1` is 0. The answer is also incomplete — it names the outcome but not the mechanism (elastic deformation), which sat in a lower-ranked chunk and went uncited.

**Finding 4 — abstention held under plausible-but-wrong context.**

Asked about a paper not in the corpus, retrieval returned five confident-looking CV passages including the Transformer's warmup schedule. The model emitted `INSUFFICIENT_CONTEXT` and named what was missing.

| Question type | Top-1 similarity |
|---|---|
| Answerable from corpus | 0.50 |
| Not in corpus | 0.45 |

The margin is small but consistent, so a score threshold is added to the comparison as a testable pre-generation filter.

### Ingestion hygiene

Chunk inspection revealed reference sections embedded as ordinary content — one U-Net chunk was 433 words consisting of one line of body text followed by fourteen citations, and it had placed 5th for the U-Net query. `strip_references()` cuts from the references heading onward, restricted to the last 40% of a document.

**Finding 5 — removing 16% of the corpus as noise did not change retrieval.**

| | Chunks | Avg words |
|---|---|---|
| References kept | 224 | 421 |
| References stripped | **188** | 415 |

| Rank | Before | After |
|---|---|---|
| 1 | 0.504 ResNet #9 | 0.504 ResNet #9 |
| 2 | 0.502 **U-Net #1** | 0.502 **U-Net #1** |
| 3 | 0.489 SimCLR #13 | 0.489 SimCLR #13 |
| 4 | 0.486 ResNet #14 | 0.486 ResNet #14 |
| 5 | 0.451 U-Net #7 *(refs)* | 0.441 ResNet #13 *(results table)* |

Ranks 1–4 identical to three decimals. The bibliography chunk was evicted and replaced by an equally irrelevant results table. The intervention did exactly what it was designed to do and produced no measurable improvement, because removing noise does not help when the remaining pool is equally irrelevant. The change is retained — 16% fewer vectors at no cost to ranking — but it rules out the cheap explanation.

### Eval set verification

**Finding 6 — the embedder repeatedly fails to match exact terminology.**

Three independent cases surfaced while locating gold chunks:

| Query | Target chunk contains | Result |
|---|---|---|
| "layer normalization before every block" | *"Layernorm (LN) is applied before every block"* verbatim | Gold absent from top 5 |
| "hierarchical feature maps various scales backbone" | *"constructs hierarchical feature maps"* verbatim | Zero Swin chunks returned; five EfficientNet chunks instead |
| "elastic deformation random displacement" | Section 3.1 Data Augmentation | Gold ranked #1 (0.440) — but the natural-language version of the same question drops it out of the top 20 entirely |

The layernorm chunk additionally ranked **3rd for an unrelated query about projection heads**. The embedder fails on near-identical text and succeeds on unrelated text.

This is the strongest argument in the project for hybrid retrieval: a BM25 sparse index would match these terms exactly and trivially. Dense retrieval alone is losing information that keyword search would preserve.

**Finding 7 — two augmentation questions each retrieve the other paper's augmentation content.**

q002 (U-Net augmentation) returns three SimCLR chunks. q017 (SimCLR augmentation) returns three *other* SimCLR chunks, not its own ablation. The embedder appears to hold "augmentation" as a topic direction without distinguishing whose augmentation is being discussed — consistent with Finding 1 at corpus scale.

### Scoring harness — *in progress*

---

## Planned comparison

One variable at a time, everything else fixed, so any difference is attributable to a single change. Baseline row is measured; the rest are pending.

| Variant | Chunk size | Top-k | Retrieval | recall@1 | recall@5 | MRR | Faithfulness | Abstention |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | 512 | 5 | dense (MiniLM) | **28%** | **56%** | — | — | — |
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
- `recall@k` — is a gold chunk in the top *k*? Reported at k=1 and k=5, since Findings 2 and 3 show these diverge sharply.
- `MRR` — mean reciprocal rank of the highest-ranked gold chunk.

**Generation**
- **Faithfulness** — is every claim supported by the retrieved context?
- **Answer relevance** — does it address the question asked?
- **Abstention correctness** — when the context genuinely lacks the answer, does the model say so?

Abstention is measured deliberately. Llama 3.1 8B abstains correctly on unknown-entity questions even *without* retrieval. The failure mode that matters in RAG is different, and Findings 2 and 6 show why: retrieval routinely surfaces plausible, on-topic, wrong chunks. Nine eval questions target exactly this, including cases where the correct paper is in the corpus but the specific fact is not.

---

## Repository layout

```
src/
  chunker.py       PDF to cleaned text to overlapping chunks, reference stripping
  embedder.py      text to normalised vectors (MPS-accelerated)
  store.py         ChromaDB persistence and k-NN search
  ingest.py        chunk, embed, and load the corpus
  generate.py      retrieved context + question to grounded answer
  app.py           FastAPI service
eval/
  questions.jsonl  45 hand-written questions with verified gold chunks
  verify.py        checks gold ids exist and are retrievable; previews recall
  fix_gold.py      records the gold id corrections applied after verification
  inspect.py       browse and search chunks
  metrics.py       retrieval + generation metrics
  run.py           runs one config, logs per-variant results
  configs/         one YAML per variant
data/papers/       source PDFs (not committed)
data/chroma/       vector store (not committed, regenerable)
```

---

## Limitations

- Corpus is 10 papers in one domain; results may not transfer to heterogeneous corpora.
- 32 scoreable questions is small — differences of a few points are not meaningful.
- Faithfulness scoring will use LLM-as-judge, which is imperfect and correlated with the generator.
- Chunks exceeding the embedder's 512-token limit are silently truncated.
- At 188 chunks Chroma uses exact search; results may shift once approximate nearest-neighbour indexing kicks in at scale.
- Gold chunks were assigned by hand. Where an answer is spread across adjacent chunks, a single gold id understates recall.

---

**Abdul Rafay Mohd** — M.S. Artificial Intelligence, University of North Texas
[GitHub](https://github.com/Mohd-Abdul-Rafay) · [LinkedIn](https://linkedin.com/in/mohd-abdul-rafay)
