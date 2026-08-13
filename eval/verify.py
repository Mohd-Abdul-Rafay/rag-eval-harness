"""
Verify every gold_chunk_id in the eval set exists, and sanity-check whether
the gold chunk is plausibly retrievable for its question.

Run this BEFORE trusting any recall or MRR number. Chunk ids depend on PDF
extraction and split settings, so a hand-written eval set must be checked
against the corpus that actually exists.
"""

import json
from pathlib import Path
from src.store import VectorStore


def load_questions(path: str = "eval/questions.jsonl"):
    qs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def main():
    store = VectorStore()
    questions = load_questions()

    all_ids = set(store.collection.get()["ids"])
    print(f"corpus holds {len(all_ids)} chunks")
    print(f"eval set holds {len(questions)} questions\n")

    missing, unretrievable, ok = [], [], []

    for q in questions:
        gold = q["gold_chunk_ids"]

        if q["type"] == "unanswerable":
            if gold:
                print(f"[WARN] {q['id']} is unanswerable but has gold ids")
            continue

        bad = [g for g in gold if g not in all_ids]
        if bad:
            missing.append((q["id"], bad))
            continue

        # is the gold chunk anywhere in the top 20?
        hits = store.search(q["question"], k=20)
        ranked = [h["id"] for h in hits]
        best = min((ranked.index(g) + 1 for g in gold if g in ranked),
                   default=None)

        if best is None:
            unretrievable.append((q["id"], q["question"], gold))
        else:
            ok.append((q["id"], best))

    print("=" * 70)
    print(f"MISSING GOLD IDS ({len(missing)}) - these ids do not exist in the corpus")
    print("=" * 70)
    for qid, bad in missing:
        print(f"  {qid}: {bad}")
        # suggest what the question actually retrieves
        q = next(x for x in questions if x["id"] == qid)
        for h in store.search(q["question"], k=3):
            print(f"      candidate: {h['id']}  ({h['similarity']:.3f})")
            print(f"                 {' '.join(h['text'].split())[:120]}...")
        print()

    print("=" * 70)
    print(f"GOLD NOT IN TOP-20 ({len(unretrievable)}) - id exists but question may be mismatched")
    print("=" * 70)
    for qid, question, gold in unretrievable:
        print(f"  {qid}: {question}")
        print(f"      gold: {gold}")
        q = next(x for x in questions if x["id"] == qid)
        for h in store.search(q["question"], k=3):
            print(f"      top:  {h['id']}  ({h['similarity']:.3f})")
        print()

    print("=" * 70)
    print(f"VERIFIED ({len(ok)})")
    print("=" * 70)
    at1 = sum(1 for _, r in ok if r == 1)
    at5 = sum(1 for _, r in ok if r <= 5)
    print(f"  gold at rank 1:      {at1}/{len(ok)}")
    print(f"  gold in top 5:       {at5}/{len(ok)}")
    print(f"  gold in top 20:      {len(ok)}/{len(ok)}")
    print("\n  (this is a preview of baseline recall - the harness measures it properly)")


if __name__ == "__main__":
    main()
