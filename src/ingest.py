"""Chunk the corpus, embed it, and load it into the vector store."""

import argparse
from pathlib import Path

from src.chunker import chunk_corpus
from src.store import VectorStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="data/papers")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--collection", default="papers")
    ap.add_argument("--reset", action="store_true",
                    help="drop the collection before ingesting")
    args = ap.parse_args()

    print(f"chunking {args.papers} (size={args.chunk_size}, overlap={args.overlap})")
    records = chunk_corpus(Path(args.papers), args.chunk_size, args.overlap)
    print(f"  {len(records)} chunks")

    store = VectorStore(collection_name=args.collection)
    if args.reset:
        print("resetting collection")
        store.reset()

    print("embedding and storing")
    store.add(records)
    print(f"  collection '{args.collection}' now holds {store.count()} chunks")


if __name__ == "__main__":
    main()