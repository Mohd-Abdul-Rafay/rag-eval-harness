"""Chunk the corpus, embed it, and load it into the vector store."""

import argparse
from pathlib import Path

from src.chunker import chunk_corpus
from src.store import VectorStore
from src.embedder import Embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="data/papers")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--collection", default="papers")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                    help="embedding model; must match the one used at query time")
    ap.add_argument("--reset", action="store_true",
                    help="drop the collection before ingesting")
    ap.add_argument("--keep-refs", action="store_true",
                    help="keep reference sections (default: strip them)")
    args = ap.parse_args()

    print(f"chunking {args.papers} (size={args.chunk_size}, overlap={args.overlap}, "
          f"refs={'kept' if args.keep_refs else 'stripped'})")
    records = chunk_corpus(Path(args.papers), args.chunk_size, args.overlap,
                           drop_refs=not args.keep_refs)
    print(f"  {len(records)} chunks")

    embedder = Embedder(args.embed_model)
    print(f"embedder: {args.embed_model} ({embedder.dim}-dim, {embedder.device})")

    store = VectorStore(collection_name=args.collection, embedder=embedder)
    if args.reset:
        print("resetting collection")
        store.reset()

    print("embedding and storing")
    store.add(records)
    print(f"  collection '{args.collection}' now holds {store.count()} chunks")


if __name__ == "__main__":
    main()