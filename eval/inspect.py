"""Browse and search chunks so eval questions can be written against real IDs."""

import argparse
from src.store import VectorStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", help="find chunks similar to this text")
    ap.add_argument("--source", help="list all chunks from one paper, e.g. 1505.04597.pdf")
    ap.add_argument("--id", help="show one chunk in full by id")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--chars", type=int, default=300)
    args = ap.parse_args()

    store = VectorStore()

    if args.id:
        res = store.collection.get(ids=[args.id])
        if not res["ids"]:
            print(f"no chunk with id {args.id}")
            return
        print(f"id:     {res['ids'][0]}")
        print(f"source: {res['metadatas'][0]['source']}")
        print(f"index:  {res['metadatas'][0]['chunk_index']}")
        print(f"words:  {len(res['documents'][0].split())}\n")
        print(res["documents"][0])
        return

    if args.source:
        res = store.collection.get(where={"source": args.source})
        order = sorted(range(len(res["ids"])),
                       key=lambda i: res["metadatas"][i]["chunk_index"])
        print(f"{len(order)} chunks in {args.source}\n")
        for i in order:
            preview = " ".join(res["documents"][i].split())[:args.chars]
            print(f"--- {res['ids'][i]}")
            print(f"    {preview}...\n")
        return

    if args.search:
        for h in store.search(args.search, k=args.k):
            preview = " ".join(h["text"].split())[:args.chars]
            print(f"{h['similarity']:.3f}  {h['id']}")
            print(f"        {preview}...\n")
        return

    print(f"collection holds {store.count()} chunks")
    print("usage:")
    print("  python -m eval.inspect --source 1505.04597.pdf")
    print("  python -m eval.inspect --search 'elastic deformation augmentation'")
    print("  python -m eval.inspect --id 1505.04597_chunk7")


if __name__ == "__main__":
    main()