"""Load PDFs and split them into overlapping chunks."""

from pathlib import Path
from typing import List, Dict
import re

from pypdf import PdfReader


def load_pdf(path: Path) -> str:
    """Extract all text from one PDF."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def clean_text(text: str) -> str:
    """Collapse whitespace and strip artifacts that add noise to embeddings."""
    text = re.sub(r"-\n(\w)", r"\1", text)      # rejoin hyphenated line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank-line runs
    text = re.sub(r"[ \t]{2,}", " ", text)      # collapse repeated spaces
    return text.strip()


def strip_references(text: str) -> str:
    """
    Cut everything from the references/bibliography heading onward.

    Reference sections mention every topic in a field without discussing any
    of them, so they embed as broadly relevant and answer nothing. Only cuts
    in the last 40% of the document, since papers sometimes use these words
    in body text.
    """
    patterns = [
        r"\nReferences\s*\n",
        r"\nREFERENCES\s*\n",
        r"\nBibliography\s*\n",
        r"\nAcknowledg(?:e)?ments?\s*\n",
    ]
    cut = len(text)
    for pat in patterns:
        for m in re.finditer(pat, text):
            if m.start() > len(text) * 0.6:
                cut = min(cut, m.start())
                break
    return text[:cut]


def split_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Recursive character splitting.

    chunk_size and overlap are measured in WORDS here, not tokens, to keep
    this dependency-free. Roughly 1 word ~= 1.3 tokens for English prose.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def _split(chunk: str, seps: List[str]) -> List[str]:
        if len(chunk.split()) <= chunk_size:
            return [chunk]
        if not seps:
            words = chunk.split()
            return [" ".join(words[i:i + chunk_size])
                    for i in range(0, len(words), chunk_size)]

        sep, rest = seps[0], seps[1:]
        parts = chunk.split(sep)

        out, buf = [], ""
        for part in parts:
            candidate = (buf + sep + part) if buf else part
            if len(candidate.split()) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                buf = part if len(part.split()) <= chunk_size else ""
                if not buf:
                    out.extend(_split(part, rest))
        if buf:
            out.append(buf)
        return out

    chunks = _split(text, separators)

    # add overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = " ".join(chunks[i - 1].split()[-overlap:])
            overlapped.append(prev_tail + " " + chunks[i])
        chunks = overlapped

    return [c.strip() for c in chunks if c.strip()]


def chunk_corpus(papers_dir: Path, chunk_size: int = 512,
                 overlap: int = 50, drop_refs: bool = True) -> List[Dict]:
    """Load every PDF in a directory and return chunks with metadata."""
    records = []
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        raw = load_pdf(pdf_path)
        cleaned = clean_text(raw)
        if drop_refs:
            cleaned = strip_references(cleaned)
        chunks = split_text(cleaned, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"{pdf_path.stem}_chunk{i}",
                "text": chunk,
                "source": pdf_path.name,
                "chunk_index": i,
            })
    return records


if __name__ == "__main__":
    papers = Path("data/papers")

    for drop in (False, True):
        records = chunk_corpus(papers, drop_refs=drop)
        lengths = [len(r["text"].split()) for r in records]
        label = "refs stripped" if drop else "refs kept"
        print(f"[{label}]  chunks: {len(records)}  "
              f"avg words: {sum(lengths) / len(lengths):.0f}  "
              f"min: {min(lengths)}  max: {max(lengths)}")