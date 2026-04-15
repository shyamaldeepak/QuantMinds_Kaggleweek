"""Chunking helpers for splitting corpus text into overlapping windows."""

import argparse
import json
from pathlib import Path

try:
    from .config import CHUNK_OVERLAP, CHUNK_SIZE
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])

        if end == text_length:
            break

        start += chunk_size - overlap

    return chunks


def chunk_corpus(corpus, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk all entries in corpus while preserving metadata."""
    chunks = []
    for entry in corpus:
        text_chunks = chunk_text(entry["text"], chunk_size, overlap)
        for chunk_text_str in text_chunks:
            chunks.append(
                {
                    "text": chunk_text_str,
                    "page": entry["page"],
                    "source": entry["source"],
                }
            )
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk corpus.json into overlapping chunks.")
    parser.add_argument("--corpus", default="data/corpus.json", help="Path to corpus JSON file")
    parser.add_argument("--output", default="data/chunks_only.json", help="Output path for chunked JSON")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap")
    args = parser.parse_args()

    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    chunks = chunk_corpus(corpus, chunk_size=args.chunk_size, overlap=args.overlap)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Chunking complete: {len(chunks)} chunks written to {output_path}")
