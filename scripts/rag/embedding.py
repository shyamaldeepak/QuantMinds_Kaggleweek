"""Embedding utilities for converting text chunks into vectors."""

import argparse
import json

import numpy as np
from pathlib import Path

try:
    from .client import get_openai_client
    from .config import EMBEDDING_MODEL
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.rag.client import get_openai_client
    from scripts.rag.config import EMBEDDING_MODEL

client = get_openai_client()


def get_embeddings(texts, model=EMBEDDING_MODEL):
    """Get embeddings for a batch of texts using OpenAI API."""
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def embed_chunks(chunks, batch_size=50):
    """Embed all chunks in batches to manage API calls."""
    print(f"Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = get_embeddings(batch)
        embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunk JSON into a NumPy vector file.")
    parser.add_argument("--chunks", default="data/chunks_only.json", help="Path to chunk JSON file")
    parser.add_argument("--output", default="data/embeddings.npy", help="Output .npy path")
    parser.add_argument("--batch-size", type=int, default=50, help="Embedding batch size")
    args = parser.parse_args()

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = embed_chunks(chunks, batch_size=args.batch_size)
    vectors = np.array(embeddings, dtype="float32")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, vectors)
    print(f"Embedding complete: {vectors.shape[0]} vectors saved to {output_path}")
