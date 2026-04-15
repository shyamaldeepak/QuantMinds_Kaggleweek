"""Indexing helpers for creating and loading FAISS vector indexes."""

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np


def build_and_save_index(chunks, embeddings, output_dir):
    """Build FAISS index from embeddings and save to disk."""
    print("\nBuilding FAISS index...")

    vectors = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    print(f"Index built with {index.ntotal} vectors")

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "my_index.faiss")
    chunks_path = os.path.join(output_dir, "chunks.json")

    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Index saved to {index_path}")
    print(f"Chunks saved to {chunks_path}")

    return index


def load_index(index_path, chunks_path):
    """Load pre-built index and chunks from disk."""
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save FAISS index from chunks and embeddings.")
    parser.add_argument("--chunks", default="data/chunks_only.json", help="Path to chunk JSON file")
    parser.add_argument("--embeddings", default="data/embeddings.npy", help="Path to embeddings .npy")
    parser.add_argument("--output-dir", default="data", help="Output directory for index and chunks")
    args = parser.parse_args()

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(args.embeddings)

    index = build_and_save_index(chunks, embeddings, args.output_dir)
    print(f"Indexing complete: {index.ntotal} vectors in FAISS index")
