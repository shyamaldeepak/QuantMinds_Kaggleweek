"""RAG pipeline compatibility entrypoint that composes modular task files."""

import json
import os
from pathlib import Path
import sys

try:
    from scripts.rag import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        build_pipeline,
        generate_answer,
        load_index,
        retrieve,
        sync_pipeline,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.rag import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        build_pipeline,
        generate_answer,
        load_index,
        retrieve,
        sync_pipeline,
    )


def track_cost(response, is_embedding=False):
    """Track API costs."""
    usage = response.usage
    if is_embedding:
        cost = usage.total_tokens * 0.02 / 1_000_000
    else:
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000

    cost_file = "cost_tracker.json"
    if os.path.exists(cost_file):
        with open(cost_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"total": 0.0}

    data["total"] += cost
    with open(cost_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"This call: ${cost:.6f} | Team total: ${data['total']:.4f} / $5.00")
    return data["total"]


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output_dir = "data"

    # Smart sync: only rebuild when PDFs/config change or artifacts are missing.
    rebuilt = sync_pipeline(project_root=project_root, output_dir=output_dir, pdfs_dir="data/pdfs")

    # Load resulting artifacts for retrieval sanity check.
    data_dir = project_root / output_dir
    index, chunks = load_index(str(data_dir / "my_index.faiss"), str(data_dir / "chunks.json"))
    
    # Test retrieval
    print("\n" + "="*60)
    print("TESTING RETRIEVAL")
    print("="*60)
    test_query = "What is machine learning?"
    print(f"\nQuery: {test_query}")
    results = retrieve(test_query, index, chunks, k=3)
    print(f"Retrieved {len(results)} results:")
    print(f"Rebuilt this run: {rebuilt}")
    for i, result in enumerate(results):
        print(f"\n{i+1}. (similarity: {result['score']:.3f}) {result['source']} p.{result['page']}")
        print(f"   {result['text'][:100]}...")
    
    print("\n" + "="*60)
