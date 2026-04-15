"""High-level orchestration for end-to-end RAG pipeline build."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

try:
    from .chunking import chunk_corpus
    from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, CHAT_MODEL
    from .embedding import embed_chunks
    from .indexing import build_and_save_index
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.rag.chunking import chunk_corpus
    from scripts.rag.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, CHAT_MODEL
    from scripts.rag.embedding import embed_chunks
    from scripts.rag.indexing import build_and_save_index


def build_pipeline(corpus_path, output_dir):
    """Build the complete RAG pipeline: load -> chunk -> embed -> index."""
    print("=" * 60)
    print("RAG PIPELINE - BUILD")
    print("=" * 60)

    print("\nStep 1: Loading corpus...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} entries from corpus")

    print("\nStep 2: Chunking corpus...")
    chunks = chunk_corpus(corpus, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("\nStep 3: Embedding chunks...")
    embeddings = embed_chunks(chunks)
    print(f"Embedded {len(embeddings)} chunks")

    print("\nStep 4: Building FAISS index...")
    index = build_and_save_index(chunks, embeddings, output_dir)

    print("\n" + "=" * 60)
    print("PIPELINE BUILD COMPLETE")
    print("=" * 60)

    return index, chunks


def _current_pipeline_config():
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
    }


def _collect_pdf_manifest(pdfs_dir):
    pdfs_path = Path(pdfs_dir)
    if not pdfs_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdfs_path}")

    entries = []
    for pdf in sorted(pdfs_path.glob("*.pdf")):
        stat = pdf.stat()
        entries.append(
            {
                "name": pdf.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    serialized = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return entries, manifest_hash


def _load_previous_state(state_file):
    path = Path(state_file)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state_file, payload):
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_extraction(project_root):
    extract_script = Path(project_root) / "scripts" / "extract.py"
    print("\nStep 0: Extracting PDFs into corpus.json...")
    subprocess.run([sys.executable, str(extract_script)], cwd=str(project_root), check=True)


def sync_pipeline(project_root, output_dir="data", pdfs_dir="data/pdfs", force=False):
    """Rebuild pipeline only when PDF set/content or config changed."""
    project_root = Path(project_root)
    output_path = project_root / output_dir
    corpus_path = output_path / "corpus.json"
    index_path = output_path / "my_index.faiss"
    chunks_path = output_path / "chunks.json"
    state_file = output_path / "pipeline_state.json"
    pdfs_path = project_root / pdfs_dir

    manifest_files, manifest_hash = _collect_pdf_manifest(pdfs_path)
    if not manifest_files:
        raise FileNotFoundError(f"No PDF files found in {pdfs_path}")

    current_config = _current_pipeline_config()
    previous_state = _load_previous_state(state_file)

    artifacts_exist = corpus_path.exists() and index_path.exists() and chunks_path.exists()
    unchanged_manifest = (
        previous_state is not None and previous_state.get("pdf_manifest_hash") == manifest_hash
    )
    unchanged_config = (
        previous_state is not None and previous_state.get("pipeline_config") == current_config
    )

    if not force and artifacts_exist and unchanged_manifest and unchanged_config:
        print("No PDF/config changes detected. Skipping extraction and index rebuild.")
        print(f"Using existing artifacts from {output_path}")
        return False

    if force:
        print("Force rebuild enabled. Rebuilding pipeline.")
    else:
        print("Changes detected. Rebuilding extraction/chunking/embedding/index.")

    _run_extraction(project_root)
    build_pipeline(str(corpus_path), str(output_path))

    _save_state(
        state_file,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pdf_manifest_hash": manifest_hash,
            "pdf_files": manifest_files,
            "pipeline_config": current_config,
            "artifacts": {
                "corpus": str(corpus_path),
                "index": str(index_path),
                "chunks": str(chunks_path),
            },
        },
    )

    print(f"State updated at {state_file}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build full RAG pipeline from corpus to FAISS index.")
    parser.add_argument("--corpus", default="data/corpus.json", help="Path to corpus JSON")
    parser.add_argument("--output-dir", default="data", help="Output directory for index artifacts")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Auto-detect PDF/config changes and rebuild only when needed",
    )
    parser.add_argument(
        "--pdfs-dir",
        default="data/pdfs",
        help="Directory containing source PDFs (used with --sync)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full rebuild even if no changes are detected (used with --sync)",
    )
    args = parser.parse_args()

    if args.sync:
        project_root = Path(__file__).resolve().parents[2]
        sync_pipeline(
            project_root=project_root,
            output_dir=args.output_dir,
            pdfs_dir=args.pdfs_dir,
            force=args.force,
        )
        raise SystemExit(0)

    if not os.path.exists(args.corpus):
        print(f"Error: {args.corpus} not found")
        raise SystemExit(1)

    build_pipeline(args.corpus, args.output_dir)
