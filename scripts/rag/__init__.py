"""Modular RAG components for chunking, embedding, indexing, retrieval, and generation."""

from .config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, CROSS_REF_K, EMBEDDING_MODEL, CHAT_MODEL
from .chunking import chunk_text, chunk_corpus
from .embedding import get_embeddings, embed_chunks
from .indexing import build_and_save_index, load_index
from .retrieval import retrieve, infer_source_hints
from .generation import generate_answer
from .pipeline import build_pipeline, sync_pipeline

__all__ = [
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "CROSS_REF_K",
    "EMBEDDING_MODEL",
    "CHAT_MODEL",
    "chunk_text",
    "chunk_corpus",
    "get_embeddings",
    "embed_chunks",
    "build_and_save_index",
    "load_index",
    "retrieve",
    "infer_source_hints",
    "generate_answer",
    "build_pipeline",
    "sync_pipeline",
]
