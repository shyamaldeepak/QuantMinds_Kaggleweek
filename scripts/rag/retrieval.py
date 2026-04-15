"""Retrieval helpers for query-time semantic search over FAISS."""

import argparse
import re
from typing import Dict, List, Set

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path

try:
    from .embedding import get_embeddings
    from .indexing import load_index
    from .config import TOP_K
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.rag.embedding import get_embeddings
    from scripts.rag.indexing import load_index
    from scripts.rag.config import TOP_K


def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _tokenize_list(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _infer_source_hints(query):
    """Infer likely target document(s) from query text."""
    q = query.lower()
    hints = {
        "apple": ["apple"],
        "jpmorgan": ["jpmorgan", "jp morgan"],
        "goldman": ["goldman", "goldman sachs"],
        "blackstone": ["blackstone", "black stone"],
        "fundamentals": ["fundamentals of finance", "fundamentals"],
        "principles": ["principles of finance", "principles"],
        "international": ["international finance", "theory and policy"],
    }

    matched = set()
    for source_key, aliases in hints.items():
        if any(alias in q for alias in aliases):
            matched.add(source_key)
    return matched


def _query_expansions(query, source_hints):
    """Create lightweight, deterministic query expansions for noisy user input."""
    variants = [query]
    q = query.lower().strip()
    compact = " ".join(q.split())
    if compact != q:
        variants.append(compact)

    if "results" in q and len(source_hints) >= 2:
        for hint in sorted(source_hints):
            variants.append(f"{hint} financial results")

    if len(source_hints) == 1 and "results" in q and "financial" not in q:
        variants.append(f"{next(iter(source_hints))} financial results")

    deduped = []
    seen = set()
    for v in variants:
        norm = " ".join(v.split())
        if norm and norm not in seen:
            deduped.append(norm)
            seen.add(norm)
    return deduped[:4]


_BM25_CACHE = {"chunks_id": None, "bm25": None, "tokenized_corpus": None}


def _get_bm25_model(chunks):
    """Build (or reuse) BM25 model for current chunks list."""
    chunks_id = id(chunks)
    if _BM25_CACHE["chunks_id"] == chunks_id and _BM25_CACHE["bm25"] is not None:
        return _BM25_CACHE["bm25"], _BM25_CACHE["tokenized_corpus"]

    tokenized_corpus = [_tokenize_list(chunk.get("text", "")) for chunk in chunks]
    bm25_model = BM25Okapi(tokenized_corpus)
    _BM25_CACHE["chunks_id"] = chunks_id
    _BM25_CACHE["bm25"] = bm25_model
    _BM25_CACHE["tokenized_corpus"] = tokenized_corpus
    return bm25_model, tokenized_corpus


def infer_source_hints(query):
    """Public helper to infer likely source hints from a query."""
    return _infer_source_hints(query)


def retrieve(
    query,
    index,
    chunks,
    k=TOP_K,
    source_filter=None,
    cross_reference_mode=False,
    strict_source_filter=False,
):
    """Retrieve top-k most relevant chunks for a query."""
    query_tokens = _tokenize(query)
    query_l = (query or "").lower()
    inferred_source_hints = _infer_source_hints(query)
    explicit_source_filter = {s.lower() for s in source_filter} if source_filter else set()

    if strict_source_filter and explicit_source_filter:
        # In strict mode, only honor the explicitly requested sources.
        source_hints = explicit_source_filter
    elif explicit_source_filter:
        source_hints = set(inferred_source_hints) | explicit_source_filter
    else:
        source_hints = inferred_source_hints
    query_variants = _query_expansions(query, source_hints)

    index_intent = (
        "table of contents" in query_l
        or "form 10-k index" in query_l
        or (" index" in query_l and "item" in query_l)
    )
    item_match = re.search(r"item\s+([0-9]+[a-z]?)", query_l)
    item_token = item_match.group(1) if item_match else None

    candidate_multiplier = 6 if cross_reference_mode else 4
    candidate_k = min(max(k * candidate_multiplier, k), index.ntotal)

    dense_score_by_idx: Dict[int, float] = {}
    dense_rank_by_idx: Dict[int, int] = {}
    for variant in query_variants:
        query_vec = np.array(get_embeddings([variant])).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, k=candidate_k)
        for rank, idx in enumerate(indices[0], start=1):
            if idx == -1:
                continue
            dense_score = float(scores[0][rank - 1])
            dense_score_by_idx[idx] = max(dense_score_by_idx.get(idx, -1.0), dense_score)
            if idx not in dense_rank_by_idx or rank < dense_rank_by_idx[idx]:
                dense_rank_by_idx[idx] = rank

    bm25_model, _ = _get_bm25_model(chunks)
    bm25_score_by_idx: Dict[int, float] = {}
    bm25_rank_by_idx: Dict[int, int] = {}
    for variant in query_variants:
        tokenized_query = _tokenize_list(variant)
        if not tokenized_query:
            continue
        bm25_scores = bm25_model.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-candidate_k:][::-1]
        for rank, idx in enumerate(top_indices, start=1):
            score = float(bm25_scores[idx])
            bm25_score_by_idx[idx] = max(bm25_score_by_idx.get(idx, -1e9), score)
            if idx not in bm25_rank_by_idx or rank < bm25_rank_by_idx[idx]:
                bm25_rank_by_idx[idx] = rank

    candidate_indices: Set[int] = set(dense_score_by_idx.keys()) | set(bm25_rank_by_idx.keys())
    if not candidate_indices:
        return []

    candidates = []
    rrf_k = 60
    for idx in candidate_indices:
        result = dict(chunks[idx])
        semantic_score = dense_score_by_idx.get(idx, 0.0)
        bm25_score = bm25_score_by_idx.get(idx, 0.0)

        chunk_tokens = _tokenize(result.get("text", ""))
        overlap_score = 0.0
        if query_tokens:
            overlap_score = len(query_tokens & chunk_tokens) / len(query_tokens)

        source_bonus = 0.0
        source_name = result.get("source", "").lower()
        if source_hints and any(hint in source_name for hint in source_hints):
            source_bonus = 0.12

        page_bonus = 0.0
        page_num = int(result.get("page", 0) or 0)
        if index_intent and page_num > 0 and page_num <= 6:
            # TOC/index entries are usually front-matter pages.
            page_bonus = 0.12

        item_bonus = 0.0
        if item_token:
            chunk_text_l = result.get("text", "").lower()
            if f"item {item_token}" in chunk_text_l:
                item_bonus = 0.08

        rrf_score = 0.0
        if idx in dense_rank_by_idx:
            rrf_score += 1.0 / (rrf_k + dense_rank_by_idx[idx])
        if idx in bm25_rank_by_idx:
            rrf_score += 1.0 / (rrf_k + bm25_rank_by_idx[idx])

        # Hybrid fusion: RRF ranking + semantic/lexical signals + source hints.
        rerank_score = (
            1.2 * rrf_score
            + 0.6 * semantic_score
            + 0.12 * overlap_score
            + source_bonus
            + page_bonus
            + item_bonus
        )
        result["score"] = semantic_score
        result["bm25_score"] = bm25_score
        result["rerank_score"] = rerank_score
        candidates.append(result)

    if strict_source_filter and source_hints:
        strict_candidates = []
        for item in candidates:
            source_name = item.get("source", "").lower()
            if any(hint in source_name for hint in source_hints):
                strict_candidates.append(item)
        candidates = strict_candidates

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    filtered = []
    unfiltered = []
    for item in candidates:
        source_name = item.get("source", "").lower()
        if source_hints and any(hint in source_name for hint in source_hints):
            filtered.append(item)
        else:
            unfiltered.append(item)

    if source_hints:
        results = (filtered + unfiltered)[:k]
    else:
        results = candidates[:k]

    for item in results:
        item.pop("rerank_score", None)
        item.pop("bm25_score", None)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run semantic retrieval against a saved FAISS index.")
    parser.add_argument("--query", required=True, help="User query for retrieval")
    parser.add_argument("--index", default="data/my_index.faiss", help="Path to FAISS index")
    parser.add_argument("--chunks", default="data/chunks.json", help="Path to chunks JSON")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-k retrieval")
    args = parser.parse_args()

    index, chunks = load_index(args.index, args.chunks)
    results = retrieve(args.query, index, chunks, k=args.k)

    print(f"Retrieved {len(results)} results")
    for i, item in enumerate(results, start=1):
        print(f"{i}. {item['source']} p.{item['page']} score={item['score']:.4f}")
        print(f"   {item['text'][:140]}...")
