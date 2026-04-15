#!/usr/bin/env python3

"""Retrieval checker for inspecting top-k chunks and expected-hit status."""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path so we can import scripts
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag import (  # noqa: E402
    CROSS_REF_K,
    TOP_K,
    infer_source_hints,
    load_index,
    retrieve,
)


def _normalize_expected_refs(question_obj):
    source = question_obj.get("source")
    page = question_obj.get("page")
    if not source or page is None:
        return []

    if isinstance(source, list):
        sources = source
    else:
        sources = [source]

    if isinstance(page, list):
        pages = page
    else:
        pages = [page]

    refs = []
    for s, p in zip(sources, pages):
        refs.append((str(s), int(p)))
    return refs


def _retrieval_hit(results, expected_refs):
    if not expected_refs:
        return None
    found = {(item.get("source"), int(item.get("page", -1))) for item in results}
    return all(ref in found for ref in expected_refs)


def _augment_multi_source_results(query, index, chunks, base_results, source_hints):
    """Ensure at least one chunk per hinted source for multi-source questions."""
    if not source_hints:
        return base_results

    per_source_min = 2
    lower_q = query.lower()

    stop_words = {
        "what", "is", "are", "the", "a", "an", "and", "or", "vs", "versus", "of", "for", "to", "on",
        "in", "with", "show", "tell", "quick", "compare", "yes", "no", "listed", "page", "pages", "item",
        "results", "result", "shares",
    }
    intent_tokens = []
    for token in lower_q.replace("?", " ").replace(",", " ").split():
        if token in stop_words:
            continue
        if any(hint in token for hint in source_hints):
            continue
        intent_tokens.append(token)

    if "risk" in lower_q and "factors" not in intent_tokens:
        intent_tokens.append("factors")
    if "risk" in lower_q and "risk" not in intent_tokens:
        intent_tokens.append("risk")
    if "result" in lower_q and "results" not in intent_tokens:
        intent_tokens.append("results")

    present_hints = set()
    for item in base_results:
        source_name = item.get("source", "").lower()
        for hint in source_hints:
            if hint in source_name:
                present_hints.add(hint)

    missing_hints = set(source_hints) - present_hints
    if not missing_hints:
        return base_results

    merged = list(base_results)
    seen = {(r.get("source"), r.get("page"), r.get("text")) for r in base_results}

    for hint in sorted(missing_hints):
        focused_query = f"{hint} {' '.join(intent_tokens[:6])}".strip()
        if focused_query == hint:
            focused_query = f"{hint} results"
        focused = retrieve(
            focused_query,
            index,
            chunks,
            k=2,
            source_filter={hint},
            cross_reference_mode=True,
            strict_source_filter=True,
        )
        for item in focused:
            key = (item.get("source"), item.get("page"), item.get("text"))
            if key not in seen:
                merged.append(item)
                seen.add(key)

    # Guarantee at least N chunks per hinted source when available.
    best_per_hint = {}
    for hint in source_hints:
        hint_best = []
        for item in merged:
            source_name = item.get("source", "").lower()
            if hint in source_name:
                hint_best.append(item)
        hint_best.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        if hint_best:
            hint_best = hint_best[:per_source_min]
            best_per_hint[hint] = hint_best

    forced = []
    forced_keys = set()
    for hint in sorted(best_per_hint.keys()):
        for item in best_per_hint[hint]:
            key = (item.get("source"), item.get("page"), item.get("text"))
            if key not in forced_keys:
                forced.append(item)
                forced_keys.add(key)

    remaining = []
    for item in merged:
        key = (item.get("source"), item.get("page"), item.get("text"))
        if key not in forced_keys:
            remaining.append(item)
    remaining.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    final_k = max(CROSS_REF_K, len(source_hints) * per_source_min + 2)
    return (forced + remaining)[:final_k]


def _load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Inspect retrieval for one question.")
    parser.add_argument("--query", help="Question text to run")
    parser.add_argument("--question-id", type=int, help="1-based index from data/questions.json")
    parser.add_argument("--questions", default="data/questions.json", help="Path to questions.json")
    parser.add_argument("--index", default="data/my_index.faiss", help="Path to FAISS index")
    parser.add_argument("--chunks", default="data/chunks.json", help="Path to chunks JSON")
    parser.add_argument("--k", type=int, help="Override top-k")
    args = parser.parse_args()

    question_obj = None
    query = args.query

    if args.question_id is not None:
        questions = _load_questions(args.questions)
        if args.question_id < 1 or args.question_id > len(questions):
            raise ValueError(f"question-id must be between 1 and {len(questions)}")
        question_obj = questions[args.question_id - 1]
        query = question_obj["question"]

    if not query:
        parser.print_help()
        raise SystemExit(2)

    index, chunks = load_index(args.index, args.chunks)

    category = (question_obj or {}).get("category", "")
    lower_q = query.lower()
    source_hints = infer_source_hints(query)
    is_cross_ref = category == "cross-reference" or len(source_hints) >= 2
    k = args.k if args.k else (CROSS_REF_K if is_cross_ref else TOP_K)

    results = retrieve(
        query,
        index,
        chunks,
        k=k,
        source_filter=source_hints if is_cross_ref else None,
        cross_reference_mode=is_cross_ref,
    )
    if is_cross_ref and len(source_hints) >= 2:
        results = _augment_multi_source_results(query, index, chunks, results, source_hints)

    print("=" * 80)
    print("RETRIEVAL CHECKER")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Category: {category or 'manual'}")
    print(f"Source hints: {sorted(source_hints) if source_hints else 'none'}")
    print(f"Cross-reference mode: {is_cross_ref}")
    print(f"Top-k: {k}")

    expected_refs = _normalize_expected_refs(question_obj or {})
    if expected_refs:
        print(f"Expected refs: {expected_refs}")

    print("\nRetrieved chunks:")
    for i, item in enumerate(results, start=1):
        snippet = item.get("text", "")[:180].replace("\n", " ")
        print(
            f"{i}. {item.get('source')} p.{item.get('page')} "
            f"score={item.get('score', 0.0):.4f}\n"
            f"   {snippet}..."
        )

    hit = _retrieval_hit(results, expected_refs)
    if hit is not None:
        print(f"\nExpected chunks retrieved: {'YES' if hit else 'NO'}")

    print("=" * 80)


if __name__ == "__main__":
    main()
