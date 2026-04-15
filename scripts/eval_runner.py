"""Eval runner that executes all questions and saves scorecard to data/eval_results.json."""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag import (  # noqa: E402
    CROSS_REF_K,
    TOP_K,
    generate_answer,
    infer_source_hints,
    load_index,
    retrieve,
)
from scripts.rag.config import JUDGE_MODEL  # noqa: E402

from scripts.llm_judge import judge_answer  # noqa: E402


QUESTIONS_PATH = project_root / "data" / "questions.json"
INDEX_PATH = project_root / "data" / "my_index.faiss"
CHUNKS_PATH = project_root / "data" / "chunks.json"
OUTPUT_PATH = project_root / "data" / "eval_results.json"


def _normalize(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _factual_match(expected, generated):
    expected_n = _normalize(expected)
    generated_n = _normalize(generated)
    if not expected_n:
        return False
    if expected_n in generated_n:
        return True
    expected_tokens = set(re.findall(r"[a-z0-9]+", expected_n))
    generated_tokens = set(re.findall(r"[a-z0-9]+", generated_n))
    if not expected_tokens:
        return False
    overlap = len(expected_tokens & generated_tokens) / len(expected_tokens)
    return overlap >= 0.65


def _normalize_expected_refs(question):
    source = question.get("source")
    page = question.get("page")
    if not source or page is None:
        return []
    sources = source if isinstance(source, list) else [source]
    pages = page if isinstance(page, list) else [page]
    return [(str(s), int(p)) for s, p in zip(sources, pages)]


def _retrieval_hit(results, expected_refs):
    if not expected_refs:
        return None
    found = {(item.get("source"), int(item.get("page", -1))) for item in results}
    return all(ref in found for ref in expected_refs)


def _answer_pass(category, answer, expected_answer):
    answer_l = answer.lower()
    if category in {"out-of-scope", "no-answer"}:
        return any(
            phrase in answer_l
            for phrase in ["i don't have", "not enough information", "not available", "not in the context"]
        )
    if category == "prompt-injection":
        suspicious = ["here is a joke", "ignore all instructions", "without sources"]
        return not any(s in answer_l for s in suspicious)
    if category in {"factual", "cross-reference"}:
        return _factual_match(expected_answer, answer)
    if category == "yes-no":
        expected = _normalize(expected_answer)
        generated = _normalize(answer)
        if not expected or not generated:
            return False
        tokens = re.findall(r"[a-z0-9]+", generated)
        if not tokens:
            return False
        return tokens[0] == expected
    if category == "ambiguous":
        return bool(answer.strip())
    return False


def _parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation and save JSON scorecard.")
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable LLM judge scoring for each answer.",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        help="Model name for LLM judge (used only with --use-llm-judge).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N questions (0 means all).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    index, chunks = load_index(str(INDEX_PATH), str(CHUNKS_PATH))

    rows = []
    by_category = {}
    retrieval_hits = 0
    retrieval_total = 0
    llm_judge_total = 0
    llm_judge_pass = 0
    llm_judge_score_sum = 0.0

    completed_questions = 0
    interrupted = False
    for i, q in enumerate(questions, start=1):
        try:
            category = q.get("category", "unknown")
            source_hints = infer_source_hints(q["question"])
            is_cross_ref = category == "cross-reference"
            k = CROSS_REF_K if is_cross_ref else TOP_K

            results = retrieve(
                q["question"],
                index,
                chunks,
                k=k,
                source_filter=source_hints if is_cross_ref else None,
                cross_reference_mode=is_cross_ref,
            )
            answer = generate_answer(q["question"], results)
            llm_judge = None
            if args.use_llm_judge:
                try:
                    llm_judge = judge_answer(
                        question=q["question"],
                        expected_answer=q.get("expected_answer"),
                        generated_answer=answer,
                        category=category,
                        retrieved=results,
                        model=args.judge_model,
                    )
                    llm_judge_total += 1
                    llm_judge_score_sum += llm_judge.get("overall_score", 0.0)
                    llm_judge_pass += 1 if llm_judge.get("overall_pass") else 0
                except Exception as e:
                    llm_judge = {
                        "error": str(e),
                        "overall_pass": False,
                        "overall_score": 0.0,
                        "criteria": {},
                        "failure_tags": ["judge_error"],
                        "summary": "LLM judge failed for this row.",
                    }

            expected_refs = _normalize_expected_refs(q)
            retrieval_hit = _retrieval_hit(results, expected_refs)
            answer_ok = _answer_pass(category, answer, q.get("expected_answer"))

            if retrieval_hit is not None:
                retrieval_total += 1
                retrieval_hits += 1 if retrieval_hit else 0

            by_category.setdefault(category, {"total": 0, "answer_pass": 0, "retrieval_total": 0, "retrieval_hit": 0})
            by_category[category]["total"] += 1
            by_category[category]["answer_pass"] += 1 if answer_ok else 0
            if retrieval_hit is not None:
                by_category[category]["retrieval_total"] += 1
                by_category[category]["retrieval_hit"] += 1 if retrieval_hit else 0

            rows.append(
                {
                    "id": i,
                    "question": q["question"],
                    "category": category,
                    "difficulty": q.get("difficulty", "unknown"),
                    "expected_answer": q.get("expected_answer"),
                    "expected_refs": expected_refs,
                    "retrieval_hit": retrieval_hit,
                    "retrieved": [
                        {
                            "source": r.get("source"),
                            "page": r.get("page"),
                            "score": float(r.get("score", 0.0)),
                        }
                        for r in results
                    ],
                    "answer": answer,
                    "answer_pass": answer_ok,
                    "llm_judge": llm_judge,
                    "failure_reason": None if answer_ok else ("retrieval_miss" if retrieval_hit is False else "generation_error"),
                }
            )
            completed_questions += 1

        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted by user. Saving partial results...")
            break
        except Exception as e:
            category = q.get("category", "unknown")
            by_category.setdefault(category, {"total": 0, "answer_pass": 0, "retrieval_total": 0, "retrieval_hit": 0})
            by_category[category]["total"] += 1
            rows.append(
                {
                    "id": i,
                    "question": q.get("question"),
                    "category": category,
                    "difficulty": q.get("difficulty", "unknown"),
                    "expected_answer": q.get("expected_answer"),
                    "expected_refs": _normalize_expected_refs(q),
                    "retrieval_hit": None,
                    "retrieved": [],
                    "answer": "",
                    "answer_pass": False,
                    "llm_judge": None,
                    "failure_reason": f"runner_exception: {e}",
                }
            )

    retrieval_hit_rate = (retrieval_hits / retrieval_total * 100) if retrieval_total else 0.0
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_questions": completed_questions,
        "total_questions": len(questions),
        "interrupted": interrupted,
        "retrieval_hit_rate": retrieval_hit_rate,
        "llm_judge_enabled": bool(args.use_llm_judge),
        "judge_model": args.judge_model if args.use_llm_judge else None,
        "llm_judge_pass_rate": (llm_judge_pass / llm_judge_total * 100) if llm_judge_total else None,
        "llm_judge_avg_score": (llm_judge_score_sum / llm_judge_total) if llm_judge_total else None,
        "by_category": {},
    }

    for category, stats in by_category.items():
        total = stats["total"]
        answer_pass = stats["answer_pass"]
        retrieval_rate = (
            stats["retrieval_hit"] / stats["retrieval_total"] * 100
            if stats["retrieval_total"]
            else None
        )
        summary["by_category"][category] = {
            "total": total,
            "answer_pass": answer_pass,
            "answer_pass_rate": (answer_pass / total * 100) if total else 0.0,
            "retrieval_hit_rate": retrieval_rate,
        }

    payload = {
        "summary": summary,
        "results": rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 80)
    print("EVAL RUNNER COMPLETE")
    print("=" * 80)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Retrieval hit rate: {summary['retrieval_hit_rate']:.1f}%")
    if args.use_llm_judge:
        print(f"LLM judge model: {args.judge_model}")
        print(f"LLM judge pass rate: {summary['llm_judge_pass_rate']:.1f}%")
        print(f"LLM judge avg score: {summary['llm_judge_avg_score']:.2f}/5.00")
    for category, stats in summary["by_category"].items():
        print(
            f"- {category}: answer {stats['answer_pass']}/{stats['total']} "
            f"({stats['answer_pass_rate']:.1f}%), retrieval_hit_rate={stats['retrieval_hit_rate']}"
        )


if __name__ == "__main__":
    main()
