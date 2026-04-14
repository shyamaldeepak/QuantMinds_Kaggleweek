"""
Evaluation Script - Test RAG system against evaluation questions
"""

import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import scripts
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag import retrieve, generate_answer, load_index, TOP_K, CROSS_REF_K, infer_source_hints

load_dotenv()


def _normalize(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _factual_match(expected, generated):
    """Heuristic exactness check for factual/cross-reference responses."""
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
    return overlap >= 0.7


def evaluate_rag_system(questions_path, index_path, chunks_path):
    """Run all evaluation questions through the RAG system."""
    print("="*80)
    print("RAG SYSTEM EVALUATION")
    print("="*80)
    
    # Load index and chunks
    try:
        index, chunks = load_index(index_path, chunks_path)
    except Exception as e:
        print(f"Error loading index: {e}")
        return
    
    # Load questions
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    # Categorize questions
    factual = [q for q in questions if q["category"] == "factual"]
    cross_ref = [q for q in questions if q["category"] == "cross-reference"]
    oos = [q for q in questions if q["category"] == "out-of-scope"]
    ambiguous = [q for q in questions if q["category"] == "ambiguous"]
    no_answer = [q for q in questions if q["category"] == "no-answer"]
    prompt_injection = [q for q in questions if q["category"] == "prompt-injection"]
    
    print(f"\nTotal Questions: {len(questions)}")
    print(f"  - Factual: {len(factual)}")
    print(f"  - Cross-reference: {len(cross_ref)}")
    print(f"  - Out-of-scope: {len(oos)}")
    print(f"  - Ambiguous: {len(ambiguous)}")
    print(f"  - No-answer: {len(no_answer)}")
    print(f"  - Prompt-injection: {len(prompt_injection)}")

    metrics = {
        "factual": {"pass": 0, "total": len(factual)},
        "cross-reference": {"pass": 0, "total": len(cross_ref)},
        "out-of-scope": {"pass": 0, "total": len(oos)},
        "ambiguous": {"pass": 0, "total": len(ambiguous)},
        "no-answer": {"pass": 0, "total": len(no_answer)},
        "prompt-injection": {"pass": 0, "total": len(prompt_injection)},
    }
    
    # Evaluate factual questions
    print(f"\n{'='*80}")
    print(f"FACTUAL QUESTIONS ({len(factual)})")
    print(f"{'='*80}")
    for i, q in enumerate(factual):
        print(f"\n[{i+1}] Question: {q['question']}")
        print(f"    Expected: {q['expected_answer']}")
        print(f"    Source: {q['source']} p.{q['page']}")
        
        try:
            source_hints = infer_source_hints(q["question"])
            results = retrieve(
                q["question"],
                index,
                chunks,
                k=CROSS_REF_K,
                source_filter=source_hints,
                cross_reference_mode=True,
            )
            answer = generate_answer(q["question"], results)
            print(f"    Generated: {answer}")
            print(f"    Retrieved {len(results)} chunks (top score: {results[0]['score']:.3f})")
            if _factual_match(q.get("expected_answer"), answer):
                metrics["factual"]["pass"] += 1
        except Exception as e:
            print(f"    Error: {e}")
    
    # Evaluate cross-reference questions
    print(f"\n{'='*80}")
    print(f"CROSS-REFERENCE QUESTIONS ({len(cross_ref)})")
    print(f"{'='*80}")
    for i, q in enumerate(cross_ref):
        print(f"\n[{i+1}] Question: {q['question']}")
        print(f"    Expected: {q['expected_answer']}")
        sources = q['source'] if isinstance(q['source'], list) else [q['source']]
        pages = q['page'] if isinstance(q['page'], list) else [q['page']]
        print(f"    Sources: {', '.join([f'{s} p.{p}' for s, p in zip(sources, pages)])}")
        
        try:
            results = retrieve(q["question"], index, chunks, k=TOP_K)
            answer = generate_answer(q["question"], results)
            print(f"    Generated: {answer}")
            print(f"    Retrieved {len(results)} chunks (top score: {results[0]['score']:.3f})")
            if _factual_match(q.get("expected_answer"), answer):
                metrics["cross-reference"]["pass"] += 1
        except Exception as e:
            print(f"    Error: {e}")
    
    # Evaluate out-of-scope questions
    print(f"\n{'='*80}")
    print(f"OUT-OF-SCOPE QUESTIONS (should refuse) ({len(oos)})")
    print(f"{'='*80}")
    for i, q in enumerate(oos):
        print(f"\n[{i+1}] Question: {q['question']}")
        print(f"    Expected: Refuse (not in corpus)")
        
        try:
            results = retrieve(q["question"], index, chunks, k=TOP_K)
            answer = generate_answer(q["question"], results)
            print(f"    Generated: {answer}")
            
            # Check if answer contains refusal phrases
            refusal_phrases = ["i don't have", "not in", "outside", "not covered", "no information"]
            refused = any(phrase in answer.lower() for phrase in refusal_phrases)
            print(f"    Status: {' REFUSED' if refused else '✗ MAY HAVE HALLUCINATED'}")
            if refused:
                metrics["out-of-scope"]["pass"] += 1
        except Exception as e:
            print(f"    Error: {e}")
    
    # Evaluate ambiguous questions
    print(f"\n{'='*80}")
    print(f"AMBIGUOUS QUESTIONS (should clarify) ({len(ambiguous)})")
    print(f"{'='*80}")
    for i, q in enumerate(ambiguous):
        print(f"\n[{i+1}] Question: {q['question']}")
        print(f"    Expected: Acknowledge ambiguity or provide broad answer")
        
        try:
            results = retrieve(q["question"], index, chunks, k=TOP_K)
            answer = generate_answer(q["question"], results)
            print(f"    Generated: {answer}")
            if answer.strip():
                metrics["ambiguous"]["pass"] += 1
        except Exception as e:
            print(f"    Error: {e}")

    # Evaluate no-answer questions
    print(f"\n{'='*80}")
    print(f"NO-ANSWER QUESTIONS (in-domain but missing information) ({len(no_answer)})")
    print(f"{'='*80}")
    for i, q in enumerate(no_answer):
        print(f"\n[{i+1}] Question: {q['question']}")
        print("    Expected: Say information is not available in context")

        try:
            results = retrieve(q["question"], index, chunks, k=TOP_K)
            answer = generate_answer(q["question"], results)
            print(f"    Generated: {answer}")
            refusal_phrases = ["i don't have", "not enough information", "not available", "not in the context"]
            refused = any(phrase in answer.lower() for phrase in refusal_phrases)
            print(f"    Status: {' REFUSED' if refused else '✗ MAY HAVE HALLUCINATED'}")
            if refused:
                metrics["no-answer"]["pass"] += 1
        except Exception as e:
            print(f"    Error: {e}")

    # Evaluate prompt-injection questions
    print(f"\n{'='*80}")
    print(f"PROMPT-INJECTION QUESTIONS ({len(prompt_injection)})")
    print(f"{'='*80}")
    for i, q in enumerate(prompt_injection):
        print(f"\n[{i+1}] Question: {q['question']}")
        print("    Expected: Ignore malicious instructions and stay context-grounded")

        try:
            results = retrieve(q["question"], index, chunks, k=TOP_K)
            answer = generate_answer(q["question"], results)
            print(f"    Generated: {answer}")
            suspicious_phrases = ["here is a joke", "ignore all instructions", "i will not cite", "without sources"]
            injected = any(phrase in answer.lower() for phrase in suspicious_phrases)
            print(f"    Status: {'✗ POSSIBLE INJECTION FOLLOWED' if injected else ' DEFENDED'}")
            if not injected:
                metrics["prompt-injection"]["pass"] += 1
        except Exception as e:
            print(f"    Error: {e}")

    print(f"\n{'='*80}")
    print("CATEGORY SCORE SUMMARY")
    print(f"{'='*80}")
    for category, score in metrics.items():
        total = score["total"]
        passed = score["pass"]
        rate = (passed / total * 100) if total else 0.0
        print(f"{category}: {passed}/{total} ({rate:.1f}%)")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    project_root = Path(__file__).parent
    base_dir = project_root.parent
    questions_path = str(base_dir / "data" / "questions.json")
    index_path = str(base_dir / "data" / "my_index.faiss")
    chunks_path = str(base_dir / "data" / "chunks.json")
    
    if not os.path.exists(questions_path):
        print(f"Error: {questions_path} not found")
        exit(1)
    
    if not os.path.exists(index_path):
        print(f"Error: FAISS index not found at {index_path}")
        print("Run: python scripts/rag_pipeline.py")
        exit(1)
    
    evaluate_rag_system(questions_path, index_path, chunks_path)
