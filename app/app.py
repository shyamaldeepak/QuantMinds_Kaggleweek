"""
RAG Web UI - Gradio Interface
"""

import gradio as gr
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import sys

# Setup
load_dotenv()

# Add parent directory to path so we can import scripts
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag import (
    retrieve,
    generate_answer,
    load_index,
    sync_pipeline,
    TOP_K,
    CROSS_REF_K,
    infer_source_hints,
)
from app.ui_assets import APP_THEME, CUSTOM_CSS
from app.session_store import (
    DEFAULT_SOURCES_MARKDOWN,
    session_choices,
    get_or_create_session,
    create_session,
    delete_session,
    clear_session,
    save_session_chat,
)

# Index artifact paths
index_path = str(project_root / "data" / "my_index.faiss")
chunks_path = str(project_root / "data" / "chunks.json")

# Lazy-loaded runtime cache (loaded only when first query is submitted)
_index = None
_chunks = None
_query_cache = {}
REFUSAL_TEXT = "I don't have enough information to answer this."
SOURCE_DISPLAY_NAMES = {
    "apple": "Apple",
    "jpmorgan": "JPMorgan",
    "goldman": "Goldman Sachs",
    "blackstone": "Blackstone",
    "fundamentals": "Fundamentals of Finance",
    "principles": "Principles of Finance",
    "international": "International Finance",
}

RESULTS_STYLE_INSTRUCTION = (
    "For this answer, output only one single-line record with no extra prose. "
    "Use exactly this format: Metric: <metric>; 2024: <value or Not available>; 2025: <value or Not available>. "
    "Use ONLY provided context. Prefer metric priority in this order: net sales, net income, total revenue, operating income, total assets."
)


def initialize_chat_ui():
    sessions, active = get_or_create_session()
    choices = session_choices(sessions)
    return (
        gr.update(choices=choices, value=active.get("id")),
        active.get("id"),
        active.get("history", []),
        active.get("last_sources", DEFAULT_SOURCES_MARKDOWN),
    )


def load_selected_session(session_id):
    sessions, active = get_or_create_session(session_id)
    choices = session_choices(sessions)
    return (
        gr.update(choices=choices, value=active.get("id")),
        active.get("id"),
        active.get("history", []),
        active.get("last_sources", DEFAULT_SOURCES_MARKDOWN),
    )


def create_new_session():
    sessions, new_session = create_session()
    choices = session_choices(sessions)
    return (
        gr.update(choices=choices, value=new_session["id"]),
        new_session["id"],
        [],
        DEFAULT_SOURCES_MARKDOWN,
        "",
    )


def delete_current_session(session_id):
    remaining, active = delete_session(session_id)
    choices = session_choices(remaining)
    return (
        gr.update(choices=choices, value=active.get("id")),
        active.get("id"),
        active.get("history", []),
        active.get("last_sources", DEFAULT_SOURCES_MARKDOWN),
        "",
    )


def clear_current_session(session_id):
    _query_cache.clear()
    sessions = clear_session(session_id)
    choices = session_choices(sessions)
    return (
        gr.update(choices=choices, value=session_id),
        session_id,
        [],
        DEFAULT_SOURCES_MARKDOWN,
        "",
    )


def ensure_pipeline_synced():
    """Run smart sync before UI launch: rebuild only if PDFs/config changed."""
    rebuilt = sync_pipeline(project_root=project_root, output_dir="data", pdfs_dir="data/pdfs")
    print(f"Pipeline sync completed. Rebuilt: {rebuilt}")


def get_loaded_index_and_chunks():
    """Load index/chunks once per server process, then reuse in memory."""
    global _index, _chunks
    if _index is not None and _chunks is not None:
        return _index, _chunks

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Run: python scripts/rag_pipeline.py"
        )

    _index, _chunks = load_index(index_path, chunks_path)
    print(f"Loaded index once: {_index.ntotal} vectors and {len(_chunks)} chunks")
    return _index, _chunks


def format_sources(results):
    """Create right-panel source page summary from retrieval results."""
    if not results:
        return "No source pages available."

    lines = ["### Source Pages"]
    seen = set()
    for result in results:
        source_key = (result["source"], result["page"])
        if source_key in seen:
            continue
        lines.append(f"- **{result['source']}** - page {result['page']} ({result['score']:.2%})")
        seen.add(source_key)
    return "\n".join(lines)


def _augment_multi_source_results(message, index, chunks, base_results, source_hints):
    """Ensure at least one chunk per hinted source for multi-source questions."""
    if not source_hints:
        return base_results

    per_source_min = 2
    lower_msg = message.lower()

    # Build intent tokens so source-focused backfill retrieves relevant sections (e.g., risk factors).
    stop_words = {
        "what", "is", "are", "the", "a", "an", "and", "or", "vs", "versus", "of", "for", "to", "on",
        "in", "with", "show", "tell", "quick", "compare", "yes", "no", "listed", "page", "pages", "item",
        "results", "result", "shares",
    }
    intent_tokens = []
    for token in lower_msg.replace("?", " ").replace(",", " ").split():
        if token in stop_words:
            continue
        if any(hint in token for hint in source_hints):
            continue
        intent_tokens.append(token)

    if "risk" in lower_msg and "factors" not in intent_tokens:
        intent_tokens.append("factors")
    if "risk" in lower_msg and "risk" not in intent_tokens:
        intent_tokens.append("risk")
    if "result" in lower_msg and "results" not in intent_tokens:
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


def _build_intent_tokens(message, source_hints):
    lower_msg = message.lower()
    stop_words = {
        "what", "is", "are", "the", "a", "an", "and", "or", "vs", "versus", "of", "for", "to", "on",
        "in", "with", "show", "tell", "quick", "compare", "yes", "no", "listed", "page", "pages", "item",
    }
    intent_tokens = []
    for token in lower_msg.replace("?", " ").replace(",", " ").split():
        if token in stop_words:
            continue
        if any(hint in token for hint in source_hints):
            continue
        intent_tokens.append(token)

    if "risk" in lower_msg and "factors" not in intent_tokens:
        intent_tokens.append("factors")
    if "risk" in lower_msg and "risk" not in intent_tokens:
        intent_tokens.append("risk")
    if "result" in lower_msg and "results" not in intent_tokens:
        intent_tokens.append("results")

    return intent_tokens


def _parse_results_fields(answer_text):
    """Parse standardized results fields from a model response."""
    text = answer_text.replace("\n", " ")
    # Normalize chained records like "... 2025: X. Metric: Y" to a semicolon-separated form.
    text = re.sub(r"\.\s*(Metric|2024|2025)\s*:", r"; \1:", text, flags=re.IGNORECASE)

    metric_match = re.search(r"metric\s*:\s*([^;]+)", text, re.IGNORECASE)
    y24_match = re.search(r"2024\s*:\s*([^;]+)", text, re.IGNORECASE)
    y25_match = re.search(r"2025\s*:\s*([^;]+)", text, re.IGNORECASE)

    metric = metric_match.group(1).strip(" .") if metric_match else "Not available"
    y24 = y24_match.group(1).strip(" .") if y24_match else "Not available"
    y25 = y25_match.group(1).strip(" .") if y25_match else "Not available"
    metric_display = {
        "net sales": "Net Sales",
        "net income": "Net Income",
        "total revenue": "Total Revenue",
        "operating income": "Operating Income",
        "total assets": "Total Assets",
    }.get(metric.lower(), metric)
    return metric_display, y24, y25


def _answer_multi_source_query(message, index, chunks, source_hints):
    """Decompose query into per-source answers and combine into one response."""
    intent_tokens = _build_intent_tokens(message, source_hints)
    is_results_query = "result" in message.lower()
    available_sections = []
    missing_sections = []
    results_rows = []
    merged_results = []
    seen = set()

    for hint in sorted(source_hints):
        if is_results_query:
            focused_query = (
                f"{hint} 2024 2025 results net sales net income total revenue operating income total assets"
            )
        else:
            focused_query = f"{hint} {' '.join(intent_tokens[:8])}".strip()
        if focused_query == hint:
            focused_query = f"{hint} results"

        focused_results = retrieve(
            focused_query,
            index,
            chunks,
            k=4,
            source_filter={hint},
            cross_reference_mode=True,
            strict_source_filter=True,
        )

        for item in focused_results:
            key = (item.get("source"), item.get("page"), item.get("text"))
            if key not in seen:
                merged_results.append(item)
                seen.add(key)

        if not focused_results:
            missing_sections.append(hint)
            continue

        answer = generate_answer(
            focused_query,
            focused_results,
            expected_sources=[hint],
            extra_instruction=RESULTS_STYLE_INSTRUCTION if is_results_query else None,
        )
        if answer.strip() == REFUSAL_TEXT:
            missing_sections.append(hint)
            continue

        company = SOURCE_DISPLAY_NAMES.get(hint, hint.title())
        if is_results_query:
            metric, y24, y25 = _parse_results_fields(answer)
            results_rows.append((company, metric, y24, y25))
        else:
            available_sections.append(f"- **{company}:** {answer}")

    if not available_sections and not results_rows:
        return REFUSAL_TEXT, merged_results

    if is_results_query:
        response_parts = [
            "I found the following company-wise results (best available metric per company):",
            "",
            "| Company | Metric | 2024 | 2025 |",
            "|---|---|---|---|",
        ]
        for company, metric, y24, y25 in results_rows:
            response_parts.append(f"| {company} | {metric} | {y24} | {y25} |")
    else:
        response_parts = [
            "I found the following company-wise information:",
            *available_sections,
        ]

    if missing_sections:
        missing_names = ", ".join(SOURCE_DISPLAY_NAMES.get(h, h.title()) for h in missing_sections)
        response_parts.append(
            f"I don't have enough information in the retrieved context for: {missing_names}."
        )

    if len(available_sections) >= 2 or len(results_rows) >= 2:
        response_parts.append("Comparison note: metrics may come from different report sections and may not be directly comparable.")

    return "\n\n".join(response_parts), merged_results


def _is_keyword_query(message):
    """Detect short keyword-style inputs like 'apple' or 'finance'."""
    cleaned = " ".join((message or "").strip().split())
    if not cleaned:
        return False
    words = cleaned.split(" ")
    if len(words) > 2:
        return False
    return all(re.fullmatch(r"[A-Za-z][A-Za-z\-]*", w) for w in words)


def _keyword_query_to_prompt(message):
    topic = " ".join(message.strip().split())
    return f"Give a concise 2-3 sentence overview of {topic} based only on the provided context."


from app.agents import Supervisor

_supervisor = None
def get_supervisor():
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor()
    return _supervisor

def chat_with_sources(message, history, use_agent_checkbox=False):
    """Handle chat interaction and update source panel."""
    history = history or []
    message = (message or "").strip()

    if not message:
        return history, "No source pages available.", ""

    try:
        if use_agent_checkbox:
            supervisor = get_supervisor()
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                answer = loop.run_until_complete(supervisor.run(message))
            except Exception as e:
                answer = f"Agent Error: {str(e)}"
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]
            return history, "Powered by Advanced Multi-Agent Workflow (Internal RAG + External Web MCP Verification).", ""

        cache_key = " ".join(message.lower().split())
        if cache_key in _query_cache:
            cached_answer, cached_sources = _query_cache[cache_key]
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": cached_answer},
            ]
            return history, cached_sources, ""

        index, chunks = get_loaded_index_and_chunks()
        lower_msg = message.lower()
        source_hints = infer_source_hints(message)
        is_cross_reference_query = any(
            phrase in lower_msg
            for phrase in ["compare", "vs", "versus", "both", "higher", "difference between"]
        )
        # Any multi-company question should use cross-reference mode.
        if len(source_hints) >= 2:
            is_cross_reference_query = True

        is_keyword_mode = _is_keyword_query(message)

        if is_cross_reference_query:
            results = retrieve(
                message,
                index,
                chunks,
                k=CROSS_REF_K,
                source_filter=source_hints,
                cross_reference_mode=True,
            )
            if len(source_hints) >= 2:
                results = _augment_multi_source_results(message, index, chunks, results, source_hints)
        else:
            results = retrieve(message, index, chunks, k=TOP_K)

        if not results:
            answer = "I couldn't find any relevant information in the corpus."
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]
            return history, "No source pages available.", ""

        if is_cross_reference_query and len(source_hints) >= 2:
            answer, merged_results = _answer_multi_source_query(message, index, chunks, source_hints)
            if merged_results:
                results = merged_results
        else:
            expected_sources = sorted(source_hints) if is_cross_reference_query and source_hints else None
            prompt_query = _keyword_query_to_prompt(message) if is_keyword_mode else message
            keyword_instruction = (
                "For short keyword prompts, provide a short descriptive overview grounded in the retrieved context. "
                "Do not refuse if the context clearly describes the topic. "
            ) if is_keyword_mode else None
            answer = generate_answer(
                prompt_query,
                results,
                expected_sources=expected_sources,
                extra_instruction=keyword_instruction,
            )
        sources_markdown = format_sources(results)
        # Do not cache refusal answers so users can re-try after query edits.
        if answer.strip() != REFUSAL_TEXT:
            _query_cache[cache_key] = (answer, sources_markdown)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, sources_markdown, ""

    except Exception as e:
        error_text = f"Error: {str(e)}"
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_text},
        ]
        return history, "No source pages available.", ""


def chat_with_session(message, history, session_id, use_agent_checkbox):
    updated_history, sources_markdown, cleared_input = chat_with_sources(message, history, use_agent_checkbox)
    sessions, active = save_session_chat(session_id, updated_history, sources_markdown)
    choices = session_choices(sessions)
    return (
        updated_history,
        sources_markdown,
        cleared_input,
        gr.update(choices=choices, value=active.get("id")),
        active.get("id"),
    )


def clear_chat_state():
    """Clear visible chat and in-memory query cache."""
    _query_cache.clear()
    return [], "### Source Pages\nNo source pages available.", ""


def toggle_history_panel(is_open):
    """Collapse/expand the left history panel."""
    new_state = not bool(is_open)
    label = "◀ Hide History" if new_state else "▶ Show History"
    return gr.update(visible=new_state), gr.update(value=label), new_state
    

# Create Gradio interface
with gr.Blocks(
    title="QuantMind (Financial Assistant)",
) as demo:
    current_session_id = gr.State(value=None)
    history_panel_open = gr.State(value=True)

    with gr.Column(elem_classes=["qm-shell"]):
        gr.Markdown("# QuantMind (Financial Assistant)", elem_classes=["qm-header", "qm-title"])
        with gr.Row(elem_classes=["qm-toolbar"]):
            history_toggle_btn = gr.Button("◀ Hide History", elem_classes=["qm-toggle-btn"])

        with gr.Row():
            with gr.Column(scale=3, visible=True) as history_col:
                with gr.Group(elem_classes=["qm-card"]):
                    gr.Markdown("### Previous Chats")
                    session_selector = gr.Radio(
                        choices=[],
                        value=None,
                        label="",
                        elem_classes=["qm-history-list"],
                    )
                    with gr.Row(elem_classes=["qm-history-actions"]):
                        new_chat_btn = gr.Button("New Chat", variant="primary")
                        delete_chat_btn = gr.Button("Delete")

            with gr.Column(scale=6):
                with gr.Group(elem_classes=["qm-card"]):
                    gr.Markdown("### Chat")
                    chatbot = gr.Chatbot(label="", elem_id="chatbox")
                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="Ask your financial question...",
                            lines=1,
                            scale=6,
                            show_label=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        use_agent_checkbox = gr.Checkbox(label="Use Multi-Agent Web Verification (Agentic)", value=False)
                        clear_btn = gr.Button("Clear Current")

            with gr.Column(scale=5):
                with gr.Group(elem_classes=["qm-card"]):
                    source_view = gr.Markdown("### Source Pages\nNo source pages available.", elem_id="source-panel")

        send_btn.click(
            fn=chat_with_session,
            inputs=[user_input, chatbot, current_session_id, use_agent_checkbox],
            outputs=[chatbot, source_view, user_input, session_selector, current_session_id],
        )

        user_input.submit(
            fn=chat_with_session,
            inputs=[user_input, chatbot, current_session_id, use_agent_checkbox],
            outputs=[chatbot, source_view, user_input, session_selector, current_session_id],
        )

        clear_btn.click(
            fn=clear_current_session,
            inputs=[current_session_id],
            outputs=[session_selector, current_session_id, chatbot, source_view, user_input],
        )

        session_selector.change(
            fn=load_selected_session,
            inputs=[session_selector],
            outputs=[session_selector, current_session_id, chatbot, source_view],
        )

        new_chat_btn.click(
            fn=create_new_session,
            inputs=None,
            outputs=[session_selector, current_session_id, chatbot, source_view, user_input],
        )

        delete_chat_btn.click(
            fn=delete_current_session,
            inputs=[current_session_id],
            outputs=[session_selector, current_session_id, chatbot, source_view, user_input],
        )

        history_toggle_btn.click(
            fn=toggle_history_panel,
            inputs=[history_panel_open],
            outputs=[history_col, history_toggle_btn, history_panel_open],
        )

        demo.load(
            fn=initialize_chat_ui,
            inputs=None,
            outputs=[session_selector, current_session_id, chatbot, source_view],
        )


if __name__ == "__main__":
    ensure_pipeline_synced()
    share_enabled = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(share=share_enabled, css=CUSTOM_CSS, theme=APP_THEME)
