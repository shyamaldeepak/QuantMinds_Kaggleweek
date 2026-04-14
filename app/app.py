"""
RAG Web UI - Gradio Interface
"""

import gradio as gr
import os
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

# Index artifact paths
index_path = str(project_root / "data" / "my_index.faiss")
chunks_path = str(project_root / "data" / "chunks.json")

# Lazy-loaded runtime cache (loaded only when first query is submitted)
_index = None
_chunks = None
_query_cache = {}


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


def chat_with_sources(message, history):
    """Handle chat interaction and update source panel."""
    history = history or []
    message = (message or "").strip()

    if not message:
        return history, "No source pages available.", ""

    try:
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
        is_cross_reference_query = any(
            phrase in lower_msg
            for phrase in ["compare", "vs", "versus", "both", "higher", "difference between"]
        )
        if is_cross_reference_query:
            source_hints = infer_source_hints(message)
            results = retrieve(
                message,
                index,
                chunks,
                k=CROSS_REF_K,
                source_filter=source_hints,
                cross_reference_mode=True,
            )
        else:
            results = retrieve(message, index, chunks, k=TOP_K)

        if not results:
            answer = "I couldn't find any relevant information in the corpus."
            _query_cache[cache_key] = (answer, "No source pages available.")
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]
            return history, "No source pages available.", ""

        answer = generate_answer(message, results)
        sources_markdown = format_sources(results)
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


APP_THEME = gr.themes.Soft(
    primary_hue="green",
    neutral_hue="stone",
    spacing_size="md",
    radius_size="lg",
)


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --qm-bg-1: #f6eefc;
    --qm-bg-2: #fde5ee;
    --qm-bg-3: #fff2cc;
    --qm-bg-4: #ece9ff;
    --qm-panel: rgba(255, 255, 255, 0.84);
    --qm-panel-strong: rgba(255, 255, 255, 0.94);
    --qm-text: #102018;
    --qm-subtle: #3f5a4d;
    --qm-accent: #1f8a5b;
    --qm-accent-2: #166a44;
    --qm-border: rgba(16, 32, 24, 0.12);
}

.gradio-container {
    font-family: 'Outfit', sans-serif !important;
    color: var(--qm-text) !important;
    background:
        radial-gradient(circle at 12% 8%, rgba(126, 87, 194, 0.38), transparent 36%),
        radial-gradient(circle at 88% 14%, rgba(244, 180, 0, 0.34), transparent 38%),
        radial-gradient(circle at 72% 82%, rgba(233, 30, 99, 0.22), transparent 42%),
        radial-gradient(circle at 24% 76%, rgba(149, 117, 205, 0.24), transparent 40%),
        linear-gradient(140deg, var(--qm-bg-1) 0%, var(--qm-bg-2) 38%, var(--qm-bg-3) 72%, var(--qm-bg-4) 100%);
    min-height: 100vh;
}

.qm-shell {
    max-width: 1280px;
    margin: 1rem auto;
    border: 1px solid var(--qm-border);
    background: var(--qm-panel);
    border-radius: 20px;
    backdrop-filter: blur(4px);
    box-shadow: 0 24px 50px rgba(15, 35, 26, 0.12);
    animation: fade-up 420ms ease-out;
}

.qm-header {
    padding: 1rem 1.2rem 0.7rem 1.2rem;
}

.qm-title {
    margin: 0;
    font-size: clamp(1.3rem, 2.4vw, 1.9rem);
    font-weight: 800;
    line-height: 1.2;
    color: #102018;
}

.qm-title h1 {
    margin: 0;
    color: #102018 !important;
    font-size: clamp(1.3rem, 2.4vw, 1.9rem);
    font-weight: 800;
}

.qm-card {
    background: var(--qm-panel-strong);
    border: 1px solid var(--qm-border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 22px rgba(15, 35, 26, 0.06);
}

.qm-card h3,
.qm-card h4 {
    margin-top: 0;
    margin-bottom: 0.45rem;
}

#chatbox {
    height: 68vh;
}

#source-panel {
    min-height: 68vh;
}

button.primary {
    background: linear-gradient(120deg, var(--qm-accent), var(--qm-accent-2)) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 8px 18px rgba(31, 138, 91, 0.28);
}

button.primary:hover {
    filter: brightness(1.04);
    transform: translateY(-1px);
}

@keyframes fade-up {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 768px) {
    .qm-shell {
        margin: 0.5rem;
        border-radius: 14px;
    }

    .qm-header {
        padding: 1rem 1rem 0.35rem 1rem;
    }

    .qm-card {
        padding: 0.85rem;
    }
}
"""
    

# Create Gradio interface
with gr.Blocks(
    title="QuantMind (Financial Assistant)",
) as demo:
    with gr.Column(elem_classes=["qm-shell"]):
        gr.Markdown("# QuantMind (Financial Assistant)", elem_classes=["qm-header", "qm-title"])

        with gr.Row():
            with gr.Column(scale=7):
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
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=5):
                with gr.Group(elem_classes=["qm-card"]):
                    source_view = gr.Markdown("### Source Pages\nNo source pages available.", elem_id="source-panel")

        send_btn.click(
            fn=chat_with_sources,
            inputs=[user_input, chatbot],
            outputs=[chatbot, source_view, user_input],
        )

        user_input.submit(
            fn=chat_with_sources,
            inputs=[user_input, chatbot],
            outputs=[chatbot, source_view, user_input],
        )

        clear_btn.click(
            fn=lambda: ([], "### Source Pages\nNo source pages available.", ""),
            inputs=None,
            outputs=[chatbot, source_view, user_input],
        )


if __name__ == "__main__":
    ensure_pipeline_synced()
    demo.launch(share=True, css=CUSTOM_CSS, theme=APP_THEME)
