"""UI theme and style assets for the Gradio app."""

from pathlib import Path

import gradio as gr

APP_THEME = gr.themes.Soft(
    primary_hue="green",
    neutral_hue="stone",
    spacing_size="md",
    radius_size="lg",
)

_CSS_PATH = Path(__file__).with_name("ui.css")
CUSTOM_CSS = _CSS_PATH.read_text(encoding="utf-8") if _CSS_PATH.exists() else ""
