"""OpenAI client utilities for RAG modules."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_openai_client():
    """Return an OpenAI client configured from environment variables."""
    timeout_sec = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=timeout_sec,
        max_retries=max_retries,
    )
