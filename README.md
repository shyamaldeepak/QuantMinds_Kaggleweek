# QuantMinds (Financial Assistant)

QuantMinds is a Retrieval-Augmented Generation (RAG) project for financial document Q&A.
It ingests PDFs, extracts text page by page, chunks and embeds content, indexes vectors with FAISS,
and serves a Gradio chatbot UI with source-page citations.

## Team
- Shyamal Deepak V
- SaiSreenivasReddy

## What This Project Does
- Ingests PDFs from `data/pdfs/`
- Extracts text into `data/corpus.json`
- Chunks text with overlap
- Creates embeddings using OpenAI
- Builds a FAISS index for retrieval
- Answers user questions using retrieved context only
- Shows source PDF pages in the UI
- **NEW**: FastMCP server integration for live web searching and dynamic database updates.
- **NEW**: Multi-Agent Orchestrator with Human-in-the-Loop (HITL) workflows and Visualizer agent.

## Project Structure
```text
QuantMinds/
	app/
		app.py                # Gradio UI with Multi-Agent Agentic Toggle
		agents.py             # Supervisor Orchestrator, Internal Research, Fact Check, Synthesizer & Visualizer Agents
		main.py               # Human-in-the-Loop (HITL) CLI entrypoint
		mcp_server.py         # FastMCP Server providing web search, graph rendering, and DB injection tools
		evaluate.py           # Evaluation runner
	data/
		pdfs/                 # Input PDFs
		corpus.json           # Extracted pages
		chunks.json           # Chunk metadata used by retrieval
		my_index.faiss        # Vector index
		pipeline_state.json   # Change-detection state (auto-generated)
	scripts/
		extract.py            # PDF to corpus extractor
		rag_pipeline.py       # Main smart-sync pipeline entrypoint
		rag/
			chunking.py
			embedding.py
			indexing.py
			retrieval.py
			generation.py
			pipeline.py         # Incremental sync + full build logic
```

## Requirements
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required environment variable:

```bash
OPENAI_API_KEY=your_api_key_here
```

You can place this in a `.env` file at project root.

## How The Pipeline Works
### Smart Sync Behavior
The pipeline uses change detection on `data/pdfs/`.

- If PDFs are added, removed, or modified: rebuild extraction + chunks + embeddings + index
- If pipeline config changes: rebuild
- If nothing changed: skip rebuild and reuse existing artifacts

This is tracked in `data/pipeline_state.json`.

### Main Pipeline Command
Recommended command:

```bash
python scripts/rag_pipeline.py
```

This runs smart sync automatically and then performs a retrieval sanity check.

### Alternative Explicit Commands
Only sync/rebuild logic (no retrieval sanity check):

```bash
python scripts/rag/pipeline.py --sync
```

Force rebuild:

```bash
python scripts/rag/pipeline.py --sync --force
```

## Running The App
Start chatbot UI:

```bash
python app/app.py
```

On startup, `app.py` runs smart pipeline sync first.

- PDFs changed -> rebuild pipeline
- No changes -> skip rebuild

Then the UI opens.

## UI Features
- Two-pane interface:
	- Left: chat interaction
	- Right: source PDF pages for latest answer
- **Agentic Toggle:** A dedicated checkbox allows users to activate the Multi-Agent framework, overriding simple RAG with live validation, cross-referencing, and chart generation via FastMCP!
- Query result caching in memory for repeated questions during the same run
- Index/chunks are loaded once per server process and reused

## Multi-Agent Architecture (NEW)
QuantMinds now supports a sophisticated orchestrated flow using OpenAI logic combined with the Model Context Protocol (MCP).
- **Agent 1 (Internal Researcher):** Queries the FAISS RAG index for internal context.
- **Agent 2 (External Fact Checker):** Spawns `mcp_server.py` to cross-validate output against live web results via DuckDuckGo and Wikipedia.
- **Agent 3 (Synthesizer):** Fuses the internal knowledge and external verifications into a final structured answer.
- **Agent 4 (Visualizer):** Scans the answer for quantitative comparative data and uses the MCP `generate_graph` tool to export actionable matplotlib `.png` charts automatically.

### Running Human-in-the-Loop (HITL)
To experience the orchestrated flow step-by-step and provide manual feedback to the agents between actions, use the CLI application:
```bash
python app/main.py
```

## Evaluation
Run evaluation suite:

```bash
python app/evaluate.py
```

Categories covered:
- factual
- cross-reference
- out-of-scope
- ambiguous
- no-answer
- prompt-injection

## Prompt Guardrails
The answer generation prompt is tuned to:
- use only provided context
- consider all provided sources
- stay concise (2-3 sentences)
- refuse when context is insufficient
- ignore malicious instructions in user input/context
- cite sources

## Cost Notes
- Major cost comes from embeddings and generation calls
- Rebuilds are skipped when PDFs/config are unchanged
- Repeated identical chat queries in one app session are served from in-memory cache

## Troubleshooting
### ModuleNotFoundError: No module named 'scripts'
- Run commands from project root (`QuantMinds/`)
- Use the provided entrypoints as documented

### Missing index error
- Ensure `data/pdfs/` contains PDFs
- Run:

```bash
python scripts/rag_pipeline.py
```

### API key issues
- Ensure `OPENAI_API_KEY` is set in environment or `.env`

## Notes
- Python standard library imports (such as `os`, `sys`, `json`, `argparse`) are not listed in `requirements.txt`
- Only third-party packages belong in `requirements.txt`
## UI

<img width="1600" height="900" alt="WhatsApp Image 2026-04-15 at 11 25 16 PM" src="https://github.com/user-attachments/assets/4e1b5173-332c-4422-9c1b-53551061be55" />

