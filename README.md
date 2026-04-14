# QuantMinds (Financial Assistant)

QuantMinds is a Retrieval-Augmented Generation (RAG) project for financial document Q&A.
It ingests PDFs, extracts text page by page, chunks and embeds content, indexes vectors with FAISS,
and serves a Gradio chatbot UI with source-page citations.

## Team
- Sai Sreenivas Putta
- Shyamal Deepak Vempadapu
- Emmanuel Nischay Gapti
- Gopi Chandu Pallapu

## What This Project Does
- Ingests PDFs from `data/pdfs/`
- Extracts text into `data/corpus.json`
- Chunks text with overlap
- Creates embeddings using OpenAI
- Builds a FAISS index for retrieval
- Answers user questions using retrieved context only
- Shows source PDF pages in the UI

## Project Structure
```text
QuantMinds/
	app/
		app.py                # Gradio UI
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
- Query result caching in memory for repeated questions during the same run
- Index/chunks are loaded once per server process and reused

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
