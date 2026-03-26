# greenpolicy_cited_rag

GreenPolicyHub is a **citation-first** (quote-grounded) Retrieval-Augmented Generation (RAG) demo for Chinese environmental policy text.
It focuses on a practical constraint that matters in compliance and policy research:

> **No citation, no claim.** Answers must be verifiable against quoted source passages.

This repository provides:

- A **single-page Streamlit UI** with two modes: **Direct RAG** vs **Agent**.
- A policy text ingestion + chunking pipeline (plain text / Markdown).
- A local vector index powered by **LlamaIndex + Chroma** and **bge-m3** embeddings.
- An “engineering-style Agent” that exposes its process (retrieval → generation → self-check → optional retry) with **quantitative diagnostics** and **run records**.

## Public repository notes (important)

This GitHub repository is intentionally **safe to publish**:

- **No real policy/legal corpora are included.**
  - Put your local documents under `data/raw/`.
  - By default, `data/raw/` is ignored by Git and will not be uploaded.
- **No run logs are included.**
  - Agent run records are written to `data/runs/`.
  - Bad case records are written to `data/badcases/`.
  - Both are ignored by Git by default.
- The background image `static/bg.jpg` is **optional and local** (ignored by default).

If you want to publish a demo video or screenshots, make sure your UI does not reveal private corpora paths or internal-only policies.

## Why “citation-first” RAG?

Environmental policy and regulatory text often has:

- frequent updates (draft vs final, national vs local implementation rules),
- strict wording requirements (one word can change obligations),
- high audit cost (someone must confirm “where does this statement come from?”).

This project is built around a simple principle: **answers are only useful if they can be checked**.
So the system forces itself to:

1) retrieve relevant passages,
2) answer using **numbered citations** like `[1][2]`,
3) output the quoted sources (with metadata) for review.

## What you can do with the demo

In the UI you can choose between two buttons:

### 1) Direct RAG

Direct RAG is the baseline:

- retrieves contexts from the vector index,
- generates an answer,
- returns citations (quoted passages).

### 2) Agent (auditable + measurable)

Agent mode is designed to feel different from the baseline:

- **Numbered citation enforcement**: the answer is expected to include `[1][2]...`.
- **Self-check + one retry**: if citations are insufficient, the agent expands recall and rewrites once.
- **Diagnostics that are computed by code**:
  - number of retrieved contexts,
  - number of unique sources/articles,
  - similarity score statistics,
  - citation coverage ratio,
  - timing trace per step.
- **Run records** (local JSONL): each run can be replayed/compared later.
- **Bad case marking** (local JSONL): one-click record with an optional note.

This is intended for iterative improvements: you can tune chunking, retrieval parameters, and prompts, then re-run the same question and compare traces.

## Repository layout

- `app.py` — Streamlit single-page UI (Direct RAG vs Agent)
- `policyhub/` — core library
  - `settings.py` — environment & configuration
  - `llm.py` — LLM provider configuration (Ollama / OpenAI / DeepSeek)
  - `documents.py` — loaders + chunking + metadata inference
  - `indexing.py` — build/load LlamaIndex + Chroma persistent store
  - `rag.py` — baseline RAG ask() and citation formatting
  - `agent.py` — agent workflow + diagnostics + JSONL run logging
  - `records.py` — JSONL append/read helpers
- `data/` — local artifacts
  - `raw/` — your private corpora (ignored)
  - `chroma/` — persistent vector store (ignored)
  - `index/` — extra index artifacts (ignored)
  - `runs/` — agent run logs (ignored)
  - `badcases/` — bad case logs (ignored)
  - `eval/` — evaluation samples
- `scripts/` — helper scripts (optional)
- `static/` — optional background image

## Requirements

- Python 3.10+ recommended
- Network access on first run (to download the embedding model)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional evaluation dependencies:

```bash
pip install -r requirements-eval.txt
```

## Configuration

Copy the environment template:

```bash
cp .env.example .env
```

### LLM providers

The code supports multiple providers through environment variables.

1) Ollama (local)

- `POLICYHUB_LLM_PROVIDER=ollama`
- `OLLAMA_MODEL=<your model name>`

If you hit timeouts on cold start:

- increase `OLLAMA_REQUEST_TIMEOUT`
- set `OLLAMA_KEEP_ALIVE=5m` to reduce cold starts

2) OpenAI

- `POLICYHUB_LLM_PROVIDER=openai`
- `OPENAI_API_KEY=...`

3) DeepSeek (OpenAI-compatible)

- `POLICYHUB_LLM_PROVIDER=deepseek`
- `DEEPSEEK_API_KEY=...`

### Embeddings

- Default embedding model: `BAAI/bge-m3`
- Device selection: `EMBEDDING_DEVICE=cpu|cuda|auto`

Notes:

- First run downloads the model and may take time.
- CUDA requires a working PyTorch CUDA setup on your machine.

## Add your documents

Put your policy text under:

```
data/raw/
```

Supported formats:

- `.txt`
- `.md`

Recommended file naming convention:

- `policy_id_versionLabel_YYYY-MM-DD.md`
- Example: `EcoEnvCode_final_2026-03-12.md`

The system infers metadata from file names (policy id, version label, version date). This metadata is shown alongside citations.

### Markdown “article-style” chunking

For law-code-like Markdown that contains an explicit per-article marker (Chinese law-code style “Article N”), the loader uses a structure-aware approach:

- reads headings (`#`/`##`/`###`/`####`) as chapter context,
- splits by the article marker (article granularity),
- applies a secondary split only if a single article is too long,
- keeps `heading_path` as metadata (shown in citations) while keeping the quoted text clean.

## Run the UI

Start Streamlit:

```bash
streamlit run app.py
```

How indexing works:

- The vector store is initialized lazily (on the first question).
- On a cold run, embedding + index load/build can take 30–180 seconds depending on hardware and corpus size.

## Understanding the outputs

### Answer card

Both modes render an answer “card” in the UI.

- In Direct RAG mode: the answer is a baseline RAG response.
- In Agent mode: the answer is expected to include numbered references like `[1]`.

### Citation panel (“Quoted sources”)

Each citation includes:

- a numbered id (`[1]`, `[2]`...),
- a title (derived from file),
- metadata fields if available (policy id, version label/date, heading path, article id),
- a quoted text snippet.

This is the audit surface: reviewers should be able to confirm key claims against quoted passages.

### Agent process panel (trace + diagnostics)

Agent mode shows a process panel before the final answer.

It includes:

- high-level guarantees (citation enforcement, self-check + retry),
- diagnostics computed by code (counts, score stats, coverage ratio),
- a step-by-step trace with timings.

This is intentionally observable so you can compare “before vs after” when you change chunking, top_k, or prompts.

## Local run logging and bad cases

When using Agent mode, the app can write records locally:

- Agent run logs: `data/runs/agent_runs.jsonl`
- Bad case records: `data/badcases/badcases.jsonl`

These files are ignored by default and are meant for your own iterative development.

Suggested workflow:

1) Ask a question that the system answers poorly.
2) Mark it as a bad case with a short note.
3) Adjust chunking / retrieval / prompt.
4) Re-run the same question and compare:
   - number of retries,
   - citation coverage ratio,
   - retrieved context distribution.

## Evaluation (RAGAS scaffolding)

The repository includes optional evaluation dependencies in `requirements-eval.txt`.

Install them (recommended in a separate venv):

- `pip install -r requirements-eval.txt`

Prepare an evaluation set as JSONL, each line like:

- `{ "question": "...", "ground_truth": "..." }` (ground_truth is optional)

Then run the included evaluator:

- `python scripts/run_eval.py --dataset data/eval/sample.jsonl`

It currently evaluates Direct RAG only and prints aggregated metrics such as:

- faithfulness
- answer relevancy
- context precision

The included `data/eval/sample.jsonl` is only a smoke-test placeholder.

## Troubleshooting

### “The UI does not reflect my latest changes”

Common cause: multiple Streamlit instances running on different ports.

- Stop old instances (or kill the process) and start a single one.
- Confirm the port you opened matches the process you launched.

### Ollama timeouts

Symptoms: `httpx.ReadTimeout` or very slow first response.

Actions:

- wait once (cold start),
- increase `OLLAMA_REQUEST_TIMEOUT`,
- set `OLLAMA_KEEP_ALIVE=5m`,
- switch provider to `deepseek` for faster remote inference.

### First run is slow

Expected on first use:

- embedding model download,
- corpus embedding,
- Chroma persistence.

Subsequent runs should be faster because the vector store persists under `data/chroma/` (ignored by default).

## Security and compliance notes

- Do not commit private corpora, internal policy documents, or proprietary PDFs.
- If you use a remote LLM provider, your prompts and retrieved context may be sent to that provider.
  - For sensitive corpora, prefer local inference (e.g., Ollama) and review provider policies.

## Roadmap (minimal, realistic)

This repo intentionally keeps the UI simple (single page). Reasonable next steps:

- add a small evaluation script for repeatable metrics,
- refine chunking rules for different policy formats,
- improve retrieval filters using metadata (policy_id / version_date) if needed.

## License

No legal/policy corpora are distributed with this repository.
Add your preferred code license if you plan to open-source it broadly.
