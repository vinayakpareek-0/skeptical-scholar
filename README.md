# Skeptical Scholar

Skeptical Scholar is a local research-assistant prototype for answering questions over a small scientific-paper corpus. It combines hybrid retrieval, reranking, evidence scoring, optional reasoning checks, and Groq-hosted answer generation.

The project is currently optimized for a local presentation workflow: run the backend locally, ask questions through the web UI, ingest more papers for new topics when needed, and keep latency low enough for recording. The deeper verification path is still available through `config.yaml`, but the default config now favors speed.

## What It Does

Given a question, the system:

1. Retrieves candidate chunks from a local SQLite paper store using BM25 and dense FAISS search.
2. Merges sparse and dense results with reciprocal rank fusion.
3. Reranks the candidates with a cross-encoder.
4. Scores evidence quality using retrieval score, chunk type, entity overlap, and contradiction signals.
5. Builds an evidence-grounded prompt and calls Groq for the final answer.
6. Optionally verifies the answer with an NLI model.

The answer includes citations to the retrieved paper chunks. If retrieval or reasoning confidence is too weak, the system returns an "I don't know" response instead of forcing an answer.

## Current Runtime Profile

The default `config.yaml` enables a faster runtime profile:

```yaml
runtime:
  fast_mode: true
  explain_idk_with_llm: false

retrieval:
  candidate_top_k: 10
  rerank_top_k: 3

reasoning:
  enable_entities: false
  enable_contradictions: false

generation:
  verify_answer: false
```

This keeps retrieval, reranking, confidence scoring, citations, and generation active, but skips the slowest CPU-bound checks: GLiNER entity extraction, pairwise NLI contradiction detection, and post-generation NLI verification.

For a fuller but slower run, set:

```yaml
reasoning:
  enable_entities: true
  enable_contradictions: true

generation:
  verify_answer: true
```

## Local Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Run the UI:

```bash
python app.py
```

The app launches a Gradio chat interface, usually at `http://localhost:7860`.

For the custom web UI:

```bash
python ui_server.py
```

This serves `web/index.html` and the local API at `http://127.0.0.1:8000`.

## Local Data

The local `data/` directory is ignored by Git. It stores:

- `data/db/arxiv.db`: SQLite paper and chunk store
- `data/processed/dense_index.index`: FAISS dense index
- `data/processed/chunk_ids.npy`: vector-to-chunk mapping
- `data/metadata/metadata_checkpoint.json`: fetched-paper metadata
- `data/raw/arxiv_papers/`: temporary PDF download directory

The current local database contains a small research corpus, not the open web. If a question is outside the local corpus, the system may refuse or retrieve weak evidence. Use query ingestion to add relevant papers.

## Add Papers For A New Topic

To fetch and ingest papers for an ad hoc topic:

```bash
python -m src.ingestion.query_ingest "retrieval augmented generation evaluation"
```

By default this fetches a small number of ArXiv and Semantic Scholar results, downloads PDFs, parses and chunks them, inserts them into SQLite, rebuilds the dense FAISS index, and clears runtime caches.

To insert papers without rebuilding the dense index:

```bash
python -m src.ingestion.query_ingest "medical question answering rag" --no-rebuild-index
```

Tune the query-ingestion defaults in `config.yaml`:

```yaml
query_ingestion:
  arxiv_max_results: 3
  arxiv_client_delay: 1
  semantic_scholar_max_results: 3
  semantic_scholar_min_citations: 0
  download_delay: 0.5
  rebuild_dense_index: true
```

This is meant for quickly expanding the local corpus around a topic. It is not a full web-scale crawler.

## Rebuild The Dense Index

If chunks already exist in SQLite and only the FAISS index needs rebuilding:

```bash
python -m src.retrieval.dense_retriever
```

## Full Corpus Pipeline

The original corpus-building pipeline still exists:

```bash
python -m src.ingestion.run_pipeline
```

It uses the static ArXiv and Semantic Scholar query lists in `config.yaml`. This is slower and better suited for batch corpus building than for interactive preparation.

## Project Structure

```text
app.py                    Gradio chat UI
config.yaml               Paths, models, thresholds, runtime settings
src/config.py             Config loader
src/runtime_cache.py      Process-level cache for heavy models/indexes
src/ingestion/            Fetch, parse, chunk, and store papers
src/retrieval/            BM25, dense FAISS search, hybrid merge, rerank
src/reasoning/            Chunk labels, optional entities/NLI, confidence
src/generation/           Prompting, Groq client, optional NLI verification
src/evaluation/           Earlier retrieval and generation evaluation files
hf-space/                 Separate Hugging Face Space deployment copy
web/                      Static custom web UI
ui_server.py              Local static UI server and /api/query endpoint
```

## Main Technologies

- Gradio for the current local UI
- SQLite for paper/chunk storage
- rank-bm25 for sparse retrieval
- sentence-transformers and FAISS for dense retrieval
- cross-encoder reranking
- optional GLiNER entity extraction
- optional DeBERTa NLI checks
- Groq for LLM generation

## Known Limits

- Quality depends on the local corpus. The system does not automatically search the live web at answer time.
- First query after startup still loads models and indexes. Later queries are faster because runtime components are cached.
- The faster runtime profile skips some verification checks. It is useful for presentation latency, but it is less strict than the full reasoning pipeline.
- The Hugging Face deployment copy under `hf-space/` is separate from the root app. Root changes should be mirrored there only when updating that deployment.
- A future Vercel UI can deploy the static `web/` frontend, but it still needs a separate backend for the Python RAG pipeline. Vercel's free frontend hosting is not a direct replacement for this local Python runtime.

## Evaluation Notes

Earlier evaluation files are in `src/evaluation/`. They tested a 20-query set across in-domain, out-of-domain, and adversarial prompts. Treat those results as historical for the original full pipeline and corpus state, not as a guarantee for every future ingested topic or fast-mode setting.
