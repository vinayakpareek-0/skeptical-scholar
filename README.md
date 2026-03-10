# Skeptical Scholar

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/Vinayak-0/skeptical-scholar)

**Robust Neuro-Symbolic RAG with Uncertainty Estimation for Scientific Literature**

A multi-layered document intelligence system that retrieves, reasons over, and generates answers from ArXiv research papers, with built-in hallucination detection and confidence-calibrated citations.

Try it: [huggingface.co/spaces/Vinayak-0/skeptical-scholar](https://huggingface.co/spaces/Vinayak-0/skeptical-scholar)

---

## Motivation

Large language models hallucinate. Standard RAG pipelines reduce this by grounding answers in retrieved documents, but they treat retrieval as a black box. If the retrieved chunks are contradictory, irrelevant, or insufficiently supported, the LLM still generates a confident-sounding answer.

Skeptical Scholar addresses this by adding a reasoning layer that evaluates evidence quality _before_ generation, and a verification layer that checks factual consistency _after_ generation. The system refuses to answer when evidence is insufficient rather than guessing.

---

## What Makes This Different?

Most RAG systems retrieve and generate. Skeptical Scholar adds a critical layer between them: reasoning + verification.

| Feature     | Typical RAG     | Skeptical Scholar                                                  |
| ----------- | --------------- | ------------------------------------------------------------------ |
| Retrieval   | Single method   | **Hybrid** (BM25 + Dense + Cross-encoder)                          |
| Reasoning   | None            | **Entity extraction, contradiction detection, confidence scoring** |
| Generation  | Direct LLM call | **NLI-verified**, confidence-calibrated citations                  |
| Uncertainty | None            | **3-layer IDK system** knows when it doesn't know                  |

---

## Architecture

```
Query
  |
  v
+-----------------------------------------+
|  RETRIEVAL ENGINE                       |
|  BM25 ---+                              |
|          +--- Reciprocal Rank Fusion    |
|  Dense --+    (Hybrid Merge)            |
|          |                              |
|  Cross-Encoder Reranker                 |
|          |                              |
|  IDK Trigger 1 (low relevance)         |
+----------+------------------------------+
           |
           v
+-----------------------------------------+
|  REASONING LAYER                        |
|  Chunk Classifier (claim/evidence)      |
|  GLiNER Entity Extraction               |
|  NLI Contradiction Detection            |
|  Multi-Signal Confidence Scoring        |
|  IDK Trigger 2 (weak evidence)         |
+----------+------------------------------+
           |
           v
+-----------------------------------------+
|  GENERATION LAYER                       |
|  Evidence-Grounded Prompt Building      |
|  Groq LLM (Llama 3.1)                  |
|  DeBERTa NLI Answer Verification       |
|  IDK Trigger 3 (hallucination)         |
+----------+------------------------------+
           |
           v
   Verified Answer with Citations
   + Confidence Score + NLI Report
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/vinayakpareek-0/skeptical-scholar.git
cd skeptical-scholar
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up API Key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com)

### 3. Run the Ingestion Pipeline

```bash
python -m src.ingestion.run_pipeline
```

This fetches papers from ArXiv + Semantic Scholar, parses PDFs, chunks text, and stores everything in SQLite.

### 4. Build Dense Index

```bash
python -m src.retrieval.dense_retriever
```

### 5. Ask a Question

```bash
python -m src.generation.run_generation "attention mechanism"
```

**Sample output:**

```
Status: answered
Confidence: 0.321
NLI: supported=0.8, contradicted=0.0
Citations: 5

Answer:
Based on the provided evidence, here's what I can infer about the attention mechanism:
1. The attention mechanism plays a dominant role in sequence generation models,
   particularly in tasks such as machine translation and abstractive text
   summarization [1].
2. It can be viewed as a mechanism for reallocating resources according to
   importance or relevance [3].
3. The attention mechanism has been used in various visual tasks, where it can
   be seen as a lightweight yet effective mechanism [2].
4. There are different types of attention mechanisms, including global attention
   mechanisms based on supervised and unsupervised learning [5].
```

### 6. Launch the Chat UI

```bash
python app.py
```

Opens a Gradio chat interface at `http://localhost:7860`. Ask multiple questions in a single session.

---

## Improving Performance

The system ships with ~100 papers. The more domain-relevant papers you ingest, the better the answers.

**Add more papers** by editing `config.yaml`:

```yaml
arxiv:
  queries:
    - "your new topic here"
    - "another topic"
  max_results: 20 # increase from 10

semantic_scholar:
  max_results: 40 # increase from 20
  min_citations: 5 # lower to include more papers
```

Then re-run ingestion and rebuild the index:

```bash
python -m src.ingestion.run_pipeline
python -m src.retrieval.dense_retriever
```

**Tune retrieval sensitivity** -- if the system triggers IDK too aggressively on valid queries, lower the threshold. If it answers questions it should refuse, raise it:

```yaml
retrieval:
  idk_threshold: 0.0 # lower = more permissive, higher = stricter
```

**Tune generation** -- adjust the LLM parameters:

```yaml
generation:
  model: "llama-3.1-8b-instant"
  temperature: 0.3 # lower = more deterministic
  max_tokens: 512 # increase for longer answers
```

**Tune confidence scoring** -- adjust how much each signal contributes:

```yaml
confidence:
  weights:
    retrieval: 0.3
    evidence_ratio: 0.3
    entity_overlap: 0.2
    contradiction_penalty: 0.2
```

---

## Three-Layer IDK System

The system knows when it doesn't know. This is a critical feature for trustworthy AI.

| Layer                  | Trigger                                            | What It Catches             |
| ---------------------- | -------------------------------------------------- | --------------------------- |
| **IDK 1** (Retrieval)  | Low rerank score                                   | Out-of-domain queries       |
| **IDK 2** (Reasoning)  | Low confidence, high contradictions, no evidence   | Weak or conflicting sources |
| **IDK 3** (Generation) | NLI contradiction, hedging language, short answers | LLM hallucinations          |

---

## Project Structure

```
skeptical-scholar/
├── app.py                       # Gradio chat UI
├── config.yaml                  # All model names, paths, thresholds
├── requirements.txt
├── .env                         # GROQ_API_KEY (not committed)
│
├── src/
│   ├── ingestion/               # Phase 1: Data Pipeline
│   │   ├── arxiv_fetcher.py     # ArXiv paper fetching with checkpoints
│   │   ├── semantic_scholar_fetcher.py
│   │   ├── pdf_parser.py        # PDF to sections with PyMuPDF
│   │   ├── chunker.py           # Section-aware chunking
│   │   ├── database.py          # SQLite document store
│   │   ├── citation_parser.py   # Citation graph builder
│   │   └── run_pipeline.py      # End-to-end ingestion
│   │
│   ├── retrieval/               # Phase 2: Retrieval Engine
│   │   ├── bm25_retriever.py    # Sparse retrieval
│   │   ├── dense_retriever.py   # BGE-large + FAISS
│   │   ├── hybrid_retriever.py  # Reciprocal Rank Fusion
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   ├── idk_trigger.py       # IDK Layer 1
│   │   └── run_rag.py           # Retrieval pipeline
│   │
│   ├── reasoning/               # Phase 3: Reasoning Layer
│   │   ├── chunk_classify.py    # Heuristic + zero-shot classification
│   │   ├── entity_extract.py    # GLiNER entity extraction
│   │   ├── contradiction_detect.py
│   │   ├── confidence_score.py  # Multi-signal confidence fusion
│   │   ├── idk_trigger_2.py     # IDK Layer 2
│   │   └── run_reasoning.py     # Reasoning pipeline
│   │
│   ├── generation/              # Phase 4: Generation Layer
│   │   ├── llm_client.py        # Groq API client
│   │   ├── prompts.py           # Evidence-grounded prompt templates
│   │   ├── nli_verifier.py      # DeBERTa answer verification
│   │   ├── idk_trigger3.py      # IDK Layer 3
│   │   └── run_generation.py    # Full generation pipeline
│   │
│   └── evaluation/              # Phase 5: Evaluation
│       ├── retrieval_eval.py    # Retrieval ablation study
│       ├── generation_eval.py   # End-to-end generation eval
│       ├── evaluation.json      # 20-query test set
│       └── README.md            # Detailed evaluation report
│
└── data/
    ├── db/arxiv.db              # SQLite document store
    ├── processed/               # FAISS index + chunk mappings
    └── metadata/                # Fetch checkpoints
```

---

## Tech Stack

| Component                   | Technology                                       |
| --------------------------- | ------------------------------------------------ |
| **Data Ingestion**          | ArXiv API, Semantic Scholar API, PyMuPDF, SQLite |
| **Sparse Retrieval**        | rank-bm25                                        |
| **Dense Retrieval**         | sentence-transformers (BGE-large-en-v1.5), FAISS |
| **Reranking**               | CrossEncoder (ms-marco-MiniLM)                   |
| **Entity Extraction**       | GLiNER (zero-shot NER)                           |
| **Contradiction Detection** | DeBERTa NLI (cross-encoder)                      |
| **Generation**              | Groq API (Llama 3.1 8B)                          |
| **Answer Verification**     | DeBERTa NLI entailment checking                  |
| **UI**                      | Gradio                                           |

---

## Evaluation Results

Tested on 20 queries: 10 in-domain, 5 out-of-domain, 5 adversarial. Full report with per-query breakdowns at [src/evaluation/README.md](src/evaluation/README.md).

### Retrieval (IDK Layer 1)

| Category      | Queries | Avg Rerank Score | IDK Accuracy        |
| ------------- | ------- | ---------------- | ------------------- |
| In-domain     | 10      | 5.030            | 0 false triggers    |
| Out-of-domain | 5       | -7.500           | 5/5 rejected (100%) |
| Adversarial   | 5       | 0.528            | 2/5 rejected at L1  |

The cross-encoder reranker is the key discriminator. BM25 scores are unreliable for OOD detection (out-of-domain queries still score 20-35 on BM25).

### End-to-End Generation (All 3 IDK Layers)

| Category      | Passed | Total  | Accuracy |
| ------------- | ------ | ------ | -------- |
| In-domain     | 8      | 10     | 80%      |
| Out-of-domain | 5      | 5      | 100%     |
| Adversarial   | 5      | 5      | 100%     |
| **Overall**   | **18** | **20** | **90%**  |

The two in-domain failures are data coverage gaps, not pipeline bugs. Both would improve with a larger corpus.

---

## Data Description

The system ingests papers from ArXiv and Semantic Scholar into a local SQLite database. After processing:

- **Papers table** - paper metadata: arxiv_id, title, authors, abstract, publication date, PDF URL
- **Chunks table** - text segments with fields: `chunk_id`, `paper_id`, `section` (e.g. Introduction, Methods), `text`, `word_count`
- **Dense index** - FAISS flat inner-product index over BGE-large-en-v1.5 embeddings (1024-dim), with a `chunk_ids.npy` mapping vector positions to chunk IDs
- **Citation graph** - NetworkX directed graph stored as JSON, mapping paper titles to cited titles

Chunks are created with section-aware splitting: max 500 words, 50-word overlap, minimum 50 words. Each chunk retains its source paper and section for citation tracing.

---

## Configuration

All model names, thresholds, and paths are centralized in `config.yaml`. Zero hardcoded values in source code.

| Parameter                 | Default                  | Description                                   |
| ------------------------- | ------------------------ | --------------------------------------------- |
| `dense.model_name`        | `BAAI/bge-large-en-v1.5` | Embedding model for dense retrieval           |
| `retrieval.idk_threshold` | `0.0`                    | Rerank score below which IDK Layer 1 triggers |
| `chunking.max_length`     | `500`                    | Max words per chunk                           |
| `chunking.overlap`        | `50`                     | Word overlap between consecutive chunks       |
| `generation.model`        | `llama-3.1-8b-instant`   | Groq LLM model for answer generation          |
| `generation.temperature`  | `0.3`                    | LLM sampling temperature                      |
| `nli.model_name`          | `nli-deberta-v3-base`    | NLI model for contradiction and verification  |
| `entity.model_name`       | `gliner_medium-v2.1`     | GLiNER model for entity extraction            |

---

## Roadmap

- [x] Phase 1: Data Pipeline (ArXiv + Semantic Scholar)
- [x] Phase 2: Retrieval Engine (Hybrid + Reranker)
- [x] Phase 3: Reasoning Layer (Entities + Contradictions + Confidence)
- [x] Phase 4: Generation Layer (Groq + NLI Verification)
- [x] Phase 5: Evaluation (90% overall, 100% OOD rejection)
- [x] Phase 6: Gradio UI + HuggingFace Spaces Deployment

---

## License

MIT
