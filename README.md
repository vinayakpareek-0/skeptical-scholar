# Skeptical Scholar:

`Robust Neuro-Symbolic RAG with Uncertainty Estimation for Scientific Literature`

A multi-layered document intelligence system that combines hybrid retrieval (BM25 + dense + cross-encoder), dynamic knowledge graph construction with multi-hop symbolic reasoning, and calibrated generation with NLI-based entailment verification. Implements selective prediction through three-layer epistemic uncertainty quantification.

`Two core differentiators:`

- Neuro-Symbolic Reasoning
- Verified Generation with Selective Prediction

`Tech Stack`

- Data: arxiv, PyMuPDF, SQLite, NetworkX
- Retrieval: rank-bm25, sentence-transformers (BGE-large, CrossEncoder rerank), faiss-cpu,
- Reasoning: gliner, networkx, scikit-learn
- Generation: ollama (Mistral/Phi-3), transformers (DeBERTa NLI)
- Evaluation: rouge-score, bert-score, matplotlib
- Serving: Gradio, FastAPI (optional), Docker

> #### `Work in progress..`
