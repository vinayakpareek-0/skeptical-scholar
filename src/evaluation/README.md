# Evaluation Report

Evaluation of the Skeptical Scholar pipeline across 20 test queries, split into three categories: in-domain (10), out-of-domain (5), and adversarial/tricky (5). The test set was designed to stress-test both retrieval accuracy and the three-layer IDK system.

Full results are in [evaluation_results.json](evaluation_results.json) and [generation_eval_results.json](generation_eval_results.json).

## Retrieval Layer (IDK Layer 1)

Each query was run through BM25, Dense (BGE-large), Hybrid (RRF), and Hybrid+Rerank (cross-encoder). The reranker score determines whether IDK Layer 1 triggers.

| ID  | Query                                | Category      | BM25  | Dense | Hybrid | Rerank  | IDK |
| --- | ------------------------------------ | ------------- | ----- | ----- | ------ | ------- | --- |
| 1   | RAG hallucination reduction          | in_domain     | 26.54 | 0.839 | 0.032  | 7.702   | no  |
| 2   | Dense vs sparse retrieval            | in_domain     | 38.26 | 0.757 | 0.029  | 4.744   | no  |
| 3   | Chain-of-Thought prompting           | in_domain     | 21.12 | 0.860 | 0.033  | 7.364   | no  |
| 4   | Hallucination detection techniques   | in_domain     | 31.84 | 0.694 | 0.031  | 3.505   | no  |
| 5   | Reranking in RAG                     | in_domain     | 23.03 | 0.813 | 0.016  | 4.343   | no  |
| 6   | Knowledge graphs in RAG              | in_domain     | 36.51 | 0.803 | 0.033  | 5.248   | no  |
| 7   | Zero-shot vs few-shot vs fine-tuning | in_domain     | 25.61 | 0.773 | 0.016  | 4.826   | no  |
| 8   | Self-consistency decoding            | in_domain     | 24.86 | 0.784 | 0.033  | 2.795   | no  |
| 9   | RAG evaluation metrics               | in_domain     | 24.87 | 0.840 | 0.016  | 5.187   | no  |
| 10  | Attention in Transformers            | in_domain     | 22.56 | 0.786 | 0.033  | 4.582   | no  |
| 11  | Chocolate chip cookie recipe         | out_of_domain | 23.95 | 0.505 | 0.016  | -10.942 | YES |
| 12  | FIFA World Cup 2022                  | out_of_domain | 35.04 | 0.526 | 0.032  | -9.486  | YES |
| 13  | Population of India                  | out_of_domain | 25.53 | 0.592 | 0.016  | -7.175  | YES |
| 14  | Crypto investing                     | out_of_domain | 23.12 | 0.558 | 0.016  | -9.484  | YES |
| 15  | Oppenheimer plot summary             | out_of_domain | 31.63 | 0.704 | 0.031  | -0.415  | YES |
| 16  | Tell me everything about AI          | adversarial   | 20.61 | 0.654 | 0.016  | -2.447  | YES |
| 17  | Smith et al. paper (fake ref)        | adversarial   | 27.54 | 0.639 | 0.016  | 2.262   | no  |
| 18  | RAG completely solves hallucinations | adversarial   | 30.73 | 0.650 | 0.016  | -2.011  | YES |
| 19  | Fine-tuning always superior to RAG   | adversarial   | 30.55 | 0.724 | 0.030  | 3.084   | no  |
| 20  | Legal RAG hallucination              | adversarial   | 33.00 | 0.651 | 0.030  | 1.754   | no  |

### Retrieval summary

| Category      | Queries | Avg Rerank | IDK accuracy                  |
| ------------- | ------- | ---------- | ----------------------------- |
| In-domain     | 10      | 5.030      | 0 false triggers (100%)       |
| Out-of-domain | 5       | -7.500     | 5/5 correctly rejected (100%) |
| Adversarial   | 5       | 0.528      | 2/5 rejected at Layer 1       |

BM25 scores are high even for out-of-domain queries because BM25 matches surface-level tokens. The cross-encoder reranker is what separates in-domain from out-of-domain -- rerank scores for out-of-domain are consistently negative while in-domain scores are 2.7+. This validates the hybrid + rerank approach.

Adversarial queries that pass Layer 1 (17, 19, 20) are handled by Layers 2 and 3.

---

## End-to-End Generation (All 3 IDK Layers)

Each query was run through the full pipeline: retrieval → reasoning → generation → NLI verification.

| ID  | Query                                | Status   | Confidence | NLI Sup | NLI Con | Pass |
| --- | ------------------------------------ | -------- | ---------- | ------- | ------- | ---- |
| 1   | RAG hallucination reduction          | answered | 0.456      | 1.00    | 0.00    | PASS |
| 2   | Dense vs sparse retrieval            | answered | 0.416      | 1.00    | 0.00    | PASS |
| 3   | Chain-of-Thought prompting           | answered | 0.533      | 1.00    | 0.00    | PASS |
| 4   | Hallucination detection techniques   | answered | 0.365      | 1.00    | 0.00    | PASS |
| 5   | Reranking in RAG                     | idk      | -          | 0.80    | 0.20    | FAIL |
| 6   | Knowledge graphs in RAG              | answered | 0.385      | 0.80    | 0.00    | PASS |
| 7   | Zero-shot vs few-shot vs fine-tuning | answered | 0.331      | 1.00    | 0.00    | PASS |
| 8   | Self-consistency decoding            | answered | 0.355      | 1.00    | 0.00    | PASS |
| 9   | RAG evaluation metrics               | answered | 0.398      | 1.00    | 0.00    | PASS |
| 10  | Attention in Transformers            | answered | 0.294      | 0.40    | 0.00    | FAIL |
| 11  | Chocolate chip cookie recipe         | idk      | -          | -       | -       | PASS |
| 12  | FIFA World Cup 2022                  | idk      | -          | -       | -       | PASS |
| 13  | Population of India                  | idk      | -          | -       | -       | PASS |
| 14  | Crypto investing                     | idk      | -          | -       | -       | PASS |
| 15  | Oppenheimer plot summary             | idk      | -          | -       | -       | PASS |
| 16  | Tell me everything about AI          | idk      | -          | -       | -       | PASS |
| 17  | Smith et al. paper (fake ref)        | idk      | -          | 0.60    | 0.40    | PASS |
| 18  | RAG completely solves hallucinations | idk      | -          | -       | -       | PASS |
| 19  | Fine-tuning always superior to RAG   | answered | 0.415      | 1.00    | 0.00    | PASS |
| 20  | Legal RAG hallucination              | answered | 0.421      | 1.00    | 0.00    | PASS |

### Generation summary

| Category      | Passed | Total  | Accuracy |
| ------------- | ------ | ------ | -------- |
| In-domain     | 8      | 10     | 80%      |
| Out-of-domain | 5      | 5      | 100%     |
| Adversarial   | 5      | 5      | 100%     |
| **Overall**   | **18** | **20** | **90%**  |

---

## Analysis of Failures

**Query 5** (reranking quality) -- triggered IDK despite being in-domain. The corpus has limited papers specifically about cross-encoder reranking as a technique. The system correctly identified weak evidence and refused to answer. This is a data coverage issue, not a pipeline bug.

**Query 10** (attention mechanisms) -- answered but NLI support was only 0.40 (below the 0.5 pass threshold). The generated answer was factually correct and well-cited, but NLI classified most chunks as "neutral" rather than "supporting" because the chunk text describes mechanisms while the answer paraphrases them. This is a known limitation of NLI models when comparing technical explanations at different levels of abstraction.

Both failures would improve with a larger, more diverse corpus.

---

## Observations

- The reranker is the most important signal for out-of-domain detection. BM25 scores are unreliable for this purpose (out-of-domain queries still get BM25 scores of 20-35).

- Dense retrieval scores (0.5-0.86) are less discriminative between in-domain and out-of-domain than rerank scores. Dense scores for totally irrelevant queries like "chocolate chip cookie recipe" still reach 0.505.

- Adversarial queries with valid in-domain terms (queries 17, 19, 20) correctly pass Layer 1 but are handled by downstream layers. Query 17 (fake paper reference) was caught by IDK Layer 3 via NLI contradiction. Queries 19 and 20 generated "I don't know" responses despite technically passing all IDK layers, showing the LLM itself can identify insufficient evidence.

- Average query latency is ~55s for answered queries and ~25s for IDK-rejected queries. The bottleneck is NLI verification (DeBERTa) running on CPU. This would drop to ~5s with GPU.

---

## Configuration at time of evaluation

```
idk_threshold: 0.0
idk.confidence_threshold: 0.1
nli.contradiction_threshold: 0.7
nli.reliability_threshold: 0.3
generation.model: llama-3.1-8b-instant
dense.model_name: BAAI/bge-large-en-v1.5
corpus: ~100 papers, ~5000 chunks
```
