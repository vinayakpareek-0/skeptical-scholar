import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config, PROJECT_ROOT
from ingestion.database import init_db, get_all_chunks as load_chunks
from retrieval.bm25_retriever import build_bm25_index, search_bm25
from retrieval.dense_retriever import load_index as load_dense_index, load_dense_model, search_dense
from retrieval.hybrid_retriever import search_hybrid
from retrieval.reranker import load_reranker, rerank
from retrieval.idk_trigger import check_retrieval_confidence


test_queries = [
    # In-domain
    {"query": "attention mechanism in transformers", "domain": "in"},
    {"query": "vision transformer adversarial robustness", "domain": "in"},
    {"query": "self-attention computation complexity", "domain": "in"},
    {"query": "how positional encoding works in transformers", "domain": "in"},
    {"query": "why transformers use multi-head attention", "domain": "in"},
    {"query": "quadratic complexity problem in transformers", "domain": "in"},
    {"query": "vision transformer patch embedding process", "domain": "in"},
    {"query": "role of layer normalization in transformers", "domain": "in"},
    {"query": "transformer architecture for image classification", "domain": "in"},
    {"query": "how transformer models handle long dependencies", "domain": "in"},

    # Borderline
    {"query": "bert vs transformer architecture differences", "domain": "border"},
    {"query": "large language model training pipeline", "domain": "border"},
    {"query": "comparison of cnn and vision transformers", "domain": "border"},
    {"query": "how retrieval augmented generation works", "domain": "border"},
    {"query": "fine tuning transformer models for classification", "domain": "border"},

    # Out-of-domain
    {"query": "best pizza recipe", "domain": "out"},
    {"query": "how to train a dog", "domain": "out"},
    {"query": "stock market prediction 2025", "domain": "out"},
    {"query": "how to grow tomatoes at home", "domain": "out"},
    {"query": "best travel destinations in europe", "domain": "out"},
    {"query": "symptoms of vitamin d deficiency", "domain": "out"},
    {"query": "how to build a gaming pc", "domain": "out"},
    {"query": "history of the roman empire", "domain": "out"},
    {"query": "how to cook pasta alfredo", "domain": "out"},
    {"query": "tips for public speaking confidence", "domain": "out"},
]


def run_evaluation():
    """
    More of "Wiring" than evaluation, can be called lite phase-2 wrapper
    """
    config = load_config()
    conn = init_db(PROJECT_ROOT / config["database"]["path"])
    chunks = load_chunks(conn)

    bm25_index, chunks = build_bm25_index(chunks)
    dense_index = load_dense_index(PROJECT_ROOT / config["dense"]["index_path"])
    dense_model = load_dense_model(config["dense"]["model_name"])
    reranker = load_reranker(config["retrieval"]["reranker_name"])

    threshold = config["retrieval"]["idk_threshold"]

    # Collect results per method
    results_table = []

    for item in test_queries:
        query = item["query"]
        domain = item["domain"]

        # BM25
        bm25_results = search_bm25(bm25_index, query, chunks, top_k=5)
        bm25_top = bm25_results[0]["score"] if bm25_results else 0

        # Dense
        dense_results = search_dense(dense_index, dense_model, query, chunks, top_k=5)
        dense_top = dense_results[0]["score"] if dense_results else 0

        # Hybrid
        hybrid_results = search_hybrid(query, bm25_index, dense_index, dense_model, chunks, top_k=20)
        hybrid_top = hybrid_results[0]["score"] if hybrid_results else 0

        # Hybrid + Rerank
        reranked = rerank(reranker, query, hybrid_results, top_k=5)
        rerank_top = reranked[0]["rerank_score"] if reranked else 0

        # IDK check
        idk = check_retrieval_confidence(reranked, threshold=threshold)

        results_table.append({
            "query": query,
            "domain": domain,
            "bm25_top": round(bm25_top, 3),
            "dense_top": round(dense_top, 3),
            "hybrid_top": round(hybrid_top, 5),
            "rerank_top": round(rerank_top, 3),
            "idk_triggered": idk["triggered"],
        })

    # Print results
    print(f"{'Query':<50} {'Domain':<8} {'BM25':>8} {'Dense':>8} {'Hybrid':>10} {'Rerank':>8} {'IDK':>6}")
    
    for r in results_table:
        idk_str = "YES" if r["idk_triggered"] else "no"
        print(f"{r['query']:<50} {r['domain']:<8} {r['bm25_top']:>8} {r['dense_top']:>8} {r['hybrid_top']:>10} {r['rerank_top']:>8} {idk_str:>6}")

    # Summary stats
    in_domain = [r for r in results_table if r["domain"] == "in"]
    border = [r for r in results_table if r["domain"] == "border"]
    out_domain = [r for r in results_table if r["domain"] == "out"]

    print("\nSUMMARY")

    if in_domain:
        avg_rerank_in = sum(r["rerank_top"] for r in in_domain) / len(in_domain)
        idk_false_triggers = sum(1 for r in in_domain if r["idk_triggered"])
        print(f"In-domain  ({len(in_domain)} queries): Avg rerank score = {avg_rerank_in:.3f}, False IDK triggers = {idk_false_triggers}")

    if border:
        avg_rerank_border = sum(r["rerank_top"] for r in border) / len(border)
        print(f"Borderline ({len(border)} queries): Avg rerank score = {avg_rerank_border:.3f}")

    if out_domain:
        avg_rerank_out = sum(r["rerank_top"] for r in out_domain) / len(out_domain)
        idk_correct = sum(1 for r in out_domain if r["idk_triggered"])
        idk_accuracy = (idk_correct / len(out_domain)) * 100
        print(f"Out-domain ({len(out_domain)} queries): Avg rerank score = {avg_rerank_out:.3f}, IDK accuracy = {idk_accuracy:.1f}%")


if __name__ == "__main__":
    run_evaluation()
