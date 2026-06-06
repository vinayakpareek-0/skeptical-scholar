"""
    RAG pipeline: Query → Hybrid Retrieval → Rerank → IDK Check → Answer
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime_cache import get_config, get_reranker
from retrieval.hybrid_retriever import run_hybrid_retrieval
from retrieval.reranker import rerank
from retrieval.idk_trigger import check_retrieval_confidence

# have to add more args for this later
def run_rag(query):
    config = get_config()
    retrieval_cfg = config["retrieval"]
    threshold = retrieval_cfg["idk_threshold"]
    candidate_top_k = retrieval_cfg.get("candidate_top_k", 20)
    rerank_top_k = retrieval_cfg.get("rerank_top_k", 5)

    candidates = run_hybrid_retrieval(query, top_k=candidate_top_k)
    if retrieval_cfg.get("enable_reranker", True):
        reranker = get_reranker()
        results = rerank(reranker, query, candidates, top_k=rerank_top_k)
    else:
        results = sorted(
            candidates,
            key=lambda item: (
                item.get("dense_score", -1.0),
                item.get("score", 0.0),
                item.get("bm25_score", 0.0),
            ),
            reverse=True,
        )[:rerank_top_k]
        for result in results:
            result["rerank_score"] = result.get("dense_score", result.get("score", 0.0))
        threshold = retrieval_cfg.get("dense_idk_threshold", threshold)

    idk = check_retrieval_confidence(results, threshold=threshold)

    if idk["triggered"]:
        print(f"[IDK] {idk['reason']}")
        return None
    return results


if __name__ == "__main__":
    query = input("Query: ") if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    results = run_rag(query)
    if results:
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['rerank_score']:.3f} | Paper: {r['paper_id']} | Section: {r['section']}")
            print(r["text"][:300])
