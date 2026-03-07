"""
    RAG pipeline: Query → Hybrid Retrieval → Rerank → IDK Check → Answer
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from retrieval.hybrid_retriever import run_hybrid_retrieval
from retrieval.reranker import load_reranker, rerank
from retrieval.idk_trigger import check_retrieval_confidence


def run_rag(query):
    config = load_config()
    reranker = load_reranker(config["retrieval"]["reranker_name"])
    threshold = config["retrieval"]["idk_threshold"]

    candidates = run_hybrid_retrieval(query, top_k=20)
    results = rerank(reranker, query, candidates, top_k=5)
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
