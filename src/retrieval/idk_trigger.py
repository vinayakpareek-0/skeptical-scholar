import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "..")))
from config import load_config
from retrieval.reranker import load_reranker, rerank
from retrieval.hybrid_retriever import run_hybrid_retrieval

def check_retrieval_confidence(results , threshold=0.5):
    
    if not results:
        return {
            "triggered":True,
            "reason": "No results found",
            "suggestion": "Try a different query",
            "top_score":0.0
        }
    top_score = results[0]["rerank_score"]
    
    if top_score<threshold:
        return {
            "triggered": True,
            "reason": f"Low retrieval confidence ({top_score:.3f} < {threshold})",
            "suggestion": "Query may be outside the corpus domain",
            "top_score": top_score
        }
    
    return {
        "triggered": False,
        "reason": "Sufficient evidence found",
        "top_score": top_score
    }

if __name__ == "__main__":
    config = load_config()
    reranker = load_reranker(config["retrieval"]["reranker_name"])
    
    # In-domain query
    results = rerank(reranker, "attention mechanism in transformers", run_hybrid_retrieval("attention mechanism in transformers"))
    print("In domain:", check_retrieval_confidence(results))
    
    # Out-of-domain query
    results = rerank(reranker, "best pizza in new york", run_hybrid_retrieval("best pizza in new york"))
    print("Out of domain:", check_retrieval_confidence(results))