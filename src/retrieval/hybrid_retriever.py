"""
Hybrid retrieval using Reciprocal Rank Fusion (RRF)
RRF_score(chunk) = 1/(k + rank_bm25) + 1/(k + rank_dense)
"""
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrieval.bm25_retriever import search_bm25
from retrieval.dense_retriever import search_dense
from config import load_config , PROJECT_ROOT
from ingestion.database import init_db, get_all_chunks as load_chunks
from retrieval.dense_retriever import load_index as load_dense_index , load_dense_model
from retrieval.bm25_retriever import build_bm25_index


def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
    scores = {}    
    
    for rank, result in enumerate(bm25_results, start=1):
        cid = result["chunk_id"]
        if cid not in scores:
            scores[cid] = {"data": result, "score": 0}
        scores[cid]["score"] += 1 / (k + rank)
    
    for rank, result in enumerate(dense_results, start=1):
        cid = result["chunk_id"]
        if cid not in scores:
            scores[cid] = {"data": result, "score": 0}
        scores[cid]["score"] += 1 / (k + rank)
    
    merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [{**item["data"], "score": item["score"]} for item in merged]

def search_hybrid(query , bm25_index , dense_index , model , chunks, top_k=20):
    bm25_results  = search_bm25(bm25_index  ,query , chunks , top_k =20)
    dense_results = search_dense(dense_index ,model , query , chunks , top_k=20 )
    return reciprocal_rank_fusion(bm25_results , dense_results , k=60)

def run_hybrid_retrieval(query , top_k=20): 
    """
    Single function runs entire retrieval logic
    Args: Query , top_k  (good for single call)

    # optim later ,might be slow for repeated calls , due to init_db again and again
    """
    config = load_config()
    conn = init_db(PROJECT_ROOT / config["database"]["path"])
    chunks = load_chunks(conn)
    bm25_index , chunks  = build_bm25_index(chunks)
    dense_index = load_dense_index(PROJECT_ROOT / config["dense"]["index_path"])
    model = load_dense_model(config["dense"]["model_name"])
    return search_hybrid(query , bm25_index , dense_index , model , chunks , top_k=top_k)

if __name__ == "__main__":
    query = "What is the role of attention mechanism in transformers?"
    results = run_hybrid_retrieval(query)
    for result in results[:5]:
        print(result,"\t")
