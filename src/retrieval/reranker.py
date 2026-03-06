from sentence_transformers import CrossEncoder
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config , PROJECT_ROOT
from ingestion.database import init_db, get_all_chunks as load_chunks
from retrieval.hybrid_retriever import run_hybrid_retrieval


def load_reranker(model_name:str):
    model = CrossEncoder(model_name)
    return model 

def rerank(reranker  , query , candidates:list[dict] , top_k =5):
    pairs = [[query , candidate["text"]] for candidate in candidates]
    scores = reranker.predict(pairs , convert_to_numpy=True)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]

if __name__ =="__main__":
    config = load_config()
    reranker = load_reranker(config["retrieval"]["reranker_name"])
    query = "What is the role of attention mechanism in transformers?"
    candidates = run_hybrid_retrieval(query , top_k =20)
    results = rerank(reranker , query , candidates , top_k=5)
    for result in results:
        print(result)
