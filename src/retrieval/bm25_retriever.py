import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PROJECT_ROOT
from config import load_config
from ingestion.database import init_db, get_all_chunks as load_chunks
from rank_bm25 import BM25Okapi
import numpy as np

def build_bm25_index(chunks:list[dict])-> tuple[BM25Okapi , list[dict]]:
    tokenized_chunks = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, chunks

def search_bm25(index , query :str , chunks:list[dict] , top_k:int = 3)-> list[dict]:
    tokenized_query= query.lower().split()
    scores = index.get_scores(tokenized_query)
    top_k_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {**chunks[i], "score": float(scores[i])} 
        for i in top_k_indices
    ]
if __name__=="__main__":
    config =load_config()
    conn = init_db(PROJECT_ROOT / config["database"]["path"])
    chunks = load_chunks(conn)
    bm25, chunks = build_bm25_index(chunks)
    query = "What is the role of attention mechanism in transformers?"
    results = search_bm25(bm25, query, chunks, top_k=10)
    for result in results:
        print(result["text"])
        print("............................")
        print(result["score"])
        print("\t")
