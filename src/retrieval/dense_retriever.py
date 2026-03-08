from sentence_transformers import SentenceTransformer 
import faiss
import sys 
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np 
from config import load_config , PROJECT_ROOT
from ingestion.database import init_db, get_all_chunks as load_chunks

def build_dense_index(chunks:list[dict] , model_name:str):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts , batch_size=32 , show_progress_bar=True , normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index , model , chunks

def search_dense(index , model , query , chunks , top_k=10):
    query = "Represent this sentence for searching relevant passages: " + query
    query_embedding = model.encode([query], normalize_embeddings=True).astype('float32')
    scores , indexes = index.search(query_embedding , top_k)
    
    chunk_map = {c["chunk_id"]: c for c in chunks}
    
    config = load_config()
    ids_path = PROJECT_ROOT / "data" / "processed" / "chunk_ids.npy"
    if ids_path.exists():
        saved_ids = np.load(str(ids_path), allow_pickle=True)
    else:
        saved_ids = [c["chunk_id"] for c in chunks]
    
    results = []
    for i, idx in enumerate(indexes[0]):
        if idx < 0 or idx >= len(saved_ids):
            continue
        cid = saved_ids[idx]
        if cid in chunk_map:
            c = chunk_map[cid]
            results.append({
                "chunk_id": c["chunk_id"],
                "paper_id": c["paper_id"],
                "section": c["section"],
                "text": c["text"],
                "word_count": c["word_count"],
                "score": float(scores[0][i])
            })
    return results

def save_index(index , path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True) 
    faiss.write_index(index , str(path))

def load_index(path):
    return faiss.read_index(str(path))

def load_dense_model(model_name:str):
    return SentenceTransformer(model_name)

if __name__ == "__main__":
    config = load_config()
    conn = init_db(PROJECT_ROOT / config["database"]["path"])
    chunks = load_chunks(conn)
    index = load_index(PROJECT_ROOT / config["dense"]["index_path"])
    model = load_dense_model(config["dense"]["model_name"])
    query = "What is the role of attention mechanism in transformers?"
    results = search_dense(index , model , query , chunks , top_k=5)
    for r in results: 
        print(f"[{r['score']:.3f}] {r['paper_id']} | {r['text'][:150]}")
