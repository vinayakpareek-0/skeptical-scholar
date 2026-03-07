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
    texts = []
    for chunk in chunks:
        texts.append(chunk["text"])
    
    embeddings = model.encode(texts , batch_size =32 , show_progress_bar =True , normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]

    index=faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index , model , chunks

def search_dense(index , model , query , chunks , top_k=10):
    query = "Represent this sentence for searching relevant passages: " + query
    query_embedding= model.encode([query], normalize_embeddings=True).astype('float32')
    scores , indexes  =index.search(query_embedding , top_k)
    results=[]
    for i,idx in enumerate(indexes[0]):
        results.append({
            "chunk_id":chunks[idx]["chunk_id"],
            "paper_id":chunks[idx]["paper_id"],
            "section":chunks[idx]["section"],
            "text":chunks[idx]["text"],
            "word_count":chunks[idx]["word_count"],
            "score":float(scores[0][i])
        })
    return results

def save_index(index , path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True) 
    faiss.write_index(index , str(path))
    print(f"Index saved to {path}")

def load_index(path):
    return faiss.read_index(str(path))

def load_dense_model(model_name:str):
    model = SentenceTransformer(model_name)
    return model

if __name__== "__main__":
    config = load_config()
    conn = init_db(PROJECT_ROOT / config["database"]["path"])
    chunks = load_chunks(conn)
    index , model , chunks = build_dense_index(chunks , config["dense"]["model_name"])
    save_index(index , PROJECT_ROOT / config["dense"]["index_path"])
    query = "What is the role of attention mechanism in transformers?"
    results = search_dense(index , model , query , chunks , top_k=5)
    for result in results: 
        print(result["text"])
        print(result["score"])
        print("\t")

