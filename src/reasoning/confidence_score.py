import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from retrieval.run_rag import run_rag
from reasoning.chunk_classify import classify_chunks
from reasoning.entity_extract import load_extractor, extract_from_chunks
from reasoning.contradiction_detect import load_nli_model , detect_contradictions

def compute_confidence(chunks, contradictions):
    scores = [c.get("rerank_score", 0) for c in chunks]
    retrieval = max(0, min(1, (max(scores) + 10) / 20))  # normalize
    
    types = [c.get("chunk_type", "background") for c in chunks]
    evidence_ratio = sum(1 for t in types if t in ["evidence", "result"]) / max(len(types), 1)
    
    all_entities = []
    for c in chunks:
        all_entities.extend([e["text"].lower() for e in c.get("entities", [])])
    unique = set(all_entities)
    shared = sum(1 for e in unique if all_entities.count(e) > 1) / max(len(unique), 1)
    
    total_pairs = max(len(chunks) * (len(chunks) - 1) / 2, 1)
    contra_penalty = len(contradictions) / total_pairs
    
    score = (0.3 * retrieval) + (0.3 * evidence_ratio) + (0.2 * shared) - (0.2 * contra_penalty)
    score = max(0.0, min(1.0, score))
    
    return {
        "score": round(score, 3),
        "breakdown": {
            "retrieval": round(retrieval, 3),
            "evidence_ratio": round(evidence_ratio, 3),
            "entity_overlap": round(shared, 3),
            "contradiction_penalty": round(contra_penalty, 3)
        }
    }

if __name__=="__main__":
    chunks =  run_rag("vision transformer robustness")
    if not chunks:
        print("No results")
        sys.exit()
    chunks = classify_chunks(chunks, method="heuristic")
    model = load_extractor()
    chunks = extract_from_chunks(model , chunks)
    nli = load_nli_model()
    contd = detect_contradictions(nli , chunks)
    confidence= compute_confidence(chunks, contd)
    print(f"Score: {confidence['score']}")
    print(f"Breakdown: {confidence['breakdown']}")