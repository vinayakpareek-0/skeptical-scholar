import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from retrieval.run_rag import run_rag
from reasoning.chunk_classify import classify_chunks
from reasoning.entity_extract import load_extractor, extract_from_chunks
from reasoning.contradiction_detect import load_nli_model , detect_contradictions
from reasoning.confidence_score import compute_confidence

def check_reasoning_confidence(confidence, contradictions, chunk_types):
    # Low confidence score
    if confidence["score"] < 0.4:
        return {
            "triggered": True,
            "reason": f"Low reasoning confidence ({confidence['score']:.3f})",
            "suggestion": "Evidence is weak or inconsistent"
        }
    
    # High contradiction ratio
    if len(contradictions) > len(chunk_types) / 2:
        return {
            "triggered": True,
            "reason": f"High contradiction ratio ({len(contradictions)} contradictions)",
            "suggestion": "Sources disagree significantly"
        }
    
    # No evidence found (all background)
    evidence_count = sum(1 for t in chunk_types if t in ["evidence", "result", "method"])
    if evidence_count == 0:
        return {
            "triggered": True,
            "reason": "No supporting evidence found",
            "suggestion": "Retrieved chunks lack substantive claims"
        }
    
    return {
        "triggered": False,
        "reason": "Sufficient reasoning confidence",
        "score": confidence["score"]
    }

if __name__=="__main__":
    chunks = run_rag("Self attention architecture")
    if not chunks:
        print("No results")
        sys.exit()
    chunks = classify_chunks(chunks, method="heuristic")
    model = load_extractor()
    chunks = extract_from_chunks(model , chunks)
    nli = load_nli_model()
    contd = detect_contradictions(nli , chunks)
    confidence= compute_confidence(chunks, contd)
    chunk_types = [c["chunk_type"] for c in chunks]
    reasoning = check_reasoning_confidence(confidence, contd, chunk_types)
    print(f"IDK Trigger 2: {reasoning}")



