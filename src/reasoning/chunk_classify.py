import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrieval.run_rag import run_rag
from transformers import pipeline


def classify_chunk_heuristic(chunk):
    text = chunk["text"].lower()
    section = chunk.get("section", "").lower()
    
    # key words
    method_kw = ["we propose", "our method", "we introduce", "our approach", "our model", 
                 "we design", "we present", "architecture", "framework", "pipeline"]
    result_kw = ["accuracy", "outperforms", "achieves", "table", "figure", "f1", 
                 "improvement", "baseline", "benchmark", "ablation", "%"]
    evidence_kw = ["shows that", "demonstrates", "found that", "indicates", "suggests", 
                   "has been shown", "empirical", "observed that", "proven"]
    claim_kw = ["we argue", "hypothesize", "we believe", "conjecture", "we expect",
                "we claim", "in this paper", "we show that", "contribution"]
    
    # count
    scores = {
        "method": sum(1 for kw in method_kw if kw in text),
        "result": sum(1 for kw in result_kw if kw in text),
        "evidence": sum(1 for kw in evidence_kw if kw in text),
        "claim": sum(1 for kw in claim_kw if kw in text),
    }
    
    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best
    
    if "experiment" in section or "result" in section:
        return "result"
    if "method" in section or "approach" in section:
        return "method"
    
    return "background"

def load_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_chunk_zeroshot(classifier, chunk):
    text = chunk["text"][:512]
    labels = ["method", "result", "evidence", "claim", "background"]
    result = classifier(text, candidate_labels=labels)
    return result["labels"][0]


def classify_chunks(chunks, method="heuristic", classifier=None):
    for chunk in chunks:
        if method == "zeroshot" and classifier:
            chunk["chunk_type"] = classify_chunk_zeroshot(classifier, chunk)
        else:
            chunk["chunk_type"] = classify_chunk_heuristic(chunk)
    return chunks


if __name__ == "__main__":
    results = run_rag("vision transformers and how are they different from transformers?")
    if not results:
        print("No results to classify")
        sys.exit()
    chunks_h = classify_chunks(results, method="heuristic")
    for c in chunks_h:
        print(f"  [{c['chunk_type']}] {c['section']} | {c['text'][:80]}")