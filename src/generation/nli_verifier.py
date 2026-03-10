from sentence_transformers import CrossEncoder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config


def load_nli():
    config = load_config()
    return CrossEncoder(config["nli"]["model_name"])

def verify_answer(nli, answer, chunks):
    config = load_config()
    threshold = config["nli"]["reliability_threshold"]
    results = {"supported": 0, "contradicted": 0, "neutral": 0}
    
    for chunk in chunks:
        scores = nli.predict([(chunk["text"][:512], answer)])
        label = ["contradicted", "neutral", "supported"][scores[0].argmax()]
        results[label] += 1
    
    total = max(sum(results.values()), 1)
    return {
        "supported": round(results["supported"] / total, 2),
        "contradicted": round(results["contradicted"] / total, 2),
        "neutral": round(results["neutral"] / total, 2),
        "is_reliable": results["contradicted"] / total < threshold
    }

if __name__ == "__main__":
    nli = load_nli()
    print(verify_answer(nli, "How does retrieval augmented generation reduce hallucination?", [{"text": "Retrieval augmented generation summarizes the query and web search to find information for llm"}]))