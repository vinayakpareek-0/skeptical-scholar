import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config


def load_nli_model():
    from sentence_transformers import CrossEncoder

    config = load_config()
    return CrossEncoder(config["nli"]["model_name"])


def detect_contradictions(nli, chunks: list[dict], threshold=None) -> list[dict]:
    if threshold is None:
        config = load_config()
        threshold = config["nli"]["contradiction_threshold"]
    
    contradictions = []
    pairs = []
    chunk_pairs = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            pairs.append([chunks[i]["text"][:512], chunks[j]["text"][:512]])
            chunk_pairs.append((chunks[i], chunks[j]))

    if not pairs:
        return contradictions

    scores = nli.predict(pairs)
    for score, (chunk1, chunk2) in zip(scores, chunk_pairs):
        contrd_score = float(score[0])
        if contrd_score > threshold:
            contradictions.append({
                "chunk1": chunk1,
                "chunk2": chunk2,
                "score": contrd_score,
                "relationship": "contradiction"
            })
    return contradictions


if __name__ == "__main__":
    from retrieval.run_rag import run_rag

    results = run_rag("retrieval augmented generation")
    if not results:
        print("No results to detect contradictions")
        sys.exit()
    nli = load_nli_model()
    contradictions = detect_contradictions(nli, results)
    print(f"Found {len(contradictions)} contradictions")
    for c in contradictions:
        print(f"  {c['chunk1']['text'][:200]} vs {c['chunk2']['text'][:200]}: {c['score']:.3f}")
