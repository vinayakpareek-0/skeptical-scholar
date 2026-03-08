from sentence_transformers import CrossEncoder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrieval.run_rag import run_rag

def load_nli_model():
    return CrossEncoder("cross-encoder/nli-deberta-v3-base")


def detect_contradictions(nli , chunks:list[dict], threshold=0.7)->list[dict]:
    contradictions=[]
    for i in range(len(chunks)):
        for j in range(i+1 , len(chunks)):
            pair = [chunks[i]["text"][:512], chunks[j]["text"][:512]]
            score= nli.predict([pair])

            contrd_score=float(score[0][0])
            relationship ="contradiction" if contrd_score>threshold else "neutral"
            if contrd_score>threshold:
                contradictions.append({
                    "chunk1":chunks[i],
                    "chunk2":chunks[j],
                    "score":contrd_score,
                    "relationship":relationship
                })
    return contradictions


if __name__ == "__main__":
    results = run_rag("retrieval augmented generation")
    if not results:
        print("No results to detect contradictions")
        sys.exit()
    nli =load_nli_model()
    contradictions = detect_contradictions(nli, results)
    print(f"Found {len(contradictions)} contradictions")
    for c in contradictions:
        print(f"  {c['chunk1']['text'][:200]} vs {c['chunk2']['text'][:200]}: {c['score']:.3f}")
