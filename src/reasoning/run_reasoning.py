"""
Reasoning pipeline: Retrieval → Classify → Entities → Contradictions → Confidence → IDK
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrieval.run_rag import run_rag
from reasoning.chunk_classify import classify_chunks
from reasoning.entity_extract import load_extractor, extract_from_chunks
from reasoning.contradiction_detect import load_nli_model, detect_contradictions
from reasoning.confidence_score import compute_confidence
from reasoning.idk_trigger_2 import check_reasoning_confidence


def run_reasoning(query):
    # retrieve + rerank + idk layer 1
    chunks = run_rag(query)
    if not chunks:
        return {"status": "idk", "reason": "IDK Layer 1: No relevant evidence found", "chunks": []}

    # classify chunks
    chunks = classify_chunks(chunks, method="heuristic")

    # extract entities
    extractor = load_extractor()
    chunks = extract_from_chunks(extractor, chunks)

    #  detect contradictions
    nli = load_nli_model()
    contradictions = detect_contradictions(nli, chunks)

    # compute confidence
    confidence = compute_confidence(chunks, contradictions)

    # idk layer 2
    chunk_types = [c["chunk_type"] for c in chunks]
    idk2 = check_reasoning_confidence(confidence, contradictions, chunk_types)

    if idk2["triggered"]:
        return {"status": "idk", "reason": f"IDK Layer 2: {idk2['reason']}", "chunks": chunks}

    return {
        "status": "ready",
        "chunks": chunks,
        "entities": [e for c in chunks for e in c.get("entities", [])],
        "contradictions": contradictions,
        "confidence": confidence,
    }


if __name__ == "__main__":
    query = input("Query: ") if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    result = run_reasoning(query)

    print(f"\nStatus: {result['status']}")
    if result["status"] == "idk":
        print(f"Reason: {result['reason']}")
    else:
        print(f"Confidence: {result['confidence']['score']}")
        print(f"Breakdown: {result['confidence']['breakdown']}")
        print(f"Entities: {len(result['entities'])}")
        print(f"Contradictions: {len(result['contradictions'])}")
        for i, c in enumerate(result["chunks"], 1):
            print(f"\n[{i}] [{c['chunk_type']}] {c['section']} | {c['text'][:150]}")
