import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config


def build_prompt(query, chunks, confidence):
    config = load_config()
    gen = config["generation"]
    limit = gen.get("chunk_text_limit", 300)
    labels = gen.get("confidence_labels", {"high": 0.7, "moderate": 0.4})

    evidence = ""
    for i, chunk in enumerate(chunks, 1):
        evidence += f"[{i} ({chunk['paper_id']}, {chunk['section']})]: {chunk['text'][:limit]}\n\n"

    conf_label = "high" if confidence["score"] > labels["high"] else "moderate" if confidence["score"] > labels["moderate"] else "low"

    return f"""You are a scientific research assistant. Answer the question using ONLY the evidence provided below.
Cite sources as [1], [2], etc. If evidence is insufficient, say "I don't know".
Evidence:
{evidence}
Confidence: {confidence['score']:.2f} ({conf_label})
Question: {query}
Answer:
    """

def build_idk_prompt(query, reason):
    return f"""The retrieval system could not find sufficient evidence to answer this question.
Reason: {reason}
Question: {query}
Provide a brief explanation of why this question cannot be answered with the available corpus.
"""