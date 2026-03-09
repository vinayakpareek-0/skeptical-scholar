def build_prompt(query , chunks , confidence):
    evidence = ""
    for i, chunk in enumerate(chunks,1):
        evidence +=f"[{i} ({chunk['paper_id']}, {chunk['section']})]: {chunk['text'][:300]}\n\n"
    
    conf_label = "high" if confidence["score"]>0.7 else "moderate" if confidence["score"]>0.4 else "low"

    return f"""You are a scientific research assistant. Answer the question using ONLY the evidence provided below.
Cite sources as [1], [2], etc. If evidence is insufficient, say "I don't know".
Evidence:
{evidence}
Confidence: {confidence['score']:.2f} ({conf_label})
Question: {query}
Answer:
    """

def build_idk_prompt(query , reason):
    return f"""The retrieval system could not find sufficient evidence to answer this question.
Reason: {reason}
Question: {query}
Provide a brief explanation of why this question cannot be answered with the available corpus.
"""