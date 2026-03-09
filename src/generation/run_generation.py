"""
Generation pipeline: Reasoning → Prompt → LLM → NLI Verify → IDK 3 → Final Answer
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reasoning.run_reasoning import run_reasoning
from generation.llm_client import load_llm, generate
from generation.prompts import build_prompt, build_idk_prompt
from generation.nli_verifier import load_nli, verify_answer
from generation.idk_trigger3 import check_generation_confidence


def run_generation(query):
    reasoning = run_reasoning(query)

    if reasoning["status"] == "idk":
        client = load_llm()
        idk_prompt = build_idk_prompt(query, reasoning["reason"])
        explanation = generate(client, idk_prompt)
        return {
            "status": "idk",
            "reason": reasoning["reason"],
            "explanation": explanation,
            "answer": None
        }

    client = load_llm()
    prompt = build_prompt(query, reasoning["chunks"], reasoning["confidence"])
    answer = generate(client, prompt)

    nli = load_nli()
    nli_result = verify_answer(nli, answer, reasoning["chunks"])

    idk3 = check_generation_confidence(answer, nli_result)
    if idk3["triggered"]:
        return {
            "status": "idk",
            "reason": f"IDK Layer 3: {idk3['reason']}",
            "answer": answer,
            "nli": nli_result
        }

    return {
        "status": "answered",
        "answer": answer,
        "confidence": reasoning["confidence"],
        "nli": nli_result,
        "citations": [{"paper_id": c["paper_id"], "section": c["section"]} for c in reasoning["chunks"]],
        "entities": reasoning.get("entities", []),
        "contradictions": reasoning.get("contradictions", [])
    }


if __name__ == "__main__":
    query = input("Query: ") if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    result = run_generation(query)

    print(f"Status: {result['status']}")
    if result["status"] == "answered":
        print(f"Confidence: {result['confidence']['score']}")
        print(f"NLI: supported={result['nli']['supported']}, contradicted={result['nli']['contradicted']}")
        print(f"Citations: {len(result['citations'])}")
        print(f"Answer:{result['answer']}")
    else:
        print(f"Reason: {result['reason']}")
        if result.get("explanation"):
            print(f"{result['explanation']}")
