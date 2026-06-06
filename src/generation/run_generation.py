"""
Generation pipeline: Reasoning → Prompt → LLM → NLI Verify → IDK 3 → Final Answer
"""
import sys
import os
from time import perf_counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reasoning.run_reasoning import run_reasoning
from generation.llm_client import generate
from generation.prompts import build_prompt, build_idk_prompt
from generation.nli_verifier import verify_answer
from generation.idk_trigger3 import check_generation_confidence
from runtime_cache import get_config, get_llm_client, get_nli_model


def run_generation(query):
    started_at = perf_counter()
    config = get_config()
    timings = {}

    reasoning_started_at = perf_counter()
    reasoning = run_reasoning(query)
    timings["reasoning"] = round(perf_counter() - reasoning_started_at, 3)

    if reasoning["status"] == "idk":
        explanation = None
        if config.get("runtime", {}).get("explain_idk_with_llm", True):
            llm_started_at = perf_counter()
            client = get_llm_client()
            idk_prompt = build_idk_prompt(query, reasoning["reason"])
            explanation = generate(client, idk_prompt)
            timings["llm"] = round(perf_counter() - llm_started_at, 3)
        timings["total"] = round(perf_counter() - started_at, 3)
        return {
            "status": "idk",
            "reason": reasoning["reason"],
            "explanation": explanation,
            "answer": None,
            "timings": timings
        }

    llm_started_at = perf_counter()
    client = get_llm_client()
    prompt = build_prompt(query, reasoning["chunks"], reasoning["confidence"])
    answer = generate(client, prompt)
    timings["llm"] = round(perf_counter() - llm_started_at, 3)

    if config["generation"].get("verify_answer", True):
        nli_started_at = perf_counter()
        nli = get_nli_model()
        nli_result = verify_answer(nli, answer, reasoning["chunks"])
        timings["nli_verification"] = round(perf_counter() - nli_started_at, 3)

        idk3 = check_generation_confidence(answer, nli_result)
        if idk3["triggered"]:
            timings["total"] = round(perf_counter() - started_at, 3)
            return {
                "status": "idk",
                "reason": f"IDK Layer 3: {idk3['reason']}",
                "answer": answer,
                "nli": nli_result,
                "timings": timings
            }
    else:
        nli_result = {
            "supported": 0,
            "contradicted": 0,
            "neutral": 0,
            "is_reliable": True,
            "skipped": True
        }
        idk3 = check_generation_confidence(answer, nli_result)
        if idk3["triggered"]:
            timings["total"] = round(perf_counter() - started_at, 3)
            return {
                "status": "idk",
                "reason": f"IDK Layer 3: {idk3['reason']}",
                "answer": answer,
                "nli": nli_result,
                "timings": timings
            }

    timings["total"] = round(perf_counter() - started_at, 3)
    return {
        "status": "answered",
        "answer": answer,
        "confidence": reasoning["confidence"],
        "nli": nli_result,
        "citations": [{"paper_id": c["paper_id"], "section": c["section"]} for c in reasoning["chunks"]],
        "entities": reasoning.get("entities", []),
        "contradictions": reasoning.get("contradictions", []),
        "timings": timings
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
