import json
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation.run_generation import run_generation


def load_evaluation_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_result(result, category):
    if category == "in_domain":
        if result["status"] == "answered" and result.get("nli", {}).get("supported", 0) > 0.5:
            return True, "Answered with NLI support"
        return False, f"Status={result['status']}, NLI={result.get('nli', {})}"

    if category == "out_of_domain":
        if result["status"] == "idk":
            return True, "Correctly rejected"
        return False, f"Should have rejected, got status={result['status']}"

    if category == "adversarial_tricky":
        if result["status"] == "idk":
            return True, "Rejected adversarial"
        conf = result.get("confidence", {}).get("score", 1)
        if conf < 0.5:                                                          # baad mei kam krna hai: adversial ke liye
            return True, f"Low confidence ({conf:.3f})"
        return False, f"Answered with high confidence ({conf:.3f})"


def run_generation_eval():
    eval_path = os.path.join(os.path.dirname(__file__), "evaluation.json")
    eval_set = load_evaluation_set(eval_path)

    results = []

    for q in eval_set["queries"]:
        query = q["query"]
        print(f"[{q['id']}/{len(eval_set['queries'])}] {query[:60]}...")
        
        start = time.time()
        result = run_generation(query)
        elapsed = round(time.time() - start, 1)

        passed, reason = score_result(result, q["category"])

        entry = {
            "id": q["id"],
            "query": query[:50],
            "category": q["category"],
            "status": result["status"],
            "confidence": result.get("confidence", {}).get("score", None),
            "nli_supported": result.get("nli", {}).get("supported", None),
            "nli_contradicted": result.get("nli", {}).get("contradicted", None),
            "answer": result.get("answer", None),
            "answer_length": len(result.get("answer", "").split()) if result.get("answer") else 0,
            "time_s": elapsed,
            "passed": passed,
            "reason": reason
        }
        results.append(entry)

    print(f"\n{'ID':>3} {'Query':<50} {'Cat':<12} {'Status':<10} {'Conf':>6} {'NLI_S':>6} {'Pass':>5} {'Time':>5}")
    print("-" * 105)

    for r in results:
        conf = f"{r['confidence']:.3f}" if r["confidence"] is not None else "  -"
        nli = f"{r['nli_supported']:.2f}" if r["nli_supported"] is not None else "  -"
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:>3} {r['query']:<50} {r['category']:<12} {r['status']:<10} {conf:>6} {nli:>6} {status:>5} {r['time_s']:>5}")

    in_d = [r for r in results if r["category"] == "in_domain"]
    out_d = [r for r in results if r["category"] == "out_of_domain"]
    adv = [r for r in results if r["category"] == "adversarial_tricky"]

    print("\nSUMMARY")
    for label, group in [("In-domain", in_d), ("Out-of-domain", out_d), ("Adversarial", adv)]:
        if group:
            passed = sum(1 for r in group if r["passed"])
            print(f"{label:<14} {passed}/{len(group)} passed")

    total_passed = sum(1 for r in results if r["passed"])
    print(f"{'Overall':<14} {total_passed}/{len(results)} ({total_passed/len(results)*100:.0f}%)")

    out_path = os.path.join(os.path.dirname(__file__), "generation_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_generation_eval()

