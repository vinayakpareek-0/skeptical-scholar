def check_generation_confidence(answer , nli_results):
    if not nli_results["is_reliable"]:
        return {
            "triggered": True,
            "reason": f"Answer contradicts {nli_results['contradicted']*100:.0f}% of sources",
            "suggestion": "Generated answer may contain hallucination"
        }
    
    if len(answer.split()) < 15:
        return {
            "triggered": True,
            "reason": "Answer too brief to be substantive",
            "suggestion": "Insufficient evidence for detailed response"
        }

    hedges = ["might", "possibly", "i'm not sure", "it is unclear", "uncertaisn"]
    hedge_count = sum(1 for h in hedges if h in answer.lower())
    if hedge_count >= 2:
        return {
            "triggered": True,
            "reason": f"Answer contains {hedge_count} hedging phrases",
            "suggestion": "Low confidence in generated response"
        }
    
    return {"triggered": False, "reason": "Answer verified"}