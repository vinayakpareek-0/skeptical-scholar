import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config, PROJECT_ROOT
from ingestion.database import init_db, get_all_chunks as load_chunks
from retrieval.bm25_retriever import build_bm25_index, search_bm25
from retrieval.dense_retriever import load_index as load_dense_index, load_dense_model, search_dense
from retrieval.hybrid_retriever import search_hybrid
from retrieval.reranker import load_reranker, rerank
from retrieval.idk_trigger import check_retrieval_confidence


def load_evaluation_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    config = load_config()
    conn = init_db(PROJECT_ROOT / config["database"]["path"])
    chunks = load_chunks(conn)

    bm25_index, chunks = build_bm25_index(chunks)
    dense_index = load_dense_index(PROJECT_ROOT / config["dense"]["index_path"])
    dense_model = load_dense_model(config["dense"]["model_name"])
    reranker = load_reranker(config["retrieval"]["reranker_name"])
    threshold = config["retrieval"]["idk_threshold"]

    eval_path = os.path.join(os.path.dirname(__file__), "evaluation.json")
    eval_set = load_evaluation_set(eval_path)

    results_table = []

    for q in eval_set["queries"]:
        query = q["query"]
        category = q["category"]

        bm25_results = search_bm25(bm25_index, query, chunks, top_k=5)
        bm25_top = bm25_results[0]["score"] if bm25_results else 0

        dense_results = search_dense(dense_index, dense_model, query, chunks, top_k=5)
        dense_top = dense_results[0]["score"] if dense_results else 0

        hybrid_results = search_hybrid(query, bm25_index, dense_index, dense_model, chunks, top_k=20)
        hybrid_top = hybrid_results[0]["score"] if hybrid_results else 0

        reranked = rerank(reranker, query, hybrid_results, top_k=5)
        rerank_top = reranked[0]["rerank_score"] if reranked else 0

        idk = check_retrieval_confidence(reranked, threshold=threshold)

        results_table.append({
            "id": q["id"],
            "query": query[:50],
            "category": category,
            "bm25_top": round(bm25_top, 3),
            "dense_top": round(dense_top, 3),
            "hybrid_top": round(hybrid_top, 5),
            "rerank_top": round(rerank_top, 3),
            "idk_triggered": idk["triggered"],
        })

    print(f"{'ID':>3} {'Query':<50} {'Cat':<12} {'BM25':>8} {'Dense':>8} {'Hybrid':>10} {'Rerank':>8} {'IDK':>5}")
    print("-" * 110)

    for r in results_table:
        idk_str = "YES" if r["idk_triggered"] else "no"
        print(f"{r['id']:>3} {r['query']:<50} {r['category']:<12} {r['bm25_top']:>8} {r['dense_top']:>8} {r['hybrid_top']:>10} {r['rerank_top']:>8} {idk_str:>5}")

    in_domain = [r for r in results_table if r["category"] == "in_domain"]
    out_domain = [r for r in results_table if r["category"] == "out_of_domain"]
    adversarial = [r for r in results_table if r["category"] == "adversarial_tricky"]

    print("\nSUMMARY")
    if in_domain:
        avg = sum(r["rerank_top"] for r in in_domain) / len(in_domain)
        false_idk = sum(1 for r in in_domain if r["idk_triggered"])
        print(f"In-domain    ({len(in_domain)}): Avg rerank = {avg:.3f}, False IDK = {false_idk}")

    if out_domain:
        avg = sum(r["rerank_top"] for r in out_domain) / len(out_domain)
        correct_idk = sum(1 for r in out_domain if r["idk_triggered"])
        print(f"Out-of-domain({len(out_domain)}): Avg rerank = {avg:.3f}, IDK accuracy = {correct_idk}/{len(out_domain)}")

    if adversarial:
        avg = sum(r["rerank_top"] for r in adversarial) / len(adversarial)
        idk_count = sum(1 for r in adversarial if r["idk_triggered"])
        print(f"Adversarial  ({len(adversarial)}): Avg rerank = {avg:.3f}, IDK triggered = {idk_count}/{len(adversarial)}")

    out_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_evaluation()
