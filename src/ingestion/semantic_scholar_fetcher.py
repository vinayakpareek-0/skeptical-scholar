"""
Semantic Scholar API fetcher — gets highly-cited papers
"""
import requests
import json
import time
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config, PROJECT_ROOT


def fetch_semantic_scholar(query, max_results=20, min_citations=10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,citationCount,externalIds,authors,year,url"
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        print(f"Semantic Scholar API error: {response.status_code}")
        return []

    data = response.json().get("data", [])
    papers = []

    for paper in data:
        ext_ids = paper.get("externalIds", {}) or {}
        arxiv_id = ext_ids.get("ArXiv")

        if not arxiv_id:
            continue
        if (paper.get("citationCount", 0) or 0) < min_citations:
            continue

        papers.append({
            "title": paper.get("title", ""),
            "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
            "abstract": paper.get("abstract", "") or "",
            "published": str(paper.get("year", "")),
            "arxiv_id": arxiv_id,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "query": query,
            "citation_count": paper.get("citationCount", 0),
            "source": "semantic_scholar"
        })

    papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
    return papers


def run_semantic_scholar_fetch():
    config = load_config()
    ss_config = config.get("semantic_scholar", {})

    if not ss_config.get("enabled", False):
        print("Semantic Scholar fetching disabled")
        return []

    queries = ss_config.get("queries", [])
    max_results = ss_config.get("max_results", 20)
    min_citations = ss_config.get("min_citations", 10)

    all_papers = []
    seen_ids = set()

    for query in queries:
        print(f"Fetching from Semantic Scholar: '{query}'")
        papers = fetch_semantic_scholar(query, max_results, min_citations)
        for p in papers:
            if p["arxiv_id"] not in seen_ids:
                all_papers.append(p)
                seen_ids.add(p["arxiv_id"])
        time.sleep(3)  # rate limit

    print(f"Total unique papers from Semantic Scholar: {len(all_papers)}")
    return all_papers


if __name__ == "__main__":
    papers = run_semantic_scholar_fetch()
    for p in papers[:10]:
        print(f"  [{p['citation_count']}] {p['arxiv_id']}: {p['title'][:80]}")
