import arxiv
import json
import time
import requests
from pathlib import Path
from typing import List, Dict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config , PROJECT_ROOT



def fetch_arxiv_papers(query: str, max_results: int) -> List[Dict]:
    client = arxiv.Client(delay_seconds=2, num_retries=3)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []

    for result in client.results(search):
        arxiv_id = result.entry_id.split("/")[-1]

        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "published": str(result.published),
            "arxiv_id": arxiv_id,
            "pdf_url": result.pdf_url,
            "query": query
        })

    return papers


def download_pdf(paper: Dict, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_path = raw_dir / f"{paper['arxiv_id']}.pdf"

    if file_path.exists():
        return  # Already downloaded

    response = requests.get(paper["pdf_url"], timeout=30)

    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed: {paper['arxiv_id']}")


def load_checkpoint(path: Path) -> List[Dict]:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_checkpoint(path: Path, data: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_pipeline():
    config = load_config()
    queries = config["arxiv"]["queries"]
    max_results = config["arxiv"]["max_results"]

    raw_dir = PROJECT_ROOT / config["arxiv"]["download_path"]
    checkpoint_path = PROJECT_ROOT / config["data"]["metadata"] / "metadata_checkpoint.json"

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    existing_metadata = load_checkpoint(checkpoint_path)
    existing_ids = {paper["arxiv_id"] for paper in existing_metadata}

    all_metadata =existing_metadata.copy()
    for query in queries:
        papers = fetch_arxiv_papers(query, max_results)
        for paper in papers:
            if paper["arxiv_id"] in existing_ids:
                print(f"Skipping (already exists): {paper['arxiv_id']}")
                continue
            download_pdf(paper, raw_dir)
            time.sleep(config["arxiv"]["download_delay"]) 
            all_metadata.append(paper)
            existing_ids.add(paper["arxiv_id"])
            print(f"Saved: {paper['arxiv_id']}")
        save_checkpoint(checkpoint_path , all_metadata)
    
    print("\nPipeline completed.")


if __name__ =="__main__":
    run_pipeline()
