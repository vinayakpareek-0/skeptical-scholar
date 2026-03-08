import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PROJECT_ROOT, load_config
from ingestion.arxiv_fetcher import run_pipeline as fetch_arxiv, load_checkpoint, save_checkpoint
from ingestion.semantic_scholar_fetcher import run_semantic_scholar_fetch
from ingestion.pdf_parser import parse_paper
from ingestion.chunker import chunk_paper
from ingestion.database import init_db, insert_papers, insert_chunks
from ingestion.citation_parser import build_citation_graph, save_graph
from ingestion.arxiv_fetcher import download_pdf
from pathlib import Path
import networkx as nx


def run_full_pipeline():
    config = load_config()
    raw_dir = PROJECT_ROOT / config["arxiv"]["download_path"]

    print("Fetching from ArXiv")
    fetch_arxiv()

    print("\nFetching from Semantic Scholar")
    ss_papers = run_semantic_scholar_fetch()

    checkpoint_path = PROJECT_ROOT / config["data"]["metadata"] / "metadata_checkpoint.json"
    existing = load_checkpoint(checkpoint_path)
    existing_ids = {p["arxiv_id"] for p in existing}

    for paper in ss_papers:
        if paper["arxiv_id"] not in existing_ids:
            download_pdf(paper, raw_dir)
            existing.append(paper)
            existing_ids.add(paper["arxiv_id"])
            print(f"Downloaded (SS): {paper['arxiv_id']} [{paper.get('citation_count', '?')} cites]")
    save_checkpoint(checkpoint_path, existing)

    print(f"\nProcessing {len(existing)} papers")
    conn = init_db(PROJECT_ROOT / config["database"]["path"])
    all_parsed = []
    total_chunks = 0

    for i, paper in enumerate(existing):
        pdf_path = raw_dir / f"{paper['arxiv_id']}.pdf"
        print(f"[{i+1}/{len(existing)}] {paper['arxiv_id']}")

        if not pdf_path.exists():
            print(f"  Skipping (PDF not found)")
            continue

        try:
            parsed = parse_paper(str(pdf_path), paper["arxiv_id"])
            chunks = chunk_paper(parsed, config["chunking"]["max_length"], config["chunking"]["overlap"])

            paper_record = {
                "paper_id": paper["arxiv_id"],
                "title": paper["title"],
                "authors": ",".join(paper["authors"]) if isinstance(paper["authors"], list) else paper["authors"],
                "abstract": paper.get("abstract", ""),
                "published_date": paper.get("published", ""),
                "arxiv_url": paper.get("pdf_url", ""),
                "pdf_path": str(pdf_path)
            }
            insert_papers(conn, [paper_record])
            insert_chunks(conn, chunks)

            all_parsed.append(parsed)
            total_chunks += len(chunks)

            os.remove(str(pdf_path))             # delete PDF after processing

        except Exception as e:
            print(f"  Error: {e}")
            continue

    conn.close()

    graph = build_citation_graph(all_parsed)
    save_graph(graph, str(PROJECT_ROOT / config["data"]["graph"] / "citation_graph.json"))

    print(f"\nDone! Papers: {len(existing)}, Chunks: {total_chunks}, Citations: {len(graph.edges)}")


if __name__ == "__main__":
    run_full_pipeline()