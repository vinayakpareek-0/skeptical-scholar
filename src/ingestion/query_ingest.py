import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PROJECT_ROOT, load_config
from ingestion.arxiv_fetcher import download_pdf, fetch_arxiv_papers
from ingestion.arxiv_fetcher import load_checkpoint, save_checkpoint
from ingestion.chunker import chunk_paper
from ingestion.database import init_db, insert_chunks, insert_papers
from ingestion.pdf_parser import parse_paper
from ingestion.semantic_scholar_fetcher import fetch_semantic_scholar
from retrieval.dense_retriever import rebuild_dense_index
from runtime_cache import clear_runtime_cache


def _existing_paper_ids(conn):
    rows = conn.execute("SELECT paper_id FROM papers").fetchall()
    return {row[0] for row in rows}


def _paper_record(paper, pdf_path: Path):
    return {
        "paper_id": paper["arxiv_id"],
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "abstract": paper.get("abstract", ""),
        "published_date": paper.get("published", ""),
        "arxiv_url": paper.get("pdf_url", ""),
        "pdf_path": str(pdf_path),
    }


def fetch_query_papers(query: str, config: dict):
    ingest_cfg = config.get("query_ingestion", {})
    papers = fetch_arxiv_papers(
        query,
        ingest_cfg.get("arxiv_max_results", 3),
        delay_seconds=ingest_cfg.get("arxiv_client_delay", 1),
    )

    if config.get("semantic_scholar", {}).get("enabled", False):
        papers.extend(
            fetch_semantic_scholar(
                query,
                max_results=ingest_cfg.get("semantic_scholar_max_results", 3),
                min_citations=ingest_cfg.get("semantic_scholar_min_citations", 0),
            )
        )

    unique = []
    seen_ids = set()
    for paper in papers:
        arxiv_id = paper.get("arxiv_id")
        if arxiv_id and arxiv_id not in seen_ids:
            unique.append(paper)
            seen_ids.add(arxiv_id)
    return unique


def ingest_query(query: str, rebuild_index: bool | None = None):
    config = load_config()
    ingest_cfg = config.get("query_ingestion", {})
    raw_dir = PROJECT_ROOT / config["arxiv"]["download_path"]
    checkpoint_path = PROJECT_ROOT / config["data"]["metadata"] / "metadata_checkpoint.json"

    conn = init_db(PROJECT_ROOT / config["database"]["path"])
    existing_ids = _existing_paper_ids(conn)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_ids = {paper.get("arxiv_id") for paper in checkpoint}

    papers = fetch_query_papers(query, config)
    downloaded = 0
    inserted = 0
    total_chunks = 0

    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        if arxiv_id in existing_ids:
            print(f"Skipping existing paper: {arxiv_id}")
            continue

        pdf_path = raw_dir / f"{arxiv_id}.pdf"
        print(f"Downloading: {arxiv_id} - {paper.get('title', '')[:80]}")
        download_pdf(paper, raw_dir)
        downloaded += 1

        if not pdf_path.exists():
            print(f"  PDF unavailable: {arxiv_id}")
            continue

        try:
            parsed = parse_paper(str(pdf_path), arxiv_id)
            chunks = chunk_paper(
                parsed,
                config["chunking"]["max_length"],
                config["chunking"]["overlap"],
            )
            insert_papers(conn, [_paper_record(paper, pdf_path)])
            insert_chunks(conn, chunks)
            existing_ids.add(arxiv_id)
            inserted += 1
            total_chunks += len(chunks)

            if arxiv_id not in checkpoint_ids:
                checkpoint.append(paper)
                checkpoint_ids.add(arxiv_id)

            print(f"  Inserted {len(chunks)} chunks")
        except Exception as exc:
            print(f"  Error processing {arxiv_id}: {exc}")
        finally:
            if pdf_path.exists():
                pdf_path.unlink()

        time.sleep(ingest_cfg.get("download_delay", 0.5))

    conn.close()
    save_checkpoint(checkpoint_path, checkpoint)

    if rebuild_index is None:
        rebuild_index = ingest_cfg.get("rebuild_dense_index", True)

    indexed_chunks = None
    if rebuild_index and inserted:
        indexed_chunks = rebuild_dense_index()
        clear_runtime_cache()

    return {
        "query": query,
        "fetched": len(papers),
        "downloaded": downloaded,
        "inserted": inserted,
        "chunks_added": total_chunks,
        "indexed_chunks": indexed_chunks,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch and ingest papers for a single query.")
    parser.add_argument("query", help="Research topic or natural-language query to ingest.")
    parser.add_argument(
        "--no-rebuild-index",
        action="store_true",
        help="Insert papers without rebuilding the dense FAISS index.",
    )
    args = parser.parse_args()

    result = ingest_query(args.query, rebuild_index=not args.no_rebuild_index)
    print(
        "Done: "
        f"fetched={result['fetched']}, "
        f"inserted={result['inserted']}, "
        f"chunks_added={result['chunks_added']}, "
        f"indexed_chunks={result['indexed_chunks']}"
    )


if __name__ == "__main__":
    main()
