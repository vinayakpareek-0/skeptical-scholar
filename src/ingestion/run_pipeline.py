"""
1. Load config
2. Fetch papers from ArXiv (arxiv_fetcher.fetch_all)
3. Load metadata checkpoint (list of paper dicts)
4. For each paper:
   a. Parse PDF → sections (pdf_parser.parse_paper)
   b. Chunk sections (chunker.chunk_paper)
   c. Insert paper metadata into SQLite (database.insert_papers)
   d. Insert chunks into SQLite (database.insert_chunks)
5. Build citation graph from all parsed papers (citation_parser.build_citation_graph)
6. Save graph to data/graph/citation_graph.json
7. Print summary stats.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PROJECT_ROOT
from config import load_config
from ingestion.arxiv_fetcher import  run_pipeline as fetch_papers , fetch_arxiv_papers, load_checkpoint
from ingestion.pdf_parser import parse_paper
from ingestion.chunker import chunk_paper
from ingestion.database import init_db, insert_papers, insert_chunks
from ingestion.citation_parser import build_citation_graph, save_graph
from pathlib import Path
import networkx as nx



if __name__ =="__main__":
    """
    Run the entire ingestion pipeline.
    """
    config =load_config()
    fetch_papers()
    metadata_checkpoint = load_checkpoint(PROJECT_ROOT / config["data"]["metadata"] / "metadata_checkpoint.json")  
    conn = init_db(PROJECT_ROOT / config["database"]["path"])
    all_parsed=[]
    total_chunks= 0
    graph_ = nx.DiGraph()
    for paper in metadata_checkpoint:
        print(f"Processing paper: {paper['arxiv_id']} {len(all_parsed)}/{len(metadata_checkpoint)}")
        path = str(PROJECT_ROOT / config["arxiv"]["download_path"] / (paper["arxiv_id"] + ".pdf"))
        try:
            parsed_paper= parse_paper(path, paper["arxiv_id"])
            chunks = chunk_paper(parsed_paper, config["chunking"]["max_length"], config["chunking"]["overlap"])
            paper_record = {
                "paper_id": paper["arxiv_id"],
                "title": paper["title"],
                "authors": ",".join(paper["authors"]),
                "abstract": paper["abstract"],
                "published_date": paper["published"],   
                "arxiv_url": paper["pdf_url"],
                "pdf_path": path
            }
            insert_papers(conn, [paper_record])
            insert_chunks(conn, chunks)

            all_parsed.append(parsed_paper)
            total_chunks+=len(chunks)

        except Exception as e:
            print(f"Error processing {paper['arxiv_id']}: {e}")
            continue    
    conn.close()
    graph_ = build_citation_graph(all_parsed)
    save_graph(graph_, str(PROJECT_ROOT / config["data"]["graph"] / "citation_graph.json")) 
 
    print(f" Total Papers: {len(metadata_checkpoint)} , Total Chunks: {total_chunks}, Total Citations: {len(graph_.edges)}")