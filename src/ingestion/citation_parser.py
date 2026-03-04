import re
from typing import List
import networkx as nx
import sys
import os
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ingestion.pdf_parser import parse_paper
from config import PROJECT_ROOT

def extract_references(references_text: str) -> List[str]:
    """
    Extracts cited paper titles from a References section.
    Handles multiple citation formats common in ArXiv papers:
    IEEE, APA, numbered, and unnumbered styles.
    """

    # Normalize whitespace (PDF text has random line breaks)
    text = re.sub(r'\s+', ' ', references_text).strip()

    # --- Step 1: Split into individual reference entries ---
    entries = []

    # Try numbered formats: [1], [2] or 1. 2. at start of entries
    numbered = re.split(r'\[\d+\]\s*', text)
    if len(numbered) > 3:
        entries = numbered
    else:
        # Try dot-numbered: "1." "2." at line-ish boundaries
        dot_numbered = re.split(r'(?:^|\s)(\d{1,3})\.\s+(?=[A-Z])', text)
        if len(dot_numbered) > 5:
            # re.split with groups interleaves numbers — take every other chunk
            entries = [dot_numbered[i] for i in range(2, len(dot_numbered), 2)]
        else:
            # Fallback: split on patterns that look like new references
            # (capital letter after a year or period)
            entries = re.split(r'(?<=\d{4}\.)\s+(?=[A-Z])', text)

    # Filter out junk entries
    entries = [e.strip() for e in entries if len(e.strip()) > 40]

    if not entries:
        return []

    # --- Step 2: Extract title from each entry ---
    titles = []

    for entry in entries:

        # Skip URLs, DOIs, access dates
        if re.match(r'^https?://', entry) or entry.startswith('doi:'):
            continue

        # Clean the entry
        entry = re.sub(r'https?://\S+', '', entry)  # remove URLs
        entry = re.sub(r'doi:\S+', '', entry)         # remove DOIs
        entry = entry.strip()

        if len(entry) < 30:
            continue

        

        title = None

        # Quoted title (IEEE style)
        # "Title of the paper," or "Title of the paper."
        quoted = re.search(r'["\u201c](.*?)["\u201d]', entry)
        if quoted and len(quoted.group(1)) > 10:
            title = quoted.group(1).rstrip('.,;')

        # APA style -  Author (YEAR). Title. Journal
        if not title:
            apa = re.search(r'\(\d{4}[a-z]?\)\.\s*(.*?)(?:\.\s+[A-Z]|\.\s*$)', entry)
            if apa and len(apa.group(1)) > 10:
                title = apa.group(1).rstrip('.')

        # Author et al. Title. Venue/Journal, YEAR
        # Look for text between first period-space and next period
        if not title:
            # Skip author block (ends with period), then grab title (ends with period)
            after_author = re.search(
                r'(?:et al\.|[A-Z]\.|[a-z]\,)\s+(.*?)\.\s+(?:In\s|Proc|IEEE|ACM|ICML|NeurIPS|ICLR|AAAI|CVPR|ECCV|NAACL|EMNLP|ACL|arXiv|Advances|Journal|Trans)',
                entry
            )
            if after_author and len(after_author.group(1)) > 10:
                title = after_author.group(1).rstrip('.,;')

        # After "et al." or comma-separated authors, grab next sentence
        if not title:
            et_al = re.search(r'et al\.\s*(.*?)\.', entry)
            if et_al and len(et_al.group(1)) > 10:
                title = et_al.group(1).strip()

        # Fallback — longest meaningful segment between periods
        if not title:
            segments = [s.strip() for s in entry.split('.') if len(s.strip()) > 15]
            # Skip first segment (likely authors) if there are multiple
            candidates = segments[1:] if len(segments) > 2 else segments
            if candidates:
                title = max(candidates, key=len)

        if title and len(title) > 10:
            # Final cleanup
            title = re.sub(r'\s+', ' ', title).strip()
            title = title.rstrip('.,;:')
            comma_count = title.count(',')
            if comma_count >= 3:
                continue
            titles.append(title)

        if len(titles) >= 150:
            break

    return titles

def build_citation_graph(papers:list[dict])-> nx.DiGraph:
    """
    Builds a citation graph where keys are paper IDs and values are lists of cited paper IDs.
    """
    graph = nx.DiGraph()

    for paper in papers:
        paper_id = paper["paper_id"]
        graph.add_node(paper_id)

        for section in paper["sections"]:
            if section["section"] == "References":
                references_text = section["text"]
                cited_titles = extract_references(references_text)
                for title in cited_titles:
                    graph.add_edge(paper_id, title, relation="cites")

    return graph

def save_graph(graph:nx.DiGraph, path:str):
    """
    Saves the citation graph to a file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Graph saved to {path}") 

def load_graph(path:str)->nx.DiGraph:
    """
    Loads the citation graph from a file.
    """
    with open(path,'r')as f:
        loaded_data = json.load(f)

    loaded_graph = nx.node_link_graph(loaded_data)
    
    print(f"Loaded graph with {loaded_graph.number_of_nodes()} nodes and {loaded_graph.number_of_edges()} edges")
    return loaded_graph

if __name__ == "__main__":
    pdf_path = "data/raw/arxiv_papers/2603.02208v1.pdf" # use PROJECT_ROOT
    parsed = parse_paper(str(PROJECT_ROOT / pdf_path), Path(pdf_path).stem)
    # Find the References section
    for section in parsed["sections"]:
        if section["section"].lower() == "references":
            real_refs = extract_references(section["text"])
            print(f"\nReal paper references found: {len(real_refs)}")
            for r in real_refs[:30]:  # print first 10
                print(f"  - {r}")
            break   