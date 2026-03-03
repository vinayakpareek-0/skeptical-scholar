from typing import List
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config , PROJECT_ROOT
from ingestion.pdf_parser import parse_paper
from pathlib import Path


def chunk_section(section_text:str , max_tokens :int , overlap:int )-> List[str]:
    if overlap>=max_tokens:
            raise ValueError("Overlap cannot be greater than or equal to max_tokens")
    
    chunks =[]
    i=0
    words = section_text.split()
    while i<len(words):
        chunk =" ".join(words[i:i+max_tokens])
        chunks.append(chunk)
        i+=(max_tokens-overlap)
    return chunks


def chunk_paper(parsed_paper:dict , max_tokens :int , overlap:int )-> List[dict]:
    section_chunks=[]
    for section in parsed_paper["sections"]:
        section_text = section["text"]
        section_name = section["section"]
        chunks = chunk_section(section_text, max_tokens , overlap)
        for i,chunk in enumerate(chunks):
            section_chunk = {
                "chunk_id":f"{parsed_paper['paper_id']}_{section_name}_{i}",
                "paper_id":parsed_paper["paper_id"],
                "section":section_name,
                "text":chunk,
                "word_count":len(chunk.split(" "))
            }
            section_chunks.append(section_chunk)
    return section_chunks

if __name__== "__main__":
    config = load_config()
    max_tokens = config["chunking"]["max_length"]
    overlap = config["chunking"]["overlap"]
    
    pdf_path = "data/raw/arxiv_papers/2603.02202v1.pdf"
    paper = parse_paper(pdf_path , paper_id=Path(pdf_path).stem)
    chunks = chunk_paper(paper , max_tokens , overlap)

    print(chunks[0],"\n")
    print(chunks[1],"\n")
    print(chunks[2],"\n")

    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"  {chunk['chunk_id']} — {chunk['word_count']} words")





