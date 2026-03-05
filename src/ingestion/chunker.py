from typing import List
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config , PROJECT_ROOT
from ingestion.pdf_parser import parse_paper
from pathlib import Path
import re


def chunk_section(section_text:str , max_tokens :int , overlap:int )-> List[str]:
    if overlap>=max_tokens:
            raise ValueError("Overlap cannot be greater than or equal to max_tokens")
    
    chunks =[]
    sentences = re.split(r'(?<=[.!?])\s+', section_text)
    current_chunk = []
    current_word_count = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current_word_count + words > max_tokens and current_chunk:
            # Save this chunk, start new one with overlap
            chunks.append(" ".join(current_chunk))
            overlap_text = " ".join(current_chunk[-2:])  # last 2 sentences as overlap
            current_chunk = [overlap_text, sentence]
            current_word_count = len(overlap_text.split()) + words
        else:
            current_chunk.append(sentence)
            current_word_count += words
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
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
    
    pdf_path = "data/raw/arxiv_papers/2206.03003v2.pdf"
    paper = parse_paper(pdf_path , paper_id=Path(pdf_path).stem)
    chunks = chunk_paper(paper , max_tokens , overlap)

    print(chunks[0],"\n")
    print(chunks[1],"\n")
    print(chunks[2],"\n")

    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"  {chunk['chunk_id']} — {chunk['word_count']} words")





