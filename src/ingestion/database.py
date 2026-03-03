
"""
-> 3 tables: 
papers    → paper_id (PK), title, authors, abstract, published_date, arxiv_url, pdf_path
chunks    → chunk_id (PK), paper_id (FK), section, text, word_count
citations → source_paper_id (FK), cited_title
"""
import sqlite3
from pathlib import Path
from typing import List, Dict
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config , PROJECT_ROOT  # type:ignore

def init_db(db_path:str):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True , exist_ok=True)
    db = sqlite3.connect(db_path)
    cursor = db.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS papers(
        paper_id TEXT PRIMARY KEY,
        title TEXT,
        authors TEXT,
        abstract TEXT,
        published_date TEXT,
        arxiv_url TEXT,
        pdf_path TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks(
        chunk_id TEXT PRIMARY KEY,
        paper_id TEXT,
        section TEXT,
        text TEXT,
        word_count INTEGER,
        FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
    ) 
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS citations(
        citation_id TEXT ,
        source_paper_id TEXT,
        cited_title TEXT,
        FOREIGN KEY (source_paper_id) REFERENCES papers(paper_id)
    )
    """)
    db.commit()

def insert_papers(conn:sqlite3.Connection , papers:List[Dict]):
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR IGNORE INTO papers(paper_id , title , authors , abstract , published_date , arxiv_url , pdf_path) Values(?,?,?,?,?,?,?)
    """,[
        (paper["paper_id"] , paper["title"] , ",".join(paper["authors"]), paper["abstract"] , paper["published_date"] , paper["arxiv_url"] , paper["pdf_path"])
        for paper in papers
    ])
    conn.commit()


def insert_chunks(conn:sqlite3.Connection , chunks:List[Dict]):
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR IGNORE INTO chunks(chunk_id , paper_id , section , text , word_count)
    VALUES (?,?,?,?,?)
    """,[
        (chunk["chunk_id"] , chunk["paper_id"] , chunk["section"] , chunk["text"] , chunk["word_count"])
        for chunk in chunks
    ])
    conn.commit()

def get_all_chunks(conn:sqlite3.Connection)->List[Dict]:
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chunks")
    rows = cursor.fetchall()
    return [dict(row) for row in rows] 

def get_paper(conn , paper_id:str)->Dict:
    conn.row_factory = sqlite3.Row
    cursor =conn.cursor() 
    cursor.execute("SELECT * FROM papers WHERE paper_id = ?",(paper_id,))
    row = cursor.fetchone()
    return dict(row) if row else None 

if __name__ =="__main__":
    config =load_config()   
    db_path = PROJECT_ROOT/config["database"]["path"]
    init_db(db_path)
    db = sqlite3.connect(db_path)
    print(get_all_chunks(db)) 