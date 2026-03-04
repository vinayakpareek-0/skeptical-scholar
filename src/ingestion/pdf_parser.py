import re
from pathlib import Path
from typing import List, Dict
import fitz 
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import load_config , PROJECT_ROOT


def extract_text(pdf_path: str) -> str:
    """
    Extracts full text from a PDF file page by page.
    """
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        text = page.get_text()
        if text:
            all_text.append(text)
    doc.close()

    return "\n".join(all_text)


SECTION_HEADERS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Background",
    "Methodology",
    "Method",
    "Methods",
    "Approach",
    "Experiments",
    "Experimental Setup",
    "Results",
    "Discussion",
    "Conclusion",
    "Conclusions",
    "References",
]

# Build dynamic regex pattern
HEADER_PATTERN = r"(?:\n|^)\s*(?:\d+\.?\s*)?(" + "|".join(SECTION_HEADERS) + r")\s*\n"


def detect_sections(full_text: str) -> List[Dict]:
    """
    Detects major academic sections using regex.
    Returns list of dicts: [{"section": "...", "text": "..."}]
    """

    matches = list(re.finditer(HEADER_PATTERN, full_text, flags=re.IGNORECASE))

    # Edge case: no sections detected
    if not matches:
        return [{
            "section": "full_text",
            "text": full_text.strip()
        }]

    sections = []

    for i, match in enumerate(matches):
        section_name = match.group(1)

        start_index = match.end()
        end_index = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        section_text = full_text[start_index:end_index].strip()

        sections.append({
            "section": section_name,
            "text": section_text
        })

    return sections


def parse_paper(pdf_path: str, paper_id: str) -> Dict:
    """
    Full pipeline:
    - Extract text
    - Detect sections
    - Return structured dict
    """

    full_text = extract_text(pdf_path)
    sections = detect_sections(full_text)

    return {
        "paper_id": paper_id,
        "sections": sections,
        "raw_text_length": len(full_text)
    }


if __name__ == "__main__":
    test_pdf = Path(PROJECT_ROOT / "data/raw/arxiv_papers").glob("*.pdf")
    test_pdf = list(test_pdf)

    if not test_pdf:
        print("No PDFs found in data/raw/arxiv_papers/")
    else:
        pdf_path = str(test_pdf[0])
        paper_id = Path(pdf_path).stem

        result = parse_paper(pdf_path, paper_id)

        print(f"\nParsed Paper: {paper_id}")
        print(f"Raw text length: {result['raw_text_length']}")
        print("\nDetected Sections:\n")

        for section in result["sections"]:
            preview = section["text"][:100].replace("\n", " ")
            print(f"[{section['section']}] -> {preview}...\n")