import os

import json
from typing import List, Dict, Tuple
import pymupdf 

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    extract text from a PDF file 
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    
    doc = pymupdf.open(pdf_path)
    text_data = []
    
    for page_id in range(len(doc)):
        page = doc[page_id]
        text = page.get_text("text", sort=True)
        text_data.append({
            "page_num": page_id + 1,
            "text": text
        })

    doc.close()
    return text_data