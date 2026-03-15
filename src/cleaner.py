import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """
    Basic text cleanup:
    - remove repeated spaces/tabs
    - normalize line endings
    - merge broken single newlines inside paragraphs
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Merge single newlines inside paragraphs as a complete sentence while keep double newlines as paragraph separators
    text = re.sub(r"(?<!\n)\n(?!\n)", "", text)

    # Clean up repeated spaces again
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()

def remove_page_noise(page_texts: List[Dict]) -> List[str]:
    """
    Remove repeated header appears at the start of many pages and clean text
    """

    if len(page_texts) < 3:
        return page_texts

    start_line_count = {}
    split_pages = []

    for p in page_texts:
        lines = [ln.strip() for ln in p["text"].splitlines() if ln.strip()]
        split_pages.append(lines)

        if lines:
            start_line_count[lines[0]] = start_line_count.get(lines[0], 0) + 1
            
    repeated_starts = {
        line for line, count in start_line_count.items()
        if count >= max(3, len(page_texts) // 2) and len(line) < 80
    }
    
    cleaned_pages = []
    for idx, lines in enumerate(split_pages):
        if lines and lines[0] in repeated_starts:
            lines = lines[1:]

        text = "\n".join(lines)
        #text = clean_text(text)
        cleaned_pages.append(text)
    
    return cleaned_pages