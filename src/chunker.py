import re
from typing import List, Dict, Tuple
from src.cleaner import clean_text

# Patterns to identify chapters and articles in documents
CHAPTER_PATTERN = re.compile(r"^第\s*[一二三四五六七八九十百零〇]+\s*章.*$")
ARTICLE_PATTERN = re.compile(r"^第\s*\d+\s*條$")

def split_sentences(text: str) -> List[str]:
    """
    Split Chinese text into sentences based on punctuation.
    """

    parts = re.split(r'(?<=[。！？；])', text)

    return [p.strip() for p in parts if p.strip()]

def parse_legal_sections(pages: List[str]) -> List[Dict]:
    """
    Parse the cleaned page texts into sections based on chapter and article patterns.
    """

    current_chapter = ""
    current_article = None
    sections = []

    for page in pages:
        
        lines = [line.strip() for line in page.splitlines() if line.strip()]

        for line in lines:

            if CHAPTER_PATTERN.match(line):
                current_chapter = line
                continue

            if ARTICLE_PATTERN.match(line):
                # save previous article
                if current_article is not None:
                    current_article["content"] = clean_text(current_article["content"])
                    sections.append(current_article)

                current_article = {
                    "chapter_title": current_chapter,
                    "article_title": line,
                    "content": ""
                }
                continue

            if current_article is not None:
                current_article["content"] += line + "\n"

    if current_article is not None:
        current_article["content"] = clean_text(current_article["content"])
        sections.append(current_article)

    return sections

def chunk_article_with_overlap(section: Dict, target_chars: int = 500, overlap_sentences: int = 1) -> List[Dict]:

    header = (
        f"{section['chapter_title']}\n"
        f"{section['article_title']}\n"
    )

    sentences = split_sentences(section["content"])
    chunks = []
    current_sents = []
    i = 0

    while i < len(sentences):
        current_sents = []
        current_text = ""

        j = i
        while j < len(sentences):
            candidate = current_text + sentences[j]
            if len(header) + len("內容：") + len(candidate) <= target_chars:
                current_text = candidate
                current_sents.append(sentences[j])
                j += 1
            else:
                break

        if current_text.strip():
            chunks.append({
                "chapter_title": section["chapter_title"],
                "article_title": section["article_title"],
                "chunk_text": header + "內容：" + current_text.strip()
            })
            

        if j == len(sentences):
            break

        i = max(i + 1, j - overlap_sentences)

    return chunks

def build_chunks(pages: List[str], target_chars: int = 500) -> List[Dict]:

    sections = parse_legal_sections(pages)

    all_chunks = []
    for section in sections:
        all_chunks.extend(chunk_article_with_overlap(section, target_chars=target_chars))

    for idx, chunk in enumerate(all_chunks, start=1):
        chunk["chunk_id"] = f"chunk_{idx:04d}"

    return all_chunks

