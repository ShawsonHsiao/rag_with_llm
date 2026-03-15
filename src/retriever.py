from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import faiss
import json

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  
    index.add(vectors)
    return index

def search_index(model: SentenceTransformer,index: faiss.Index,chunks: List[Dict],query: str,top_k: int = 3) -> List[Dict]:
    
    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = chunks[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return results

def save_artifacts(index: faiss.Index, chunks: List[Dict], index_path: str, meta_path: str) -> None:
    
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)