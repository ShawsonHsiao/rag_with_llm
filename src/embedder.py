from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import os

def load_embedding_model(model_name: str) -> SentenceTransformer:
    
    model_path = "./models"
    os.makedirs(model_path, exist_ok=True)
    
    model = SentenceTransformer(
        model_name,
        cache_folder=model_path
    )

    return model


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return vectors.astype("float32")