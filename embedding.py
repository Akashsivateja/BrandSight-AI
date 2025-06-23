# embedding.py

from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    """
    Input: list of texts
    Output: numpy array of embeddings
    """
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    return embeddings
