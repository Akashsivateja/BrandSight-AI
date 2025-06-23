# vector_search.py

import faiss
import numpy as np

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim after normalization)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def search_similar(index, query_emb, k=5):
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, k)
    return distances, indices
