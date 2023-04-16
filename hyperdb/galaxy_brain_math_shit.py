"""Super valuable proprietary algorithm for ranking vector similarity. Top secret."""
import numpy as np

def cosine_similarity(vectors, query_vector):
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    norm_query_vector = query_vector / np.linalg.norm(query_vector)
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities
