"""Super valuable proprietary algorithm for ranking vector similarity. Top secret."""
import numpy as np


def cosine_similarity(vectors, query_vector):
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    norm_query_vector = query_vector / np.linalg.norm(query_vector)
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities


def hyper_SVM_ranking_algorithm_sort(vectors, query_vector, top_k=5):
    """HyperSVMRanking (Such Vector, Much Ranking) algorithm proposed by Andrej Karpathy (2023) https://arxiv.org/abs/2303.18231"""
    similarities = cosine_similarity(vectors, query_vector)
    top_indices = np.argsort(similarities, axis=0)[-top_k:][::-1]
    return top_indices.flatten()