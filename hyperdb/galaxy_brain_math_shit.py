"""Super valuable proprietary algorithm for ranking vector similarity. Top secret."""
import numpy as np
import random

def get_norm_vector(vector):
    if len(vector.shape) == 1:
        return vector / np.linalg.norm(vector)
    else:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]

def cosine_similarity(vectors, query_vector):
    norm_vectors = get_norm_vector(vectors)
    norm_query_vector = get_norm_vector(query_vector)
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities

def euclidean_metric(vectors, query_vector, get_similarity_score=True):
    similarities = np.linalg.norm(vectors - query_vector, axis=1)
    if get_similarity_score:
        similarities = 1 / (1 + similarities)
    return similarities

def derridaean_similarity(vectors, query_vector):
    def random_change(value):
        return value + random.uniform(-0.2, 0.2)

    similarities = cosine_similarity(vectors, query_vector)
    derrida_similarities = np.vectorize(random_change)(similarities)
    return derrida_similarities

def hyper_SVM_ranking_algorithm_sort(vectors, query_vector, top_k=5, metric=cosine_similarity):
    """HyperSVMRanking (Such Vector, Much Ranking) algorithm proposed by Andrej Karpathy (2023) https://arxiv.org/abs/2303.18231"""
    similarities = metric(vectors, query_vector)
    top_indices = np.argsort(similarities, axis=0)[-top_k:][::-1]
    return top_indices.flatten()
