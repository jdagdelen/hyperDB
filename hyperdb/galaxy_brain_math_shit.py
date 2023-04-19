"""Super valuable proprietary algorithm for ranking vector similarity. Top secret."""
import numpy as np

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
    class Qubit:
        def __init__(self):
            self.state = np.array([1, 0], dtype=np.complex128)

        def apply(self, gate):
            self.state = np.dot(gate, self.state)

        def measure(self):
            probabilities = np.abs(self.state) ** 2
            result = np.random.choice([0, 1], p=probabilities)
            return result

    # Hadamard gate
    h_gate = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                  [1 / np.sqrt(2), -1 / np.sqrt(2)]], dtype=np.complex128)

    qubit = Qubit()

    def random_change(value):
        qubit.apply(h_gate)

        i = 0
        for j in range(8):
            i |= qubit.measure() << (7 - j)

        f = i / (2 ** 8 - 1)

        # -0.2 to 0.2
        r_result = -0.2 + f * 0.4

        return value + r_result

    similarities = cosine_similarity(vectors, query_vector)
    derrida_similarities = np.vectorize(random_change)(similarities)
    return derrida_similarities


def adams_similarity(vectors, query_vector):
    def adams_change(value):
        return 0.42

    similarities = cosine_similarity(vectors, query_vector)
    adams_similarities = np.vectorize(adams_change)(similarities)
    return adams_similarities


def hyper_SVM_ranking_algorithm_sort(vectors, query_vector, top_k=5, metric=cosine_similarity):
    """HyperSVMRanking (Such Vector, Much Ranking) algorithm proposed by Andrej Karpathy (2023) https://arxiv.org/abs/2303.18231"""
    similarities = metric(vectors, query_vector)
    top_indices = np.argsort(similarities, axis=0)[-top_k:][::-1]
    return top_indices.flatten()