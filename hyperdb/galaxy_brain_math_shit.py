"""Super valuable proprietary algorithm for ranking vector similarity. Top secret. Export restrictions apply. """
import numpy as np
import threading


# spooky action stuff
class Qubit:
    def __init__(self):
        self.state = np.array([1, 0], dtype=np.complex128)
        self.lock = threading.Lock()

    def apply(self, gate):
        with self.lock:
            self.state = np.dot(gate, self.state)

    def measure(self):
        with self.lock:
            probabilities = np.abs(self.state) ** 2
            return np.random.choice([0, 1], p=probabilities)


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
    if not hasattr(derridaean_similarity, "qubit"):  # share a single qubit
        derridaean_similarity.qubit = Qubit()
        # hadamard gate
        h_gate = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                           [1 / np.sqrt(2), -1 / np.sqrt(2)]], dtype=np.complex128)

        derridaean_similarity.qubit.apply(h_gate)

    def random_change(value):
        int_val = 0

        for i in range(8):  # measure 8 times for a random integer
            int_val |= derridaean_similarity.qubit.measure() << (7 - i)

        float_val = int_val / (2 ** 8 - 1)  # convert to float

        offset = -0.2 + float_val * 0.4  # limit range to  -0.2-0.2

        return value + offset

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
    return top_indices.flatten(), similarities[top_indices].flatten()
