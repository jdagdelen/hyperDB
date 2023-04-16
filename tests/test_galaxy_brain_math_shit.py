import numpy as np
from hyperdb import euclidean_metric

# Test totally written by me and not gpt/co-pilot
def test_euclid_bros_metric():
    data_vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    query_vector = np.array([1, 1, 1])
    result = euclidean_metric(data_vectors, query_vector)

    # All important test to see if shapes are based
    try:
        assert result.shape == (3,)
    except AssertionError:
        raise AssertionError("Oh my god, you killed Euclid! You bas-turd!")