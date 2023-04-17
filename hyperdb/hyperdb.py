import pickle
import numpy as np
import gzip
import openai
from hyperdb.galaxy_brain_math_shit import cosine_similarity, euclidean_metric, derridaean_similarity, hyper_SVM_ranking_algorithm_sort

def get_embedding(documents, key=None, model="text-embedding-ada-002"):
    """Default embedding function that uses OpenAI Embeddings."""
    if isinstance(documents, list):
        if isinstance(documents[0], dict):
            if "." in key:
                key_chain = key.split(".")
            else: 
                key_chain = [key]
            texts = []
            for doc in documents:
                for key in key_chain:
                    doc = doc[key]
                texts.append(doc.replace("\n", " "))
        elif isinstance(documents[0], str):
            texts = documents
    response = openai.Embedding.create(input=texts, model=model)
    embeddings = [np.array(item["embedding"]) for item in response["data"]]
    return embeddings


class HyperDB:
    def __init__(self, documents, key, embedding_function=None, similarity_metric="cosine"):
        if embedding_function is None:
            embedding_function = lambda docs: get_embedding(docs, key=key)
        self.documents = []
        self.embedding_function = embedding_function
        self.vectors = None
        self.add_documents(documents)
        if similarity_metric.__contains__("cosine"):
            self.similarity_metric = cosine_similarity
        elif similarity_metric.__contains__("euclidean"):
            self.similarity_metric = euclidean_metric
        elif similarity_metric.__contains__("derrida"):
            self.similarity_metric = derridaean_similarity
        else:
            raise Exception("Similarity metric not supported. Please use either 'cosine', 'euclidean' or 'derrida'.")

    def add_document(self, document, vector=None):
        if vector is None:
            vector = self.embedding_function([document])[0]
        if self.vectors is None:
            self.vectors = np.empty((0, len(vector)), dtype=np.float32)
        elif len(vector) != self.vectors.shape[1]:
            raise ValueError("All vectors must have the same length.")
        self.vectors = np.vstack([self.vectors, vector]).astype(np.float32)
        self.documents.append(document)

    def add_documents(self, documents, vectors=None):
        if vectors is None:
            vectors = np.array(self.embedding_function(documents)).astype(np.float32)
        for vector, document in zip(vectors, documents):
            self.add_document(document, vector)

    def save(self, storage_file):
        data = {
            "vectors": self.vectors,
            "documents": self.documents
        }
        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(storage_file, "wb") as f:
                pickle.dump(data, f)

    def load(self, storage_file):
        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "rb") as f:
                data = pickle.load(f)
        else:
            with open(storage_file, "rb") as f:
                data = pickle.load(f)
        self.vectors = data["vectors"].astype(np.float32)
        self.documents = data["documents"]

    def query(self, query_text, top_k=5):
        query_vector = self.embedding_function([query_text])[0]
        ranked_results = hyper_SVM_ranking_algorithm_sort(self.vectors, query_vector, top_k=top_k, metric=self.similarity_metric)
        return [self.documents[index] for index in ranked_results]
