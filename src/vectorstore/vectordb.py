import faiss
import json
import numpy as np

class VectorDB:
    def __init__(self, index_path, texts_path):
        self.index = faiss.read_index(index_path)

        with open(texts_path, "r", encoding="utf-8") as f:
            self.texts = json.load(f)

    def search(self, query_vector, top_k=5):
        query_vector = np.array([query_vector]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = [self.texts[i] for i in indices[0]]

        return results