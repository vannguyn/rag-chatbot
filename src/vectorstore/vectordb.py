import faiss
import json
import numpy as np

class VectorDB:
    def __init__(self, index_path, texts_path):
        self.index = faiss.read_index(index_path)

        with open(texts_path, "r", encoding="utf-8") as f:
            self.texts = json.load(f)

    def search(self, query_vector, top_k=5):
        # chuyển về numpy
        query_vector = np.array([query_vector]).astype("float32")

        # 🔥 normalize để dùng cosine similarity
        faiss.normalize_L2(query_vector)

        # search bằng inner product
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.texts[idx],
                "score": float(scores[0][i])
            })

        return results