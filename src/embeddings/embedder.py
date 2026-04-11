import json
import os
import faiss
import numpy as np

from .embedding_model import EmbeddingModel


class Embedder:
    def __init__(self, vector_db_path="embeddings/vector_db"):
        self.vector_db_path = vector_db_path
        self.embedding_model = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        os.makedirs(self.vector_db_path, exist_ok=True)

    def load_chunks(self, path):
        """ Load chunks from JSON """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = []

        for item in data:
            if isinstance(item, dict):
                texts.append(item["content"])
            else:
                texts.append(item)

        return texts

    def build_vector_db(self, chunks_path):
        """ Create FAISS index from chunks  """

        texts = self.load_chunks(chunks_path)

        print(f"Loaded {len(texts)} chunks")

        embeddings = self.embedding_model.embed_texts(texts)

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)

        index.add(embeddings)

        faiss.write_index(index, f"{self.vector_db_path}/faiss.index")

        with open(f"{self.vector_db_path}/texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=4)

        print("Vector DB created successfully")