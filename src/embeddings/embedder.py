import json
import os
import faiss
import numpy as np

from src.embeddings.embedding_model import EmbeddingModel


class Embedder:
    def __init__(self):
        """Initialize Embedder with hard paths"""

        # ===== HARD PATH =====
        self.data_path = "data"
        self.vector_db_path = "embeddings/vector_db"
        self.embedding_model_path = "embeddings/paraphrase-multilingual-MiniLM-L12-v2"

        # tạo thư mục nếu chưa có
        # os.makedirs(self.vector_db_path, exist_ok=True)

        # load embedding model
        print("Loading embedding model...")
        self.embedding_model = EmbeddingModel()
        print("Embedding model loaded!")

    def load_chunks(self, path=None):
        """Load chunks from JSON"""

        if path is None:
            path = os.path.join(self.data_path, "chunks.json")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Chunks file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = []
        for item in data:
            if isinstance(item, dict):
                texts.append(item.get("content", ""))
            else:
                texts.append(item)

        return texts

    def build_vector_db(self, chunks_path=None):
        """Create FAISS index from chunks"""

        texts = self.load_chunks(chunks_path)

        if len(texts) == 0:
            raise ValueError("No text chunks found!")

        print(f"Loaded {len(texts)} chunks")

        # embedding
        embeddings = self.embedding_model.embed_docs(texts)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]

        # tạo FAISS index
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # lưu index
        index_path = os.path.join(self.vector_db_path, "faiss.index")
        faiss.write_index(index, index_path)

        # lưu text
        texts_path = os.path.join(self.vector_db_path, "texts.json")
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=4)

        print(f"Vector DB created at: {self.vector_db_path}")