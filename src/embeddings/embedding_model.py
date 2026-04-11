from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """Initialize embedding model"""
        self.model = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def embed_texts(self, texts: list[str]):
        """Convert list of texts -> embeddings"""
        return self.model.embed_documents(texts)

    def embed_query(self, query: str):
        """Convert query -> embedding"""
        return self.model.embed_query(query)

