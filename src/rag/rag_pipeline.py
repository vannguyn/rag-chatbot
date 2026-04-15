from src.vectorstore.vectordb import VectorDB
from src.retriever.retriever import Retriever
from src.retriever.reranker import Reranker
from src.prompt.prompt_template import PromptTemplate
from src.llm.groq_client import GroqClient
from src.embeddings.embedding_model import EmbeddingModel


class RAGPipeline:
    """
    End-to-end RAG pipeline
    """

    def __init__(
        self,
        index_path="embeddings/vector_db/faiss.index",
        texts_path="embeddings/vector_db/texts.json"
    ):

        # Embedding model
        self.embedding_model = EmbeddingModel()

        # Vector database
        self.vectordb = VectorDB(index_path, texts_path)

        # Retriever
        self.retriever = Retriever(self.embedding_model, self.vectordb)

        # Reranker
        self.reranker = Reranker()

        # Prompt builder
        self.prompt_builder = PromptTemplate()

        # LLM client
        self.llm = GroqClient()

        # Chat history
        self.history = []

    def ask(self, query: str):

        # 1 retrieve documents
        docs = self.retriever.retrieve(query)

        # 2 rerank documents
        docs = self.reranker.rerank(query, docs)

        # 3 build prompt
        prompt = self.prompt_builder.build_prompt(
            query=query,
            contexts=docs,
            history=self.history
        )

        # 4 generate answer
        answer = self.llm.generate(prompt)

        # 5 save history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})

        return answer