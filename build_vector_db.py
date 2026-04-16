import os
import json

from src.data_pipeline.loader import Loader
from src.data_pipeline.cleaner import Cleaner
from src.data_pipeline.chunker import TextSplitter
from src.embeddings.embedder import Embedder


RAW_DATA_FOLDER = "data/raw"
CHUNKS_PATH = "data/processed/chunks.json"


class VectorDBPipeline:

    def __init__(self):

        self.loader = Loader(RAW_DATA_FOLDER)
        self.splitter = TextSplitter()
        self.embedder = Embedder()

    # -----------------------------
    # Step 1: Load + Convert JSON
    # -----------------------------

    def load_and_prepare_documents(self):

        print("Loading JSON documents...")

        documents = self.loader.load_documents()

        texts = []

        for item in documents:

            markdown = Cleaner.json_to_markdown(item)

            texts.append(markdown)

        print(f"Total documents: {len(texts)}")

        return texts

    # -----------------------------
    # Step 2: Chunk documents
    # -----------------------------

    def split_documents(self, texts):

        print("Splitting documents into chunks...")

        chunks = []

        for text in texts:

            docs = self.splitter.split(text)

            for doc in docs:

                chunks.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        print(f"Total chunks: {len(chunks)}")

        return chunks

    # -----------------------------
    # Step 3: Save chunks
    # -----------------------------

    def save_chunks(self, chunks):

        os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)

        print(f"Chunks saved → {CHUNKS_PATH}")

    # -----------------------------
    # Pipeline
    # -----------------------------

    def run(self):

        texts = self.load_and_prepare_documents()

        chunks = self.split_documents(texts)

        self.save_chunks(chunks)

        print("Building vector database...")

        self.embedder.build_vector_db(CHUNKS_PATH)


def main():

    pipeline = VectorDBPipeline()

    pipeline.run()


if __name__ == "__main__":
    main()