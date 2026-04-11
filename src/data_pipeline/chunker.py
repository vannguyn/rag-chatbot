from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import json
class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
        )
    
    def split(self, text: str):
        docs = [Document(page_content=text)]
        return self.splitter.split_documents(docs)
    
    def save_to_json(self, chunks, path):
        data = []

        for doc in chunks:
            data.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


