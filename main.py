from src.data_pipeline.loader import Loader
from src.data_pipeline.cleaner import Cleaner
from src.data_pipeline.chunker import TextSplitter
from src.embeddings.embedder import Embedder
def main():
    # Load data
    input_folder_path = "data/raw"
    output_folder_path = "data/processed"
    loader = Loader(input_folder_path)
    raw_data = loader.load_documents()
    markdown = ""
    for item in raw_data:
        markdown += Cleaner.json_to_markdown(item)
    
    text_chunker = TextSplitter(chunk_size=1200, chunk_overlap=200)
    all_chunks = text_chunker.split(markdown)

    text_chunker.save_to_json(all_chunks, f"{output_folder_path}/chunks.json")
def embedding():
    embedder = Embedder()
    chunks_path = "data/processed/chunks.json"
    embedder.build_vector_db(chunks_path)

if __name__ == "__main__":
    embedding()