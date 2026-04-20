from transformers import AutoTokenizer, AutoModel
import torch


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 👉 CLS pooling (chuẩn của BGE)
        embeddings = outputs.last_hidden_state[:, 0]

        # normalize (cosine similarity)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def embed_docs(self, docs):
        # docs không cần prefix
        return self._encode(docs)

    def embed_query(self, query):
        # 👉 BGE khuyến nghị thêm instruction
        instruction = "Represent this sentence for searching relevant passages: "
        query = instruction + query

        return self._encode(query)[0]