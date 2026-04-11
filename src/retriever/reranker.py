class Reranker:
    def rerank(self, query, docs):
        scored = [(doc, len(doc)) for doc in docs]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored[:3]]