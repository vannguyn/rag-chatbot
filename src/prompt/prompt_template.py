class PromptTemplate:
    """
    Build prompt for RAG chatbot
    """

    def __init__(self):
        pass

    def build_prompt(self, query: str, contexts: list[str], history: list = None) -> str:
        """
        Build prompt for LLM

        Args:
            query (str): user question
            contexts (list[str]): retrieved documents
            history (list): previous chat messages

        Returns:
            str: final prompt
        """

        context_text = "\n\n".join(contexts)

        history_text = ""
        if history:
            history_lines = []
            for h in history:
                role = h.get("role", "")
                content = h.get("content", "")
                history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)

        prompt = f"""
You are an intelligent AI assistant.

Use ONLY the provided context to answer the question, when answer the question, you will answer top 3 most relevant places, with each place described
in detail.
If the answer is not in the context, say you don't know.

=====================
Context:
{context_text}
=====================

Chat History:
{history_text}

User Question:
{query}

Answer:
"""

        return prompt.strip()