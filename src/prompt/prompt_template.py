class PromptTemplate:
    """
    Build prompt for RAG chatbot
    """

    def __init__(self):
        pass

    def build_prompt(self, query, contexts, history):
        processed_contexts = []

        for c in contexts:
            if isinstance(c, dict):
                processed_contexts.append(c["text"])
            else:
                processed_contexts.append(c)

        context_text = "\n\n".join(processed_contexts)

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
The answer must follow the rule below:
1) You must follow the format below, and answer top 3 most relevant places, with each place described in detail.
2) With each place, you must provide the name of the place and a detailed description of the place, and the description must be based on the context provided, and you must not make up any information that is not in the context.
Follow the format below:
 - Place Name: [name of the place] (written in the main language of the question, if available in the context, otherwise write in English)
 - Address: [address of the place if available in the context, otherwise don't add it to the answer]
 - Description: [detailed description of the place based on the context]
 - Avarage Rating: [average rating of the place if available in the context, otherwise don't include this field] (don't need to provide the review count)
 - URL: [URL of the place if available in the context, otherwise don't add it to the answer]

3) Don't include any information that is not in the context
4) If the answer is not clearly, let based on the context and your knowledge revelant to the context of the place to answer.
5) Don't answer the reviews number after rating
6) If the address is not available in the context, don't include it in the answer, and don't make up any information about the address.
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