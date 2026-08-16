from langchain_core.prompts import ChatPromptTemplate


def create_rag_prompt():

    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a helpful AI assistant.

Answer the user's question using the provided evidence.

The evidence may come from:

1. INTERNAL DOCUMENT
2. WEB

Prefer information from the INTERNAL DOCUMENT when it directly
answers the question.

Use WEB information when the internal document does not contain
enough information.

Do not invent facts.

If the available evidence is insufficient, clearly say so.

When using information from the web, mention the relevant source
or URL when appropriate.

Evidence:

{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ])