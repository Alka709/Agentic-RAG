from langchain_core.prompts import ChatPromptTemplate

def create_rag_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer the
question, say that the information is not available in the provided
context.

Do not make up information.

Context:
{context}
"""
        ),
        (
            "human",
            "{question}"
        )        
    ])