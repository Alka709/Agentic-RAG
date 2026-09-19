from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class RetrievalEvaluation(BaseModel):

    relevance: float = Field(
        description="How relevant the retrieved context is to the question, from 0 to 1."
    )

    coverage: float = Field(
        description="How completely the context covers the information needed to answer the question, from 0 to 1."
    )

    sufficient: bool = Field(
        description="Whether the retrieved context is sufficient to answer the question accurately."
    )

    reason: str = Field(
        description="Brief explanation for the evaluation."
    )


def create_evaluator_prompt():

    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a retrieval quality evaluator for a RAG system.

Evaluate whether the provided context is sufficient to answer the user's question.

Consider:
1. Relevance: Does the context discuss the subject of the question?
2. Coverage: Does the context contain enough information to answer the question?
3. Sufficiency: Can the question be answered accurately using only the context?

Give your evaluation based ONLY on the provided context. Do not use outside knowledge."""
        ),
        (
            "human",
            """Question:
{question}

Context:
{context}"""
        )
    ])


def evaluate_retrieval(llm, question, results):

    if not results:
        return {
            "relevance": 0.0,
            "coverage": 0.0,
            "sufficient": False,
            "reason": "No relevant context was retrieved from internal documents."
        }

    context = "\n\n".join(
        result["content"] if isinstance(result, dict) else str(result)
        for result in results
    ).strip()

    if not context:
        return {
            "relevance": 0.0,
            "coverage": 0.0,
            "sufficient": False,
            "reason": "Retrieved context was empty."
        }

    prompt = create_evaluator_prompt()

    structured_llm = llm.with_structured_output(
        RetrievalEvaluation
    )

    chain = prompt | structured_llm

    evaluation = chain.invoke({
        "question": question,
        "context": context
    })

    if isinstance(evaluation, dict):
        result = evaluation
    elif hasattr(evaluation, "model_dump"):
        result = evaluation.model_dump()
    else:
        result = {
            "relevance": getattr(evaluation, "relevance", 0.0),
            "coverage": getattr(evaluation, "coverage", 0.0),
            "sufficient": getattr(evaluation, "sufficient", False),
            "reason": getattr(evaluation, "reason", "")
        }

    result["sufficient"] = (
        result.get("relevance", 0.0) >= 0.5
        and result.get("coverage", 0.0) >= 0.5
    )

    return result