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

Evaluate whether the provided context is sufficient to answer
the user's question.

Consider:

1. Relevance:
   Does the context discuss the subject of the question?

2. Coverage:
   Does the context contain enough information to answer the question?

3. Sufficiency:
   Can the question be answered accurately using only the context?

Give your evaluation based ONLY on the provided context.

Do not use outside knowledge.

Question:
{question}

Context:
{context}
"""
        )
    ])


def evaluate_retrieval(llm, question, results):

    context = "\n\n".join(
        result["content"]
        for result in results
    )

    prompt = create_evaluator_prompt()

    structured_llm = llm.with_structured_output(
        RetrievalEvaluation
    )

    chain = prompt | structured_llm

    evaluation = chain.invoke({
        "question": question,
        "context": context
    })

    return evaluation.model_dump()