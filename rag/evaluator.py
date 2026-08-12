from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

def create_evaluator_prompts():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a retrieval quality evaluator for a RAG system.

Your job is to determine whether the retrieved context is
good enough to answer the user's question.

Evaluate the context based on:

1. Relevance:
   Does the context discuss the subject of the question?

2. Coverage:
   Does the context contain enough information to answer
   the question?

3. Sufficiency:
   Can the question be answered accurately using only
   the retrieved context?

Return ONLY valid JSON in this exact format:

{{
    "relevance": 0.0,
    "coverage": 0.0,
    "sufficient": true,
    "reason": "short explanation"
}}

Scores must be between 0 and 1.

Do not use information outside the provided context.

Context:
{context}
"""
        ),
        (
            "human",
            "Question: {question}"
        )
    ])

def evaluate_retrieval(llm,question,results):
    context="\n\n".join(
        result["content"]
        for result in results
    )

    prompt=create_evaluator_prompts()

    messages=prompt.invoke({
        "question":question,
        "context":context
    })

    response=llm.invoke(messages)

    parser=JsonOutputParser()

    return parser.parse(response.content)