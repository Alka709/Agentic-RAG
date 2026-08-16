from rag.retriever import retrieve_documents
from rag.evaluator import evaluate_retrieval
from rag.generator import generate_answer
from rag.web_search import search_web

def retrieve_node(state,vector_store,top_k):
    question = state["question"]
    
    results=retrieve_documents(
        vector_store,
        question,
        top_k
    )

    return{
        "retrieved_documents":results
    }

def evaluate_node(state,llm):
    question=state["question"]
    results=state["retrieved_documents"]

    evaluation=evaluate_retrieval(
        llm,
        question,results
    )

    return{
        "evaluation": evaluation
    }

def answer_node(state, llm, prompt):

    question = state["question"]

    vector_results = state.get(
        "retrieved_documents",
        []
    )

    web_results = state.get(
        "web_results",
        []
    )

    context_parts = []

    # -----------------------------
    # Internal knowledge
    # -----------------------------

    for result in vector_results:

        context_parts.append(
            f"""
SOURCE: INTERNAL DOCUMENT
CONTENT:
{result["content"]}
"""
        )

    # -----------------------------
    # Web knowledge
    # -----------------------------

    for result in web_results:

        context_parts.append(
            f"""
SOURCE: WEB
TITLE: {result["title"]}
URL: {result["url"]}
CONTENT:
{result["content"]}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    answer = generate_answer(
        llm,
        prompt,
        question,
        context
    )

    return {
        "context": context,
        "answer": answer
    }

def web_search_node(state,web_client):
    question=state["question"]
    results=search_web(
        web_client,
        question
    )

    return{
        "web_results":results
    }