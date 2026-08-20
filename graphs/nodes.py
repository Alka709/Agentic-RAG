from mcp_client.client import invoke_mcp_tool
from rag.evaluator import evaluate_retrieval
from rag.generator import generate_answer

def retrieve_node(state, top_k):

    documents = invoke_mcp_tool(
        "vector_search",
        {
            "query": state["question"],
            "top_k": top_k
        }
    )

    return {
        "documents": documents
    }

def evaluate_node(state,llm):
    evaluation=evaluate_retrieval(
        llm,
        state["question"],
        state["documents"]
    )

    return{
        "evaluation": evaluation
    }

def web_search_node(state, max_results=5):

    web_results = invoke_mcp_tool(
        "web_search",
        {
            "query": state["question"],
            "max_results": max_results
        }
    )

    return {
        "web_results": web_results
    }

def answer_node(state, llm, prompt):

    question = state["question"]

    vector_results = state.get("documents", [])

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

