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


def evaluate_node(state, llm):
    evaluation = evaluate_retrieval(
        llm,
        state["question"],
        state["documents"]
    )

    return {
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
    # Internal knowledge (Multimodal: Text, Tables, Images)
    # -----------------------------

    for result in vector_results:
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        content_type = result.get("content_type") or metadata.get("content_type", "text")
        page = result.get("page") or metadata.get("page", 1)
        content = result.get("content", str(result))

        if content_type == "image":
            image_id = result.get("image_id") or metadata.get("image_id", "img")
            image_path = result.get("image_path") or metadata.get("image_path", "")
            context_parts.append(
                f"""
SOURCE: INTERNAL DOCUMENT [IMAGE] (Page {page}, Image ID: {image_id}, Path: {image_path})
DESCRIPTION:
{content}
"""
            )
        elif content_type == "table":
            table_id = result.get("table_id") or metadata.get("table_id", "tbl")
            context_parts.append(
                f"""
SOURCE: INTERNAL DOCUMENT [TABLE] (Page {page}, Table ID: {table_id})
CONTENT:
{content}
"""
            )
        else:
            chunk_id = result.get("chunk_id") or metadata.get("chunk_id", "chunk")
            context_parts.append(
                f"""
SOURCE: INTERNAL DOCUMENT [TEXT] (Page {page}, Chunk ID: {chunk_id})
CONTENT:
{content}
"""
            )

    # -----------------------------
    # Web knowledge
    # -----------------------------

    for result in web_results:

        context_parts.append(
            f"""
SOURCE: WEB
TITLE: {result.get("title", "")}
URL: {result.get("url", "")}
CONTENT:
{result.get("content", "")}
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
