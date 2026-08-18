from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from config import (EMBEDDING_MODEL, VECTOR_DB_DIR)

from rag.embeddings import get_embedding_model
from rag.vector_store import load_vector_store
from rag.retriever import retrieve_documents
from rag.web_search import (
    create_web_search_client,
    search_web
)

mcp = FastMCP("Agentic RAG Tools")

embeddings = get_embedding_model(EMBEDDING_MODEL)

# Load an existing saved FAISS index built by app.py
# Run app.py first to create and save the vector store
if not VECTOR_DB_DIR.exists():
    raise RuntimeError(
        f"Vector store not found at '{VECTOR_DB_DIR}'. "
        "Please run app.py first to load a document and build the index."
    )

vector_store = load_vector_store(VECTOR_DB_DIR, embeddings)

web_client = create_web_search_client()


# VECTOR SEARCH TOOL
@mcp.tool()
def vector_search(query: str, top_k: int = 5) -> list[dict]:
    """Search the vector store for documents related to a query."""
    return retrieve_documents(vector_store, query, top_k)


# WEB SEARCH TOOL
@mcp.tool()
def web_search(
    query: str,
    max_results: int = 5
) -> list[dict]:
    """
    Search the web for information related to a query.
    """
    return search_web(
        web_client,
        query,
        max_results
    )


if __name__ == "__main__":
    mcp.run()