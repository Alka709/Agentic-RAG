from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        from mcp.server.mcpserver import MCPServer as FastMCP

from config import (EMBEDDING_MODEL, VECTOR_DB_DIR)

from rag.embeddings import get_embedding_model
from rag.vector_store import load_hybrid_stores
from rag.retriever import retrieve_documents
from rag.web_search import (
    create_web_search_client,
    search_web
)

mcp = FastMCP("Agentic RAG Tools")

_embeddings = None
_web_client = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embedding_model(EMBEDDING_MODEL)
    return _embeddings


def get_web_client():
    global _web_client
    if _web_client is None:
        try:
            _web_client = create_web_search_client()
        except Exception:
            _web_client = None
    return _web_client


# VECTOR / HYBRID SEARCH TOOL
@mcp.tool()
def vector_search(query: str, top_k: int = 5) -> list[dict]:
    """Search documents related to a query using Hybrid Retrieval (Dense FAISS + Sparse BM25 fused with RRF)."""
    if not VECTOR_DB_DIR.exists() or not (VECTOR_DB_DIR / "index.faiss").exists():
        return []

    embeddings = get_embeddings()
    vector_store, bm25_index = load_hybrid_stores(VECTOR_DB_DIR, embeddings)

    return retrieve_documents(
        vector_store=vector_store,
        query=query,
        top_k=top_k,
        bm25_index=bm25_index
    )


# WEB SEARCH TOOL
@mcp.tool()
def web_search(
    query: str,
    max_results: int = 5
) -> list[dict]:
    """
    Search the web for information related to a query.
    """
    client = get_web_client()
    if client is None:
        return []

    return search_web(
        client,
        query,
        max_results
    )


if __name__ == "__main__":
    mcp.run()