import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    IMAGE_OUTPUT_DIR,
    LLM_MODEL,
    TOP_K,
    UPLOAD_DIR,
    VECTOR_DB_DIR,
)
from rag.bm25_retriever import create_bm25_index
from rag.embeddings import get_embedding_model
from rag.generator import create_llm
from rag.loader import load_documents
from rag.prompts import create_rag_prompt
from rag.splitter import split_documents
from rag.vector_store import create_vector_store, save_hybrid_stores
from graphs.graph import build_rag_graph

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_rag_api")

app = FastAPI(
    title="DocuMind AI API",
    description="Backend API for DocuMind AI: Multimodal RAG with Hybrid Retrieval and MCP Tool Calling",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload and image directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount static uploads directory for image and file previews
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Global RAG Graph and embedding cache
_cached_embeddings = None
_cached_llm = None
_cached_prompt = None
_cached_rag_graph = None


def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = get_embedding_model(EMBEDDING_MODEL)
    return _cached_embeddings


def get_rag_graph():
    global _cached_llm, _cached_prompt, _cached_rag_graph
    if _cached_rag_graph is None:
        if _cached_llm is None:
            _cached_llm = create_llm(LLM_MODEL)
        if _cached_prompt is None:
            _cached_prompt = create_rag_prompt()
        _cached_rag_graph = build_rag_graph(_cached_llm, _cached_prompt, TOP_K)
    return _cached_rag_graph


def reset_rag_graph():
    global _cached_rag_graph
    _cached_rag_graph = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K


class IngestRequest(BaseModel):
    url: Optional[str] = None
    file_path: Optional[str] = None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_db_ready": VECTOR_DB_DIR.exists() and (VECTOR_DB_DIR / "index.faiss").exists(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL
    }


def _process_and_index_file(file_path: Path) -> Dict[str, Any]:
    """Loads, splits, and indexes a multimodal document into FAISS & BM25."""
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found at: {file_path}")

    logger.info(f"Loading multimodal document from {file_path}...")
    documents = load_documents(file_path)

    text_docs = [d for d in documents if d.metadata.get("content_type") == "text"]
    table_docs = [d for d in documents if d.metadata.get("content_type") == "table"]
    image_docs = [d for d in documents if d.metadata.get("content_type") == "image"]

    logger.info(
        f"Extracted: {len(text_docs)} text pages, {len(table_docs)} tables, {len(image_docs)} images"
    )

    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content chunks could be extracted from the document.")

    logger.info(f"Indexing {len(chunks)} chunks into FAISS + BM25...")
    embeddings = get_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    bm25_index = create_bm25_index(chunks)

    save_hybrid_stores(vector_store, bm25_index, VECTOR_DB_DIR)
    reset_rag_graph()

    return {
        "status": "success",
        "message": f"Successfully ingested {file_path.name}. {len(chunks)} chunks indexed.",
        "filename": file_path.name,
        "chunks_count": len(chunks),
        "text_pages": len(text_docs),
        "tables_count": len(table_docs),
        "images_count": len(image_docs),
    }


@app.post("/ingest")
async def ingest_document(
    file: Optional[UploadFile] = File(None),
    file_path: Optional[str] = Form(None)
):
    """
    Ingests a document via file upload or local file path.
    Extracts text, tables, and images, builds FAISS + BM25 indexes, and saves hybrid store.
    """
    try:
        if file and file.filename:
            target_path = UPLOAD_DIR / file.filename
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            return _process_and_index_file(target_path)

        elif file_path and file_path.strip():
            target_path = Path(file_path.strip())
            return _process_and_index_file(target_path)

        else:
            raise HTTPException(
                status_code=400,
                detail="Please upload a document file to ingest."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


def _format_error(e: BaseException) -> str:
    if hasattr(e, "exceptions") and e.exceptions:
        return "; ".join([_format_error(sub) for sub in e.exceptions])
    if getattr(e, "__cause__", None):
        return f"{str(e)} (Caused by: {_format_error(e.__cause__)})"
    return str(e)


@app.post("/query")
def query_rag(request: QueryRequest):
    """
    Executes a query against the multimodal hybrid RAG pipeline.
    Returns generated answer, retrieved source documents, and web fallback status.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not VECTOR_DB_DIR.exists() or not (VECTOR_DB_DIR / "index.faiss").exists():
        raise HTTPException(
            status_code=400,
            detail="Vector database is empty. Please ingest a document first."
        )

    try:
        rag_graph = get_rag_graph()
        result = rag_graph.invoke({"question": request.question.strip()})

        retrieved_raw = result.get("documents", [])
        formatted_docs = []

        for doc in retrieved_raw:
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            content_type = doc.get("content_type") or metadata.get("content_type", "text")
            page = doc.get("page") or metadata.get("page", 1)
            source_raw = doc.get("source") or metadata.get("source", "Document")
            source_name = Path(source_raw).name if source_raw else "Document"
            ident = doc.get("identifier") or doc.get("chunk_id") or doc.get("table_id") or doc.get("image_id", "")
            content = doc.get("content", "")
            score = doc.get("rrf_score", doc.get("score", 0.0))
            img_path = doc.get("image_path") or metadata.get("image_path", "")

            # Build accessible web URL for extracted images if present
            image_url = None
            if img_path:
                try:
                    p = Path(img_path)
                    if p.name and (IMAGE_OUTPUT_DIR / p.name).exists():
                        image_url = f"/uploads/extracted_images/{p.name}"
                    elif p.exists():
                        image_url = f"/uploads/{p.name}"
                except Exception:
                    pass

            formatted_docs.append({
                "source": source_name,
                "page": page,
                "content_type": content_type,
                "identifier": ident,
                "content": content,
                "score": round(float(score), 6) if score else 0.0,
                "image_path": img_path,
                "image_url": image_url
            })

        web_results = result.get("web_results", [])
        web_fallback_triggered = bool(web_results)

        raw_answer = result.get("answer", "No answer could be synthesized.")
        if isinstance(raw_answer, list):
            answer_text = "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in raw_answer
            )
        elif isinstance(raw_answer, dict):
            answer_text = raw_answer.get("text", str(raw_answer))
        else:
            answer_text = str(raw_answer)

        return {
            "question": request.question,
            "answer": answer_text,
            "documents": formatted_docs,
            "web_results": web_results,
            "web_fallback": web_fallback_triggered,
            "retrieval_mode": "Hybrid RRF",
            "evaluation": result.get("evaluation", {})
        }

    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query error: {_format_error(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
