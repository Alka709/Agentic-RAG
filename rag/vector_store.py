from pathlib import Path
from typing import List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.bm25_retriever import BM25Index, create_bm25_index, save_bm25_index, load_bm25_index


def create_vector_store(documents: List[Document], embeddings) -> FAISS:
    """Creates a FAISS vector store from document objects."""
    return FAISS.from_documents(documents, embeddings)


def save_vector_store(vector_store: FAISS, save_path: Path | str) -> None:
    """Saves FAISS index to the specified directory."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))


def load_vector_store(save_path: Path | str, embeddings) -> FAISS:
    """Loads FAISS index from the specified directory."""
    return FAISS.load_local(str(save_path), embeddings, allow_dangerous_deserialization=True)


def save_hybrid_stores(
    vector_store: FAISS,
    bm25_index: BM25Index,
    save_dir: Path | str
) -> None:
    """Saves both FAISS vector store and BM25 index into the target directory."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_vector_store(vector_store, save_dir)
    save_bm25_index(bm25_index, save_dir / "bm25_index.pkl")


def load_hybrid_stores(
    save_dir: Path | str,
    embeddings
) -> Tuple[FAISS, Optional[BM25Index]]:
    """Loads FAISS vector store and BM25 index (if present) from directory."""
    save_dir = Path(save_dir)
    vector_store = load_vector_store(save_dir, embeddings)
    bm25_index = load_bm25_index(save_dir / "bm25_index.pkl")
    return vector_store, bm25_index