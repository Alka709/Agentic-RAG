import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def get_deterministic_identifier(
    content: str,
    metadata: Dict[str, Any]
) -> str:
    """
    Extracts existing chunk/table/image identifiers or builds a deterministic
    content-based ID to ensure both dense and sparse retrieval reference the exact same item.
    """
    if not metadata:
        metadata = {}

    # Check for existing identifiers
    if metadata.get("chunk_id"):
        return str(metadata["chunk_id"])
    if metadata.get("table_id"):
        return str(metadata["table_id"])
    if metadata.get("image_id"):
        return str(metadata["image_id"])

    # Deterministic fallback ID based on content, source, and page
    source = metadata.get("source", "")
    page = metadata.get("page", 1)
    normalized_content = " ".join(content.strip().split())
    hash_key = f"{source}__p{page}__{normalized_content}".encode("utf-8")
    return f"doc_{hashlib.sha256(hash_key).hexdigest()[:16]}"


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text into lowercase alphanumeric tokens for BM25 indexing and querying.
    """
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """
    Modular BM25 retriever index supporting text chunks, structured tables,
    and image descriptions with full metadata preservation.
    """

    def __init__(self, documents: Optional[List[Document]] = None):
        self.documents: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if documents:
            self.build_index(documents)

    def build_index(self, documents: List[Document]) -> None:
        """
        Builds the BM25 index over the provided documents.
        """
        self.documents = []
        self.corpus_tokens = []

        for doc in documents:
            content = doc.page_content or ""
            metadata = doc.metadata.copy() if doc.metadata else {}
            identifier = get_deterministic_identifier(content, metadata)
            content_type = metadata.get("content_type", "text")

            # Ensure identifier is stored in metadata
            if content_type == "image":
                metadata.setdefault("image_id", identifier)
            elif content_type == "table":
                metadata.setdefault("table_id", identifier)
            else:
                metadata.setdefault("chunk_id", identifier)

            tokens = tokenize_text(content)
            self.corpus_tokens.append(tokens)

            self.documents.append({
                "content": content,
                "metadata": metadata,
                "identifier": identifier,
                "content_type": content_type,
                "source": metadata.get("source", ""),
                "page": metadata.get("page", 1),
            })

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes BM25 sparse search and returns top-k ranked documents with scores.
        """
        if not self.bm25 or not self.documents:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        scored_indices = sorted(
            enumerate(doc_scores),
            key=lambda x: x[1],
            reverse=True
        )

        top_indices = scored_indices[:top_k]
        results = []

        for rank, (idx, score) in enumerate(top_indices, 1):
            doc_entry = self.documents[idx]
            metadata = doc_entry["metadata"]
            content_type = doc_entry["content_type"]

            item = {
                "content": doc_entry["content"],
                "score": float(score),
                "content_type": content_type,
                "source": doc_entry["source"],
                "page": doc_entry["page"],
                "identifier": doc_entry["identifier"],
                "metadata": metadata,
                "bm25_rank": rank,
            }

            if content_type == "image":
                item["image_id"] = metadata.get("image_id", doc_entry["identifier"])
                item["image_path"] = metadata.get("image_path", "")
                item["description"] = metadata.get("description", doc_entry["content"])
            elif content_type == "table":
                item["table_id"] = metadata.get("table_id", doc_entry["identifier"])
            else:
                item["chunk_id"] = metadata.get("chunk_id", doc_entry["identifier"])

            results.append(item)

        return results

    def save(self, save_path: Path) -> None:
        """
        Saves BM25 index and documents to disk.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "corpus_tokens": self.corpus_tokens,
                "bm25": self.bm25,
            }, f)

    @classmethod
    def load(cls, load_path: Path) -> "BM25Index":
        """
        Loads BM25 index and documents from disk.
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {load_path}")

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        index = cls()
        index.documents = data.get("documents", [])
        index.corpus_tokens = data.get("corpus_tokens", [])
        index.bm25 = data.get("bm25")
        return index


def create_bm25_index(documents: List[Document]) -> BM25Index:
    """Helper factory function to create BM25Index."""
    return BM25Index(documents)


def save_bm25_index(index: BM25Index, save_path: Path) -> None:
    """Helper function to save BM25Index."""
    index.save(save_path)


def load_bm25_index(save_path: Path) -> Optional[BM25Index]:
    """Helper function to load BM25Index safely."""
    save_path = Path(save_path)
    if not save_path.exists():
        return None
    return BM25Index.load(save_path)
