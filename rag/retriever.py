import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS

from config import BM25_CANDIDATES, DENSE_CANDIDATES, RRF_K
from rag.bm25_retriever import BM25Index, get_deterministic_identifier

logger = logging.getLogger(__name__)


def dense_search(
    vector_store: FAISS,
    query: str,
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Retrieves candidates using FAISS dense vector similarity.
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)
    retrieved = []

    for rank, (document, score) in enumerate(results, 1):
        content = document.page_content or ""
        metadata = document.metadata.copy() if document.metadata else {}
        content_type = metadata.get("content_type", "text")
        identifier = get_deterministic_identifier(content, metadata)

        item = {
            "content": content,
            "score": float(score),
            "content_type": content_type,
            "source": metadata.get("source", ""),
            "page": metadata.get("page", 1),
            "identifier": identifier,
            "metadata": metadata,
            "dense_rank": rank,
        }

        if content_type == "image":
            item["image_id"] = metadata.get("image_id", identifier)
            item["image_path"] = metadata.get("image_path", "")
            item["description"] = metadata.get("description", content)
        elif content_type == "table":
            item["table_id"] = metadata.get("table_id", identifier)
        else:
            item["chunk_id"] = metadata.get("chunk_id", identifier)

        retrieved.append(item)

    return retrieved


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int,
    rrf_k: int = RRF_K
) -> List[Dict[str, Any]]:
    """
    Combines dense and sparse search rankings using Reciprocal Rank Fusion (RRF):
        RRF_score(d) = Σ 1 / (k + rank(d))
    where rank starts at 1. Documents found in both result sets accumulate scores.
    """
    doc_map: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = {}

    # Accumulate Dense RRF scores
    for rank, doc in enumerate(dense_results, 1):
        doc_id = doc["identifier"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    # Accumulate BM25 RRF scores
    for rank, doc in enumerate(bm25_results, 1):
        doc_id = doc["identifier"]
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    # Sort documents by accumulated RRF score in descending order
    sorted_doc_ids = sorted(
        rrf_scores.keys(),
        key=lambda d_id: rrf_scores[d_id],
        reverse=True
    )

    final_results: List[Dict[str, Any]] = []
    for rank, doc_id in enumerate(sorted_doc_ids[:top_k], 1):
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = rrf_scores[doc_id]
        doc["final_rank"] = rank
        final_results.append(doc)

    return final_results


def log_retrieval_summary(
    mode: str,
    dense_count: int,
    bm25_count: int,
    unique_count: int,
    final_candidates: List[Dict[str, Any]]
) -> None:
    """
    Outputs concise retrieval summary and structured ranking debug logs.
    """
    summary_lines = [
        f"Retrieval mode: {mode}",
        f"Dense candidates: {dense_count}",
        f"BM25 candidates: {bm25_count}",
        f"Unique candidates: {unique_count}",
        f"Final candidates: {len(final_candidates)}",
    ]
    logger.info(" | ".join(summary_lines))

    for doc in final_candidates:
        rank = doc.get("final_rank", 1)
        content_type = doc.get("content_type", "text")
        source = Path(doc.get("source", "")).name if doc.get("source") else "unknown"
        identifier = doc.get("identifier", "unknown")
        rrf_score = doc.get("rrf_score", doc.get("score", 0.0))
        logger.debug(f"  rank: {rank} | content_type: {content_type} | source: {source} | identifier: {identifier} | rrf_score: {rrf_score:.6f}")


def retrieve_documents(
    vector_store: FAISS,
    query: str,
    top_k: int = 5,
    bm25_index: Optional[BM25Index] = None,
    dense_candidates_k: Optional[int] = None,
    bm25_candidates_k: Optional[int] = None,
    rrf_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves documents using Hybrid Retrieval (Dense FAISS + Sparse BM25)
    fused with Reciprocal Rank Fusion (RRF). Gracefully falls back to dense-only
    if BM25 index is not provided.
    """
    num_dense = dense_candidates_k or max(top_k, DENSE_CANDIDATES)
    num_bm25 = bm25_candidates_k or max(top_k, BM25_CANDIDATES)
    k_const = rrf_k or RRF_K

    # 1. Dense retrieval (FAISS)
    dense_results = dense_search(vector_store, query, top_k=num_dense)

    # 2. If no BM25 index is present, return dense results with fallback logging
    if bm25_index is None:
        final_results = dense_results[:top_k]
        for rank, doc in enumerate(final_results, 1):
            doc["final_rank"] = rank
            doc["rrf_score"] = 1.0 / (k_const + rank)

        log_retrieval_summary(
            mode="dense",
            dense_count=len(dense_results),
            bm25_count=0,
            unique_count=len(dense_results),
            final_candidates=final_results
        )
        return final_results

    # 3. Sparse retrieval (BM25)
    bm25_results = bm25_index.search(query, top_k=num_bm25)

    # 4. Calculate unique candidates count
    unique_ids = set(d["identifier"] for d in dense_results) | set(d["identifier"] for d in bm25_results)

    # 5. Reciprocal Rank Fusion
    final_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        top_k=top_k,
        rrf_k=k_const
    )

    # 6. Concise Logging
    log_retrieval_summary(
        mode="hybrid",
        dense_count=len(dense_results),
        bm25_count=len(bm25_results),
        unique_count=len(unique_ids),
        final_candidates=final_results
    )

    return final_results