from typing import Any, Dict, List
from langchain_community.vectorstores import FAISS


def retrieve_documents(
    vector_store: FAISS,
    query: str,
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Retrieves the most relevant multimodal chunks (text, table, or image description)
    from the FAISS index with similarity scores and metadata.
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)

    retrieved = []

    for document, score in results:
        metadata = document.metadata or {}
        content_type = metadata.get("content_type", "text")

        item = {
            "content": document.page_content,
            "score": float(score),
            "content_type": content_type,
            "source": metadata.get("source", ""),
            "page": metadata.get("page", 1),
            "metadata": metadata,
        }

        # Add specific identifiers and paths
        if content_type == "image":
            item["image_id"] = metadata.get("image_id", "")
            item["image_path"] = metadata.get("image_path", "")
            item["description"] = metadata.get("description", document.page_content)
        elif content_type == "table":
            item["table_id"] = metadata.get("table_id", "")
        else:
            item["chunk_id"] = metadata.get("chunk_id", "")

        retrieved.append(item)

    return retrieved