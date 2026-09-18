from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """
    Splits text documents into chunks while preserving table and image
    documents intact with rich metadata (chunk_id, table_id, image_id).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    final_chunks: List[Document] = []
    text_docs_to_split: List[Document] = []

    for doc in documents:
        content_type = doc.metadata.get("content_type", "text")
        if content_type in ["table", "image"]:
            # Tables and images are already bounded semantic units
            final_chunks.append(doc)
        else:
            text_docs_to_split.append(doc)

    if text_docs_to_split:
        split_texts = splitter.split_documents(text_docs_to_split)
        for idx, chunk in enumerate(split_texts, 1):
            source_path = Path(chunk.metadata.get("source", "doc"))
            page = chunk.metadata.get("page", 1)
            chunk.metadata["content_type"] = "text"
            chunk.metadata["chunk_id"] = f"{source_path.stem}_p{page}_c{idx}"
            final_chunks.append(chunk)

    return final_chunks