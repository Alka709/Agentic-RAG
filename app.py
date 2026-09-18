from pathlib import Path

from config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    VECTOR_DB_DIR
)

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.embeddings import get_embedding_model
from rag.bm25_retriever import create_bm25_index
from rag.vector_store import create_vector_store, save_hybrid_stores
from rag.prompts import create_rag_prompt
from rag.generator import create_llm

from graphs.graph import build_rag_graph


def main():

    file_path = Path(
        input("Enter document path: ").strip()
    )

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"\nProcessing multimodal document: {file_path.name}...")
    documents = load_documents(file_path)

    text_docs = [d for d in documents if d.metadata.get("content_type") == "text"]
    table_docs = [d for d in documents if d.metadata.get("content_type") == "table"]
    image_docs = [d for d in documents if d.metadata.get("content_type") == "image"]

    print(
        f"Extracted elements:\n"
        f"  - Text pages: {len(text_docs)}\n"
        f"  - Structured tables: {len(table_docs)}\n"
        f"  - Processed images/charts: {len(image_docs)}"
    )

    chunks = split_documents(
        documents,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    print(
        f"Total indexed chunks: {len(chunks)}"
    )

    print("Building FAISS vector index & BM25 sparse index...")
    embeddings = get_embedding_model(
        EMBEDDING_MODEL
    )

    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    bm25_index = create_bm25_index(
        chunks
    )

    save_hybrid_stores(
        vector_store,
        bm25_index,
        VECTOR_DB_DIR
    )
    print("Hybrid indexes (FAISS + BM25) successfully created and saved.")

    llm = create_llm(
        LLM_MODEL
    )

    prompt = create_rag_prompt()

    rag_graph = build_rag_graph(
        llm,
        prompt,
        TOP_K,
    )

    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        )

        if question.lower() == "exit":
            break

        result = rag_graph.invoke({
            "question": question
        })

        retrieved_docs = result.get("documents", [])
        if retrieved_docs:
            print("\nRetrieved Evidence:")
            for i, doc in enumerate(retrieved_docs, 1):
                c_type = doc.get("content_type", "text").upper()
                page = doc.get("page", 1)
                img_path = doc.get("image_path", "")
                ident = doc.get("identifier", "")
                score = doc.get("rrf_score", doc.get("score", 0.0))
                extra = f" | File: {img_path}" if img_path else ""
                print(f"  [{i}] Type: {c_type} (Page {page}{extra}) | ID: {ident} | RRF Score: {score:.6f}")

        evaluation = result.get(
            "evaluation",
            {}
        )

        print("\nRetrieval Evaluation:")
        print(evaluation)

        web_results = result.get(
            "web_results",
            []
        )

        if web_results:

            print(
                f"\nWeb search used: "
                f"{len(web_results)} results"
            )

            for item in web_results:
                print(
                    f"- {item['title']}"
                )

                print(
                    f"  {item['url']}"
                )

        else:
            print(
                "\nWeb search not required."
            )

        print("\nAnswer:")

        print(
            result.get(
                "answer",
                "Unable to generate an answer."
            )
        )


if __name__ == "__main__":
    main()