from pathlib import Path
from config import(
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.embeddings import get_embedding_model
from rag.vector_store import create_vector_store
from rag.retriever import retrieve_documents
from rag.prompts import create_rag_prompt
from rag.generator import create_llm,generate_answer
from rag.evaluator import evaluate_retrieval

def main():
    file_path = input("Enter document path: ").strip()

    path = Path(file_path)
    
    documents=load_documents(path)

    chunks=split_documents(
        documents,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    embeddings=get_embedding_model(EMBEDDING_MODEL)

    vector_store=create_vector_store(chunks,embeddings)

    llm=create_llm(LLM_MODEL)

    prompt=create_rag_prompt()

    question=input("\nAsk a question: ")

    results=retrieve_documents(vector_store,question,TOP_K)

    evaluation = evaluate_retrieval(
    llm,
    question,
    results
    )

    print("\nRetrieval Evaluation:")
    print(evaluation)

    context="\n\n".join(result["content"] for result in results)

    answer=generate_answer(
        llm,
        prompt,
        question,
        context)

    print("\nRetrieved Documents:")

    for i, result in enumerate(results, start=1):

        print(
            f"\n{i}. "
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Source: "
            f"{result['metadata'].get('source', 'unknown')}"
        )

    print("\nAnswer:")
    print(answer)

if __name__=="__main__":
    main()