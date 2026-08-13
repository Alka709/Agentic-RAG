from pathlib import Path

from config import (
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
from rag.prompts import create_rag_prompt
from rag.generator import create_llm

from graphs.graph import build_rag_graph

def main():
    file_path=Path(input("Enter document path: ").strip())

    documents=load_documents(file_path)

    print(f"Loaded {len(documents)} document pages")

    chunks=split_documents(documents,CHUNK_SIZE,CHUNK_OVERLAP)

    print(f"Created {len(chunks)} chunks")

    embeddings=get_embedding_model(EMBEDDING_MODEL)

    vector_store=create_vector_store(chunks,embeddings)

    llm=create_llm(LLM_MODEL)

    prompt=create_rag_prompt()

    rag_graph=build_rag_graph(vector_store,llm,prompt,TOP_K)

    while True:
        question=input("\nAsk a question (or type 'exit'): ")

        if question.lower() == 'exit':
            break
        
        result=rag_graph.invoke({
            "question": question
        })

        evaluation=result.get("evaluation",{})

        print("\nRetrieval Evaluation:")
        print(evaluation)

        if result.get("answer"):
            print("\nAnswer:")
            print(result["answer"])
        else:
            print(
                "\nThe retrieved context was not sufficient. Web search will be used in the next phase."
            )

if __name__== "__main__":
    main()