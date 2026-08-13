from langchain_community.vectorstores import FAISS

def retrieve_documents(vector_store,query,top_k):
    results=vector_store.similarity_search_with_score(query,k=top_k)

    retrieved=[]

    for document,score in results:
        retrieved.append({
            "content": document.page_content,
            "score":float(score),
            "metadata":document.metadata
        })

    return retrieved