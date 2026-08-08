from langchain_community.vectorstores import FAISS

def retrieve_documents(vectore_store,query,top_k):
    results=vectore_store.similarity_search_with_score(query,k=top_k)

    return results