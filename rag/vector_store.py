from langchain_community.vectorstores import FAISS

def create_vector_store(documents,embeddings):
    return FAISS.from_documents(documents,embeddings)

def save_vector_store(vector_store,save_path):
    vector.store.save_local(str(save_path))

def load_vector_store(vector_store,save_path):
    return FAISS.load_local(str(save_path),embeddings,allow_dangerous_deserialization=True)