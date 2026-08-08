from langchain_text_splitters import RecursiveCharacterTextSpliter

def split_documents(documents,chunk_size,chunk_overlap):
    splitter=RecursiveCharacterTextSpliter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        seprators=["\n\n","\n","."," ",""]
    )

    return splitter.split_documents(documents)