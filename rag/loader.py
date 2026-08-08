from pathlib import Path

from langchain_community.document_loaders import(
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

SUPPORTED_LOADERS={
    ".pdf":PyPDFLoader,
    ".docx":Docx2txtLoader,
    ".txt":TextLoader,
    ".md":UnstructuredMarkdownLoader
}

def load_documents(file_path):
    file_path=Path(file_path)
    suffix=file_path.suffix.lower()
    if suffix not in SUPPORTED_LOADERS:
        raise ValueError(f"Unsupported file type:{suffix}")
    
    loader_class=SUPPORTED_LOADERS[suffix]

    loader=loader_class(str(file_path))

    return loader.load()