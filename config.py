from pathlib import Path

BASE_DIR = Path(__file__).parent

UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 5