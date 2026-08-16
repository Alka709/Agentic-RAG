import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 3

TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")