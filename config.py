import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

UPLOAD_DIR = BASE_DIR / "uploads"
IMAGE_OUTPUT_DIR = UPLOAD_DIR / "extracted_images"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "gemini-3.6-flash")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

TOP_K = int(os.getenv("TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
DENSE_CANDIDATES = int(os.getenv("DENSE_CANDIDATES", "5"))
BM25_CANDIDATES = int(os.getenv("BM25_CANDIDATES", "5"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
