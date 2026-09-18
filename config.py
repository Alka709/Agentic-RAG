import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

UPLOAD_DIR = BASE_DIR / "uploads"
IMAGE_OUTPUT_DIR = UPLOAD_DIR / "extracted_images"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "llama3.2-vision")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

TOP_K = int(os.getenv("TOP_K", "3"))

TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")