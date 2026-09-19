import os
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_API_KEY, EMBEDDING_MODEL


def get_embedding_model(
    model_name: Optional[str] = None,
    google_api_key: Optional[str] = None
) -> GoogleGenerativeAIEmbeddings:
    """
    Returns a Google Generative AI Embeddings model instance.
    Uses Gemini API embeddings to keep runtime memory lean (< 100MB) without local PyTorch models.
    """
    model = model_name or EMBEDDING_MODEL or "models/gemini-embedding-001"

    # Normalize model name / sanitize old sentence-transformers references
    if "sentence-transformers" in model or not model.strip():
        model = "models/gemini-embedding-001"
    elif not model.startswith("models/") and "/" not in model:
        model = f"models/{model}"

    api_key = google_api_key or GEMINI_API_KEY
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key
    )