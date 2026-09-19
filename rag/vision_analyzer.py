import base64
import logging
from pathlib import Path
from typing import Dict, Optional

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import VISION_LLM_MODEL, GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Cache model availability so we don't spam failed API requests on every image
_MODEL_AVAILABILITY_CACHE: Dict[str, bool] = {}


def encode_image_to_base64(image_path: Path) -> str:
    """Reads an image file and returns its base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def describe_image(
    image_path: Path,
    model_name: Optional[str] = None,
    source: str = "",
    page: int = 0,
    # kept for backward-compat; ignored (Gemini doesn't need a base_url)
    base_url: Optional[str] = None,
) -> str:
    """
    Uses a Gemini vision model to generate a detailed, retrieval-friendly
    textual description of an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    model_name = model_name or VISION_LLM_MODEL

    fallback_description = (
        f"Image extracted from page {page} of {source}. "
        f"Filename: {image_path.name}."
    )

    # If this model previously failed, skip directly to fallback
    if _MODEL_AVAILABILITY_CACHE.get(model_name) is False:
        return fallback_description

    try:
        base64_image = encode_image_to_base64(image_path)

        prompt_text = (
            f"You are an expert technical visual document analyzer for a retrieval-augmented generation (RAG) system.\n"
            f"Analyze this image extracted from page {page} of document '{source}'.\n\n"
            f"Generate a comprehensive, retrieval-friendly technical description containing:\n"
            f"1. Primary Subject: What this image depicts (e.g., system architecture, line chart, flowchart, UI screenshot, diagram).\n"
            f"2. Visible Text & Labels: Transcribe any visible titles, axis labels, legends, callouts, and key values accurately.\n"
            f"3. Key Metrics & Numbers: Specific data points, percentages, benchmark scores, or parameters shown.\n"
            f"4. Structural Relationships: Arrows, sequence flows, inputs/outputs, parent-child nodes, or component connections.\n"
            f"5. Technical Summary: A concise explanation of the main takeaway or technical insight conveyed by this image.\n\n"
            f"Provide ONLY the detailed description without conversational filler."
        )

        vision_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )

        response = vision_llm.invoke([message])
        description = response.content.strip()
        if description:
            _MODEL_AVAILABILITY_CACHE[model_name] = True
            return description

    except Exception as e:
        error_msg = str(e)
        if _MODEL_AVAILABILITY_CACHE.get(model_name) is None:
            if "api_key" in error_msg.lower() or "api key" in error_msg.lower() or "credentials" in error_msg.lower():
                print(
                    f"\n[Notice] Gemini vision model '{model_name}' failed: invalid or missing API key.\n"
                    f"-> Make sure GEMINI_API_KEY is set correctly in your .env file.\n"
                    f"-> Proceeding with structural image metadata for retrieval.\n"
                )
            elif "not found" in error_msg.lower() or "404" in error_msg:
                print(
                    f"\n[Notice] Gemini model '{model_name}' was not found.\n"
                    f"-> Check that LLM_MODEL / VISION_LLM_MODEL in .env is a valid Gemini model name (e.g. gemini-3.6-flash).\n"
                    f"-> Proceeding with structural image metadata for retrieval.\n"
                )
            else:
                print(
                    f"\n[Notice] Vision model '{model_name}' error: {error_msg}.\n"
                    f"-> Proceeding with structural image metadata for retrieval.\n"
                )
        _MODEL_AVAILABILITY_CACHE[model_name] = False

    return fallback_description
