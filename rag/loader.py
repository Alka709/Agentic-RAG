import io
import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from pypdf import PdfReader

from config import IMAGE_OUTPUT_DIR, VISION_LLM_MODEL
from rag.table_extractor import extract_tables_from_pdf
from rag.vision_analyzer import describe_image

logger = logging.getLogger(__name__)

SUPPORTED_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader
}

# Minimum dimensions to filter out tiny decorative icons/bullets from PDFs
MIN_IMAGE_WIDTH = 60
MIN_IMAGE_HEIGHT = 60


def extract_images_from_pdf(
    file_path: Path,
    output_dir: Optional[Path] = None,
    vision_model: Optional[str] = None
) -> List[Document]:
    """
    Extracts embedded images from a PDF, saves them to disk, generates
    retrieval-friendly descriptions using the configured vision model,
    and returns Document objects with content_type='image'.
    """
    file_path = Path(file_path)
    output_dir = output_dir or IMAGE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_model = vision_model or VISION_LLM_MODEL
    image_documents: List[Document] = []
    doc_stem = file_path.stem

    try:
        reader = PdfReader(str(file_path))
        for page_idx, page in enumerate(reader.pages, 1):
            if not hasattr(page, "images") or not page.images:
                continue

            for img_idx, image_obj in enumerate(page.images, 1):
                # Filter out small decorative icons/bullets
                try:
                    with Image.open(io.BytesIO(image_obj.data)) as img:
                        w, h = img.size
                        if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
                            continue
                except Exception:
                    pass

                image_filename = f"{doc_stem}_p{page_idx}_img{img_idx}.png"
                image_path = output_dir / image_filename

                # Save the raw image to disk
                with open(image_path, "wb") as f:
                    f.write(image_obj.data)

                # Generate a detailed technical description via Vision LLM
                description = describe_image(
                    image_path=image_path,
                    model_name=vision_model,
                    source=file_path.name,
                    page=page_idx
                )

                image_id = f"{doc_stem}_p{page_idx}_img{img_idx}"
                metadata = {
                    "content_type": "image",
                    "source": str(file_path),
                    "page": page_idx,
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "description": description,
                }

                image_documents.append(
                    Document(page_content=description, metadata=metadata)
                )

    except Exception as e:
        logger.error(f"Error extracting images from {file_path}: {e}")

    return image_documents


def load_documents(file_path: str | Path) -> List[Document]:
    """
    Loads and extracts multimodal content (text, tables, and images)
    from a document.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_LOADERS and suffix not in [".png", ".jpg", ".jpeg", ".webp"]:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Direct image loading support
    if suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = IMAGE_OUTPUT_DIR / file_path.name
        if file_path.resolve() != dest_path.resolve():
            dest_path.write_bytes(file_path.read_bytes())

        description = describe_image(
            image_path=dest_path,
            source=file_path.name,
            page=1
        )
        return [
            Document(
                page_content=description,
                metadata={
                    "content_type": "image",
                    "source": str(file_path),
                    "page": 1,
                    "image_id": file_path.stem,
                    "image_path": str(dest_path),
                    "description": description,
                }
            )
        ]

    # Standard text document loading
    loader_class = SUPPORTED_LOADERS[suffix]
    loader = loader_class(str(file_path))
    text_docs = loader.load()

    # Ensure all text documents have content_type="text"
    for idx, doc in enumerate(text_docs, 1):
        doc.metadata["content_type"] = "text"
        if "page" in doc.metadata and isinstance(doc.metadata["page"], int):
            doc.metadata["page"] = doc.metadata["page"] + 1  # make 1-indexed
        elif "page" not in doc.metadata:
            doc.metadata["page"] = idx

    # For PDF documents, extract tables and images
    if suffix == ".pdf":
        table_docs = extract_tables_from_pdf(file_path)
        image_docs = extract_images_from_pdf(file_path)
        return text_docs + table_docs + image_docs

    return text_docs