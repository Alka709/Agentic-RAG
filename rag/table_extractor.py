import logging
from pathlib import Path
from typing import List, Optional

import pdfplumber
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def format_table_to_text(table_data: List[List[Optional[str]]], page_num: int, source_name: str, table_idx: int) -> str:
    """
    Converts raw table grid data into meaningful, retrieval-friendly text preserving:
    - Column headers
    - Rows & numerical values
    - Relational cell statements
    - Clean Markdown table representation
    """
    cleaned_rows = [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in table_data
        if any(cell is not None and str(cell).strip() != "" for cell in row)
    ]

    if not cleaned_rows:
        return ""

    headers = cleaned_rows[0]
    data_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []

    header_names = [h if h else f"Column_{i+1}" for i, h in enumerate(headers)]

    text_parts = [
        f"Table {table_idx} from page {page_num} in {source_name}.",
        f"Columns: {', '.join(header_names)}.",
    ]

    # Relational sentences for rows
    if data_rows:
        text_parts.append("Data Entries:")
        for row_idx, row in enumerate(data_rows, 1):
            row_items = []
            for col_idx, cell_value in enumerate(row):
                col_name = header_names[col_idx] if col_idx < len(header_names) else f"Column_{col_idx+1}"
                if cell_value:
                    row_items.append(f"{col_name}: {cell_value}")
            if row_items:
                text_parts.append(f"- Row {row_idx}: {', '.join(row_items)}")

    # Add Markdown table format
    text_parts.append("\nMarkdown Table:")
    md_header = "| " + " | ".join(header_names) + " |"
    md_divider = "| " + " | ".join(["---"] * len(header_names)) + " |"
    text_parts.append(md_header)
    text_parts.append(md_divider)

    for row in data_rows:
        padded_row = row + [""] * max(0, len(header_names) - len(row))
        md_row = "| " + " | ".join(padded_row[:len(header_names)]) + " |"
        text_parts.append(md_row)

    return "\n".join(text_parts)


def extract_tables_from_pdf(file_path: Path) -> List[Document]:
    """
    Extracts all tables from a PDF using pdfplumber and returns a list of Document objects
    with content_type='table' metadata.
    """
    file_path = Path(file_path)
    table_documents: List[Document] = []
    source_name = file_path.name
    doc_stem = file_path.stem

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                if not tables:
                    continue

                for table_idx, table_data in enumerate(tables, 1):
                    formatted_text = format_table_to_text(
                        table_data=table_data,
                        page_num=page_idx,
                        source_name=source_name,
                        table_idx=table_idx,
                    )

                    if not formatted_text.strip():
                        continue

                    table_id = f"{doc_stem}_p{page_idx}_tbl{table_idx}"
                    metadata = {
                        "content_type": "table",
                        "source": str(file_path),
                        "page": page_idx,
                        "table_id": table_id,
                    }

                    table_documents.append(
                        Document(page_content=formatted_text, metadata=metadata)
                    )

    except Exception as e:
        logger.error(f"Error extracting tables from {file_path}: {e}")

    return table_documents
