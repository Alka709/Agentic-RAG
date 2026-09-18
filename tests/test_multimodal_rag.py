import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.table_extractor import format_table_to_text
from rag.vision_analyzer import describe_image
from rag.splitter import split_documents
from rag.vector_store import create_vector_store
from rag.retriever import retrieve_documents
from rag.embeddings import get_embedding_model
from config import EMBEDDING_MODEL


class TestMultimodalRAG(unittest.TestCase):

    def test_table_formatting(self):
        table_data = [
            ["Model", "Accuracy", "Latency"],
            ["BERT", "92%", "100ms"],
            ["GPT", "95%", "150ms"]
        ]
        text = format_table_to_text(table_data, page_num=4, source_name="paper.pdf", table_idx=1)

        self.assertIn("Table 1 from page 4 in paper.pdf.", text)
        self.assertIn("Columns: Model, Accuracy, Latency.", text)
        self.assertIn("BERT", text)
        self.assertIn("92%", text)
        self.assertIn("100ms", text)
        self.assertIn("Markdown Table:", text)
        self.assertIn("| Model | Accuracy | Latency |", text)

    def test_vision_analyzer_fallback(self):
        # Test fallback when vision model is not reachable
        test_img = Path("test_dummy.png")
        test_img.write_bytes(b"dummy image data")
        try:
            with patch("rag.vision_analyzer.ChatOllama") as mock_ollama:
                mock_instance = MagicMock()
                mock_instance.invoke.side_effect = Exception("Ollama connection failed")
                mock_ollama.return_value = mock_instance

                desc = describe_image(test_img, source="sample.pdf", page=2)
                self.assertIn("Image extracted from page 2 of sample.pdf", desc)
        finally:
            if test_img.exists():
                test_img.unlink()

    def test_multimodal_splitter(self):
        text_doc = Document(
            page_content="This is a long text paragraph that discusses AI systems in great detail." * 10,
            metadata={"source": "doc.pdf", "page": 1, "content_type": "text"}
        )
        table_doc = Document(
            page_content="Table 1 from page 2. Columns: A, B. Row 1: A: 1, B: 2",
            metadata={"source": "doc.pdf", "page": 2, "content_type": "table", "table_id": "doc_p2_tbl1"}
        )
        image_doc = Document(
            page_content="Architecture diagram showing input -> model -> output",
            metadata={"source": "doc.pdf", "page": 3, "content_type": "image", "image_id": "doc_p3_img1", "image_path": "uploads/img1.png"}
        )

        docs = [text_doc, table_doc, image_doc]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)

        # Ensure all types exist in chunks
        content_types = [c.metadata.get("content_type") for c in chunks]
        self.assertIn("text", content_types)
        self.assertIn("table", content_types)
        self.assertIn("image", content_types)

        # Check metadata IDs
        table_chunks = [c for c in chunks if c.metadata.get("content_type") == "table"]
        image_chunks = [c for c in chunks if c.metadata.get("content_type") == "image"]
        text_chunks = [c for c in chunks if c.metadata.get("content_type") == "text"]

        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(table_chunks[0].metadata["table_id"], "doc_p2_tbl1")
        self.assertEqual(len(image_chunks), 1)
        self.assertEqual(image_chunks[0].metadata["image_id"], "doc_p3_img1")
        self.assertIn("chunk_id", text_chunks[0].metadata)

    def test_multimodal_vector_retrieval(self):
        embeddings = get_embedding_model(EMBEDDING_MODEL)

        docs = [
            Document(
                page_content="The quarterly revenue reached 50 million dollars in Q3.",
                metadata={"source": "report.pdf", "page": 1, "content_type": "text", "chunk_id": "report_p1_c1"}
            ),
            Document(
                page_content="Table 1 from page 2. Columns: Product, Sales, Profit. Product: Alpha, Sales: $10M, Profit: $2M.",
                metadata={"source": "report.pdf", "page": 2, "content_type": "table", "table_id": "report_p2_tbl1"}
            ),
            Document(
                page_content="Bar chart diagram showing quarterly sales growth trends from Q1 to Q4.",
                metadata={"source": "report.pdf", "page": 3, "content_type": "image", "image_id": "report_p3_img1", "image_path": "uploads/chart.png"}
            )
        ]

        vector_store = create_vector_store(docs, embeddings)

        # Retrieve table
        results = retrieve_documents(vector_store, "What were the sales and profit for product Alpha?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content_type"], "table")
        self.assertEqual(results[0]["table_id"], "report_p2_tbl1")

        # Retrieve image
        img_results = retrieve_documents(vector_store, "Show me the bar chart diagram of sales growth", top_k=1)
        self.assertEqual(len(img_results), 1)
        self.assertEqual(img_results[0]["content_type"], "image")
        self.assertEqual(img_results[0]["image_path"], "uploads/chart.png")


if __name__ == "__main__":
    unittest.main()
