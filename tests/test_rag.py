import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from nephrology_agent.rag import (
    answer_question,
    chunk_text,
    create_shared_ephemeral_collection,
    index_preloaded_pdfs,
    index_pdf,
)


class ChunkTextTests(unittest.TestCase):
    def test_empty_and_whitespace_text_produce_no_chunks(self) -> None:
        self.assertEqual(chunk_text(" \n\t "), [])

    def test_chunks_respect_size_and_preserve_full_text(self) -> None:
        source = "alpha beta gamma delta epsilon"
        chunks = chunk_text(source, size=12, overlap=3)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 12 for chunk in chunks))
        self.assertIn("alpha", chunks[0])
        self.assertIn("epsilon", chunks[-1])

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("text", size=10, overlap=10)


class SharedCollectionTests(unittest.TestCase):
    def test_shared_collection_data_is_visible_to_both_clients(self) -> None:
        _, first_collection = create_shared_ephemeral_collection()
        _, second_collection = create_shared_ephemeral_collection()
        document_id = uuid4().hex
        first_collection.add(
            ids=[document_id],
            embeddings=[[0.0] * 768],
            documents=["session-specific text"],
            metadatas=[{"filename": "guide.pdf", "page": 1, "sha256": "one"}],
        )

        result = second_collection.get(ids=[document_id])
        self.assertEqual(result["documents"], ["session-specific text"])


class PreloadedPdfTests(unittest.TestCase):
    @patch("nephrology_agent.rag.index_pdf")
    def test_indexes_all_preloaded_pdfs_in_filename_order(self, mock_index_pdf: Mock) -> None:
        with TemporaryDirectory() as temporary_directory:
            pdf_directory = Path(temporary_directory)
            (pdf_directory / "b.pdf").write_bytes(b"%PDF-b")
            (pdf_directory / "a.pdf").write_bytes(b"%PDF-a")

            collection = Mock()
            collection.count.return_value = 4
            indexed_count = index_preloaded_pdfs(
                client=Mock(),
                collection=collection,
                pdf_directory=pdf_directory,
                embedding_model="test-embedding-model",
            )

        self.assertEqual(indexed_count, 4)
        self.assertEqual(
            [entry.kwargs["filename"] for entry in mock_index_pdf.call_args_list],
            ["a.pdf", "b.pdf"],
        )

    def test_missing_preloaded_pdfs_are_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                index_preloaded_pdfs(
                    client=Mock(),
                    collection=Mock(),
                    pdf_directory=Path(temporary_directory),
                    embedding_model="test-embedding-model",
                )


class AnswerQuestionTests(unittest.TestCase):
    def test_answer_includes_metadata_for_retrieved_citations(self) -> None:
        collection = Mock()
        collection.count.return_value = 1
        collection.query.return_value = {
            "documents": [["The source states a general educational fact."]],
            "metadatas": [[
                {
                    "filename": "kidney-guide.pdf",
                    "page": 4,
                    "source_url": "https://example.org/kidney-guide.pdf",
                }
            ]],
            "distances": [[0.1]],
        }

        client = Mock()
        client.models.embed_content.return_value.embeddings = [
            SimpleNamespace(values=[0.0] * 768)
        ]
        client.models.generate_content.return_value.text = "A cited educational answer [1]."

        answer, sources = answer_question(
            client=client,
            collection=collection,
            question="What does the source explain?",
            model="test-model",
            embedding_model="test-embedding-model",
        )

        self.assertEqual(answer, "A cited educational answer [1].")
        self.assertEqual(sources[0]["filename"], "kidney-guide.pdf")
        self.assertEqual(sources[0]["page"], 4)
        self.assertEqual(sources[0]["source_url"], "https://example.org/kidney-guide.pdf")
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("The source states a general educational fact.", prompt)
        self.assertIn("general overview of relevant evaluation or treatment approaches", prompt)
        self.assertIn("clearly labeling them as general information", prompt)
        self.assertIn("confirm treatment decisions with their physician", prompt)
        self.assertIn("Never provide medication doses", prompt)
        self.assertIn("lead with advice to seek urgent or emergency care", prompt)


class IndexPdfTests(unittest.TestCase):
    @patch("nephrology_agent.rag._embed")
    @patch("nephrology_agent.rag.PdfReader")
    def test_index_stores_chunks_with_page_and_source_metadata(
        self,
        pdf_reader: Mock,
        embed: Mock,
    ) -> None:
        pdf_reader.return_value.pages = [
            SimpleNamespace(extract_text=lambda: "A sample nephrology education passage.")
        ]
        embed.return_value = [[0.0] * 768]
        collection = Mock()
        collection.get.return_value = {"ids": []}

        chunks_added, already_indexed = index_pdf(
            client=Mock(),
            collection=collection,
            filename="sample.pdf",
            content=b"%PDF-1.4\nsample",
            embedding_model="test-embedding-model",
            source_url="https://example.org/sample.pdf",
        )

        self.assertEqual(chunks_added, 1)
        self.assertFalse(already_indexed)
        metadata = collection.add.call_args.kwargs["metadatas"][0]
        self.assertEqual(metadata["filename"], "sample.pdf")
        self.assertEqual(metadata["page"], 1)
        self.assertEqual(metadata["source_url"], "https://example.org/sample.pdf")

    @patch("nephrology_agent.rag._embed", side_effect=RuntimeError("embedding failed"))
    @patch("nephrology_agent.rag.PdfReader")
    def test_embedding_failure_does_not_store_partial_pdf(
        self,
        pdf_reader: Mock,
        _embed: Mock,
    ) -> None:
        pdf_reader.return_value.pages = [
            SimpleNamespace(extract_text=lambda: "A sample nephrology education passage.")
        ]
        collection = Mock()
        collection.get.return_value = {"ids": []}

        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            index_pdf(
                client=Mock(),
                collection=collection,
                filename="sample.pdf",
                content=b"%PDF-1.4\nsample",
                embedding_model="test-embedding-model",
            )

        collection.add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
