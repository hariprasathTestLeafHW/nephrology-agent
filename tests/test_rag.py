import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nephrology_agent.rag import (
    answer_question,
    chunk_text,
    create_ephemeral_collection,
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


class EphemeralCollectionTests(unittest.TestCase):
    def test_collections_are_isolated_between_sessions(self) -> None:
        _, first_collection = create_ephemeral_collection()
        _, second_collection = create_ephemeral_collection()
        first_collection.add(
            ids=["one"],
            embeddings=[[0.0] * 768],
            documents=["session-specific text"],
            metadatas=[{"filename": "guide.pdf", "page": 1, "sha256": "one"}],
        )

        self.assertEqual(first_collection.count(), 1)
        self.assertEqual(second_collection.count(), 0)


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
