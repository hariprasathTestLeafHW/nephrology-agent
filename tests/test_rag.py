import hashlib
import subprocess
import sys
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

    def test_persistent_collection_survives_client_recreation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            document_id = uuid4().hex
            script = """
import sys
from pathlib import Path
from nephrology_agent.rag import create_shared_persistent_collection

client, collection = create_shared_persistent_collection(Path(sys.argv[1]))
if sys.argv[2] == "write":
    collection.add(
        ids=[sys.argv[3]],
        embeddings=[[0.0] * 768],
        documents=["persisted source text"],
        metadatas=[{"filename": "guide.pdf", "page": 1, "sha256": "one"}],
    )
else:
    print(collection.get(ids=[sys.argv[3]])["documents"][0])
client.close()
"""
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    temporary_directory,
                    "write",
                    document_id,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    temporary_directory,
                    "read",
                    document_id,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.stdout.strip(), "persisted source text")


class PreloadedPdfTests(unittest.TestCase):
    @patch("nephrology_agent.rag.index_pdf")
    def test_indexes_all_preloaded_pdfs_in_filename_order(self, mock_index_pdf: Mock) -> None:
        with TemporaryDirectory() as temporary_directory:
            pdf_directory = Path(temporary_directory)
            (pdf_directory / "b.pdf").write_bytes(b"%PDF-b")
            (pdf_directory / "a.pdf").write_bytes(b"%PDF-a")

            collection = Mock()
            collection.count.return_value = 4
            collection.get.side_effect = [
                {"ids": [], "metadatas": []},
                {"ids": [], "metadatas": []},
                {"ids": [], "metadatas": []},
            ]
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
        collection.get.assert_called()

    def test_missing_preloaded_pdfs_are_reported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                index_preloaded_pdfs(
                    client=Mock(),
                    collection=Mock(),
                    pdf_directory=Path(temporary_directory),
                    embedding_model="test-embedding-model",
                )

    @patch("nephrology_agent.rag.index_pdf")
    def test_removes_stale_files_from_persistent_collection(
        self,
        mock_index_pdf: Mock,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            pdf_directory = Path(temporary_directory)
            (pdf_directory / "current.pdf").write_bytes(b"%PDF-current")

            collection = Mock()
            collection.get.return_value = {
                "ids": ["removed-document"],
                "metadatas": [{"filename": "removed.pdf"}],
            }
            collection.count.return_value = 3

            index_preloaded_pdfs(
                client=Mock(),
                collection=collection,
                pdf_directory=pdf_directory,
                embedding_model="test-embedding-model",
            )

        collection.delete.assert_called_once_with(ids=["removed-document"])
        mock_index_pdf.assert_called_once()


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

    def test_adjacent_retrieved_chunks_do_not_repeat_their_overlap(self) -> None:
        overlap = " ".join(f"w{index}" for index in range(25))
        earlier = f"Earlier evidence. {overlap}"
        later = f"{overlap} Later evidence."
        earlier_metadata = {
            "filename": "kidney-guide.pdf",
            "page": 4,
            "chunk_number": 0,
            "source_url": "",
        }
        later_metadata = {
            "filename": "kidney-guide.pdf",
            "page": 4,
            "chunk_number": 1,
            "source_url": "",
        }
        collection = Mock()
        collection.count.return_value = 2
        collection.query.return_value = {
            "documents": [[earlier, later]],
            "metadatas": [[earlier_metadata, later_metadata]],
            "distances": [[0.1, 0.2]],
        }

        client = Mock()
        client.models.embed_content.return_value.embeddings = [
            SimpleNamespace(values=[0.0] * 768)
        ]
        client.models.generate_content.return_value.text = "A cited educational answer [1]."

        _, sources = answer_question(
            client=client,
            collection=collection,
            question="What does the source explain?",
            model="test-model",
            embedding_model="test-embedding-model",
        )

        prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(prompt.count(overlap), 1)
        self.assertIn("Earlier evidence.", prompt)
        self.assertIn("Later evidence.", prompt)
        self.assertEqual([source["number"] for source in sources], [1, 2])

        collection.query.return_value = {
            "documents": [[later, earlier]],
            "metadatas": [[later_metadata, earlier_metadata]],
            "distances": [[0.1, 0.2]],
        }
        _, reverse_sources = answer_question(
            client=client,
            collection=collection,
            question="What does the source explain?",
            model="test-model",
            embedding_model="test-embedding-model",
        )
        reverse_prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(reverse_prompt.count(overlap), 1)
        self.assertIn("Earlier evidence.", reverse_prompt)
        self.assertIn("Later evidence.", reverse_prompt)
        self.assertEqual([source["number"] for source in reverse_sources], [1, 2])


class IndexPdfTests(unittest.TestCase):
    @patch("nephrology_agent.rag._embed")
    @patch("nephrology_agent.rag.PdfReader")
    def test_existing_current_index_skips_embedding(
        self,
        pdf_reader: Mock,
        embed: Mock,
    ) -> None:
        content = b"%PDF-1.4\nsample"
        collection = Mock()
        collection.get.return_value = {
            "ids": ["existing-document"],
            "metadatas": [{
                "filename": "sample.pdf",
                "sha256": hashlib.sha256(content).hexdigest(),
                "embedding_model": "test-embedding-model",
                "embedding_dimensions": 768,
                "chunk_size": 1200,
                "chunk_overlap": 180,
                "index_version": 1,
            }],
        }

        chunks_added, already_indexed = index_pdf(
            client=Mock(),
            collection=collection,
            filename="sample.pdf",
            content=content,
            embedding_model="test-embedding-model",
        )

        self.assertEqual(chunks_added, 0)
        self.assertTrue(already_indexed)
        pdf_reader.assert_not_called()
        embed.assert_not_called()

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
        self.assertEqual(metadata["chunk_number"], 0)
        self.assertEqual(metadata["source_url"], "https://example.org/sample.pdf")
        self.assertEqual(metadata["embedding_model"], "test-embedding-model")
        self.assertEqual(metadata["embedding_dimensions"], 768)
        self.assertEqual(metadata["chunk_size"], 1200)
        self.assertEqual(metadata["chunk_overlap"], 180)
        self.assertEqual(metadata["index_version"], 1)

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
