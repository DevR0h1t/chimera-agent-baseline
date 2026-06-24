"""Tests for the RAG module."""

import pytest

from chimera_agent_baseline.rag import GuidelinesSearch


class TestGuidelinesSearch:
    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Guidelines DB not found"):
            GuidelinesSearch(tmp_path)

    def test_search_with_synthetic_db(self, tmp_path):
        """Build a small ChromaDB collection and verify search works."""
        from chimera_agent_baseline.rag import SOCKET_PATH

        if not SOCKET_PATH.exists():
            pytest.skip("Embedding service not running (start it from the main process)")

        import chromadb

        db_path = tmp_path / "guidelines_db"
        db_path.mkdir()

        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.create_collection("guidelines")
        collection.add(
            ids=["chunk-0", "chunk-1", "chunk-2"],
            documents=[
                "Active surveillance is recommended for low-risk prostate cancer with Gleason 3+3.",
                "Radical prostatectomy is indicated for high-risk localized prostate cancer.",
                "PI-RADS 4 and 5 lesions on MRI are suspicious for clinically significant cancer.",
            ],
            metadatas=[
                {"page": 42, "section": "6.1 Active Surveillance"},
                {"page": 55, "section": "6.2 Radical Prostatectomy"},
                {"page": 30, "section": "5.2 MRI"},
            ],
        )

        search = GuidelinesSearch(tmp_path)
        results = search.query("active surveillance criteria", top_k=2)
        assert len(results) == 2
        assert all("text" in r for r in results)
        assert all("page" in r for r in results)
        assert all("section" in r for r in results)


class TestMCPServerFallback:
    def test_create_server_without_resource_dir(self, tmp_path):
        """Server creates fine without resource_dir (guidelines returns empty)."""
        from chimera_agent_baseline.mcp_server import create_server

        server = create_server(str(tmp_path))
        assert server is not None

    def test_create_server_with_missing_db(self, tmp_path):
        """Server creates fine when resource_dir exists but has no guidelines_db."""
        from chimera_agent_baseline.mcp_server import create_server

        server = create_server(str(tmp_path), resource_dir=str(tmp_path))
        assert server is not None
