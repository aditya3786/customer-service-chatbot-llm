"""
Task 3 — Dynamic Knowledge Base Expansion
Tests for URL ingestion, deduplication, text chunking, and scheduler.
Run: pytest tests/test_task3_knowledge_base.py -v
"""

import sys
import os
import json
import hashlib
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


# ── Deduplication Logic ───────────────────────────────────────────────────────

class TestDeduplication:
    def test_same_text_produces_same_hash(self):
        text = "This is a test document."
        h1 = hashlib.sha256(text.encode()).hexdigest()
        h2 = hashlib.sha256(text.encode()).hexdigest()
        assert h1 == h2

    def test_different_text_produces_different_hash(self):
        h1 = hashlib.sha256("text one".encode()).hexdigest()
        h2 = hashlib.sha256("text two".encode()).hexdigest()
        assert h1 != h2

    def test_hash_is_64_hex_characters(self):
        h = hashlib.sha256("test".encode()).hexdigest()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_whitespace_difference_changes_hash(self):
        h1 = hashlib.sha256("hello world".encode()).hexdigest()
        h2 = hashlib.sha256("hello  world".encode()).hexdigest()
        assert h1 != h2


# ── Knowledge Expander Module ─────────────────────────────────────────────────

class TestKnowledgeExpanderModule:
    def test_module_imports(self):
        import knowledge_expander
        assert hasattr(knowledge_expander, "ingest_url")
        assert hasattr(knowledge_expander, "ingest_manual_text")
        assert hasattr(knowledge_expander, "get_ingestion_history")
        assert hasattr(knowledge_expander, "get_index_doc_count")
        assert hasattr(knowledge_expander, "KB_PATHS")

    def test_kb_paths_has_both_knowledge_bases(self):
        import knowledge_expander
        assert "Customer Service" in knowledge_expander.KB_PATHS
        assert "Medical Q&A" in knowledge_expander.KB_PATHS

    def test_kb_paths_are_strings(self):
        import knowledge_expander
        for kb, path in knowledge_expander.KB_PATHS.items():
            assert isinstance(path, str)

    def test_get_ingestion_history_returns_list(self):
        import knowledge_expander
        history = knowledge_expander.get_ingestion_history()
        assert isinstance(history, list)

    def test_get_index_doc_count_returns_int(self):
        import knowledge_expander
        count = knowledge_expander.get_index_doc_count("Customer Service")
        assert isinstance(count, int)
        assert count >= 0


# ── Text Chunking ─────────────────────────────────────────────────────────────

class TestTextChunking:
    def test_short_text_produces_one_chunk(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text("This is a short text.")
        assert len(chunks) == 1

    def test_long_text_is_split(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        long_text = "word " * 200
        chunks = splitter.split_text(long_text)
        assert len(chunks) > 1

    def test_chunks_preserve_content(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text = "The quick brown fox jumps over the lazy dog."
        chunks = splitter.split_text(text)
        combined = " ".join(chunks)
        for word in ["quick", "brown", "fox", "lazy", "dog"]:
            assert word in combined

    def test_chunk_size_respected(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        long_text = "A" * 500
        chunks = splitter.split_text(long_text)
        for chunk in chunks:
            assert len(chunk) <= 60  # allow small buffer


# ── Ingestion Log ─────────────────────────────────────────────────────────────

class TestIngestionLog:
    def test_log_file_is_valid_json_if_exists(self):
        log_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "ingestion_log.json"
        )
        if os.path.exists(log_path):
            with open(log_path) as f:
                data = json.load(f)
            assert isinstance(data, (list, dict))

    def test_log_entries_have_expected_keys(self):
        log_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "ingestion_log.json"
        )
        if os.path.exists(log_path):
            with open(log_path) as f:
                data = json.load(f)
            # Log is stored as dict keyed by content hash
            if isinstance(data, dict) and data:
                entry = next(iter(data.values()))
                for key in ["source", "kb", "timestamp", "chunks_added"]:
                    assert key in entry, f"Missing key: {key}"
            elif isinstance(data, list) and data:
                entry = data[0]
                for key in ["source", "kb", "timestamp", "chunks_added"]:
                    assert key in entry, f"Missing key: {key}"

    def test_manual_text_dedup(self):
        import knowledge_expander
        text = "unique_dedup_test_content_xyz_12345"
        result1 = knowledge_expander.ingest_manual_text(
            "Customer Service", text, "Dedup test answer."
        )
        result2 = knowledge_expander.ingest_manual_text(
            "Customer Service", text, "Dedup test answer."
        )
        assert result1["status"] in ("added", "skipped")
        assert result2["status"] == "skipped"


# ── Scheduler ─────────────────────────────────────────────────────────────────

class TestScheduler:
    def test_module_imports(self):
        import kb_scheduler
        assert hasattr(kb_scheduler, "start_scheduler")
        assert hasattr(kb_scheduler, "stop_scheduler")
        assert hasattr(kb_scheduler, "is_running")
        assert hasattr(kb_scheduler, "add_source")
        assert hasattr(kb_scheduler, "get_sources")
        assert hasattr(kb_scheduler, "run_now")

    def test_initially_not_running(self):
        import kb_scheduler
        kb_scheduler.stop_scheduler()
        assert kb_scheduler.is_running() is False

    def test_start_and_stop(self):
        import kb_scheduler
        kb_scheduler.start_scheduler(interval_minutes=60)
        assert kb_scheduler.is_running() is True
        kb_scheduler.stop_scheduler()
        assert kb_scheduler.is_running() is False

    def test_get_sources_returns_list(self):
        import kb_scheduler
        sources = kb_scheduler.get_sources()
        assert isinstance(sources, list)

    def test_add_source_no_duplicate(self):
        import kb_scheduler
        initial = len(kb_scheduler.get_sources())
        kb_scheduler.add_source("Customer Service", "https://example.com/test-unique-url-1")
        kb_scheduler.add_source("Customer Service", "https://example.com/test-unique-url-1")
        sources = kb_scheduler.get_sources()
        count = sum(
            1 for s in sources
            if s["url"] == "https://example.com/test-unique-url-1"
        )
        assert count == 1

    def test_reschedule_does_not_crash(self):
        import kb_scheduler
        kb_scheduler.start_scheduler(interval_minutes=30)
        kb_scheduler.start_scheduler(interval_minutes=60)  # reschedule
        assert kb_scheduler.is_running() is True
        kb_scheduler.stop_scheduler()
