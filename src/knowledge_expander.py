"""
Dynamic Knowledge Base Expansion
=================================
Mechanism to incrementally update a FAISS vector store with new information
from external sources (URLs, manual entries) without rebuilding the index
from scratch.

Dedup is content-hash based: re-ingesting unchanged content is skipped, but
if a source's content changes, the new hash differs and it's ingested as
fresh information — this is what makes the system "dynamic" rather than a
one-time import.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

import json
import hashlib
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_helper import embeddings, vectordb_file_path as CS_INDEX_PATH
from medical_helper import MEDICAL_VECTORDB_PATH

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(_SRC_DIR, "ingestion_log.json")

KB_PATHS = {
    "Customer Service": CS_INDEX_PATH,
    "Medical Q&A": MEDICAL_VECTORDB_PATH,
}

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def _load_log() -> dict:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {}


def _save_log(log: dict):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_url_text(url: str) -> str:
    """Fetch a webpage and extract its visible text content."""
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def add_to_knowledge_base(kb_name: str, text: str, source_label: str) -> dict:
    """
    Chunk, dedup-check, embed, and merge new text into the target FAISS index.

    Returns {"status": "added"|"skipped", "chunks_added": int}
    """
    if kb_name not in KB_PATHS:
        raise ValueError(f"Unknown knowledge base: {kb_name}")

    index_path = KB_PATHS[kb_name]
    content_hash = _content_hash(text)
    log = _load_log()

    if content_hash in log:
        return {"status": "skipped", "reason": "duplicate content", "chunks_added": 0}

    chunks = _splitter.split_text(text)
    docs = [Document(page_content=chunk, metadata={"source": source_label}) for chunk in chunks]

    if os.path.exists(index_path):
        vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        vectordb.add_documents(docs)
    else:
        vectordb = FAISS.from_documents(docs, embeddings)

    vectordb.save_local(index_path)

    log[content_hash] = {
        "source": source_label,
        "kb": kb_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "chunks_added": len(chunks),
    }
    _save_log(log)

    return {"status": "added", "chunks_added": len(chunks)}


def ingest_url(kb_name: str, url: str) -> dict:
    text = fetch_url_text(url)
    return add_to_knowledge_base(kb_name, text, source_label=url)


def ingest_manual_text(kb_name: str, prompt: str, response: str) -> dict:
    text = f"prompt: {prompt}\nresponse: {response}"
    return add_to_knowledge_base(kb_name, text, source_label=f"manual: {prompt[:50]}")


def get_ingestion_history() -> list:
    """Return ingestion log entries, most recent first."""
    log = _load_log()
    history = list(log.values())
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    return history


def get_index_doc_count(kb_name: str) -> int:
    """Return the number of vectors currently stored in a KB's FAISS index."""
    index_path = KB_PATHS.get(kb_name)
    if not index_path or not os.path.exists(index_path):
        return 0
    vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectordb.index.ntotal
