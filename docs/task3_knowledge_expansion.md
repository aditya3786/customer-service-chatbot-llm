# Task 3 — Dynamic Knowledge Base Expansion

## Objective
Build a mechanism to periodically update the chatbot's FAISS vector database with new information from user-specified sources, so the chatbot incorporates new knowledge over time without manual rebuilding.

---

## Methodology

### Incremental FAISS Updates
Rather than rebuilding the entire index on every update, new content is merged into the existing index:

```
Load existing FAISS index from disk
    ↓
Embed new chunks with HuggingFace all-MiniLM-L6-v2
    ↓
FAISS.add_documents(new_chunks)   ← fast, non-destructive
    ↓
Save updated index back to disk
```

This preserves all previously indexed documents and avoids re-embedding the full dataset.

### Deduplication via SHA-256 Content Hashing
Every ingested text is SHA-256 hashed and stored in `src/ingestion_log.json`:

```python
content_hash = hashlib.sha256(text.encode()).hexdigest()
# If hash already in log → skip (status: "skipped")
# If hash is new → ingest + log (status: "added")
```

**Why this makes updates "dynamic":** If a registered URL's page content changes, the new hash differs from the stored one — the updated content is automatically re-ingested on the next refresh cycle.

### URL Ingestion — BeautifulSoup
```python
response = requests.get(url, timeout=10)
soup = BeautifulSoup(response.text, "html.parser")
# Remove noise: script, style, nav, footer tags
for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()
text = soup.get_text(separator=" ", strip=True)
```

### Text Chunking
Long documents are split before embedding:
```
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```
Overlap of 100 characters ensures sentence context is not lost at chunk boundaries.

### Periodic Refresh — APScheduler
`APScheduler.BackgroundScheduler` runs as a daemon thread inside the Streamlit process:

```python
_scheduler = BackgroundScheduler()
_scheduler.add_job(_run_scheduled_refresh, "interval", minutes=interval_minutes)
_scheduler.start()
```

**Why module-level globals?** Streamlit re-executes the full script on every UI interaction. `st.session_state` resets between re-renders for scheduler objects. Module-level `_scheduler` persists for the lifetime of the Python process — the only correct place to hold APScheduler state.

### Source Persistence
Registered URL sources are written to `src/sources_config.json` so they survive app restarts:

```json
[
  {"kb": "Customer Service", "url": "https://example.com/faq"},
  {"kb": "Medical Q&A", "url": "https://example.com/health-update"}
]
```

### Generic Across Both Knowledge Bases
The same mechanism targets either:
- `src/faiss_index/` — Customer Service KB
- `src/medical_faiss_index/` — Medical Q&A KB

Selectable per source in the UI.

---

## Pipeline Flow
```
User registers URL + target KB
    ↓
fetch_url_text(url) → BeautifulSoup text extraction
    ↓
SHA-256 hash check against ingestion_log.json
    ├── Same hash → skip (no change detected)
    └── New hash → chunk → embed → FAISS.add_documents() → save index
    ↓
Log entry written: {source, kb, timestamp, chunks_added}
    ↓
(Optional) Register in sources_config.json for periodic refresh
    ↓
BackgroundScheduler re-runs fetch on interval → auto re-ingest on change
```

---

## Files
| File | Role |
|------|------|
| `src/knowledge_expander.py` | Core ingestion: fetch, dedup, chunk, FAISS update |
| `src/kb_scheduler.py` | APScheduler wrapper + source persistence |
| `src/ingestion_log.json` | Runtime log (gitignored, regenerated at runtime) |
| `src/sources_config.json` | Registered URL sources (gitignored, regenerated at runtime) |

## Key Functions
```python
# knowledge_expander.py
ingest_url(kb_name: str, url: str) -> dict          # {"status": "added"|"skipped", "chunks_added": int}
ingest_manual_text(kb_name, prompt, response) -> dict
get_ingestion_history() -> list[dict]
get_index_doc_count(kb_name: str) -> int

# kb_scheduler.py
start_scheduler(interval_minutes: int)
stop_scheduler()
is_running() -> bool
add_source(kb_name: str, url: str)
get_sources() -> list[dict]
run_now() -> list[dict]
```

---

## UI Features (🔄 Auto-Update Tab)
- **Live vector counts** per knowledge base
- **Add URL source** form — "Ingest once now" or "Register for periodic refresh"
- **Add manual Q&A** form — instant embedding of typed content
- **Scheduler controls** — start/stop with configurable interval (1–1440 min)
- **Ingestion history** table — source, target KB, timestamp, chunks added

---

## Test Results

| Test | Result |
|------|--------|
| Same content → same SHA-256 hash | ✅ |
| Different content → different hash | ✅ |
| Manual text dedup (same input twice → second is skipped) | ✅ |
| Scheduler starts and stops correctly | ✅ |
| `add_source` prevents duplicate registrations | ✅ |
| `get_ingestion_history()` returns list | ✅ |
| `get_index_doc_count()` returns non-negative int | ✅ |

---

## How to Run Tests
```bash
pytest tests/test_task3_knowledge_base.py -v
```
**Note:** The `test_manual_text_dedup` test requires the Customer Service FAISS index to be built first (click "Create Knowledgebase" in the Chat tab).

---

## Example Workflow

### One-off URL Ingestion
1. Go to 🔄 Auto-Update tab
2. Enter a URL (e.g., a course FAQ page)
3. Select "Customer Service" as target
4. Click **"Ingest once now"**
5. Go to 💬 Chat tab → ask a question about the newly added content

### Periodic Auto-Refresh
1. Register the same URL with "Register for periodic refresh"
2. Set interval to 60 minutes
3. Click **"▶ Start auto-refresh"**
4. When the source page updates, the next scheduled run detects the hash change and re-ingests automatically
