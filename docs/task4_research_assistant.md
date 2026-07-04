# Task 4 — arXiv Research Assistant (Domain Expert Chatbot)

## Objective
Build a domain-expert chatbot on CS research papers with: advanced NLP (information extraction + abstractive summarization), an open-source LLM, semantic paper search, multi-turn conversation support, and concept visualization.

---

## Dataset — arXiv CS Papers

**Source:** [arXiv.org](https://arxiv.org) via the `arxiv` Python library (avoids downloading the 4GB Kaggle file)

| Category | Description | Papers Fetched |
|----------|-------------|----------------|
| cs.AI | Artificial intelligence | ~600 |
| cs.LG | Machine learning | ~421 |
| cs.CL | Computational linguistics / NLP | ~437 |
| cs.CV | Computer vision | ~431 |
| cs.NE | Neural and evolutionary computing | ~586 |
| **Total** | | **2,475** |

**Fetch script:** `dataset/fetch_arxiv.py`
```bash
python dataset/fetch_arxiv.py
# Runtime: ~5 minutes (API rate-limited)
# Output: dataset/arxiv_cs_sample.csv
```

**Dataset columns:** `arxiv_id`, `title`, `authors`, `abstract`, `categories`, `year`, `url`

---

## Methodology

### Open-Source LLM — DistilBART Summarization
**Model:** `sshleifer/distilbart-cnn-12-6` (HuggingFace Transformers)

| Property | Value |
|----------|-------|
| Size | ~300 MB |
| Inference | CPU (no GPU required) |
| Type | Abstractive summarization |
| License | Apache 2.0 (open-source) |
| API dependency | None — fully offline |

Lazy loading: the model is downloaded and cached only when the user first clicks "Summarize", keeping app startup fast.

```python
# summarizer.py
def get_summary(text, max_length=150, min_length=40):
    pipe = _get_pipeline()  # loads model on first call
    return pipe(text[:1024_words], max_length=max_length, ...)[0]["summary_text"]
```

### RAG with Conversation Memory
`ConversationalRetrievalChain` (LangChain) enables true follow-up questions:

```
Turn 1: "What are diffusion models?"
    → chain condenses to: "What are diffusion models?"
    → retrieves relevant papers → Gemini answers

Turn 2: "How are they used in image generation?"
    → chain condenses: "How are diffusion models used in image generation?"
    → correct follow-up, no need to repeat context
```

The `chat_history` list `[(human, ai), ...]` is stored in `st.session_state` and passed explicitly on each turn — no module-level state, one history per browser session.

### CS NER — Information Extraction
Four categories, same regex pattern as Task 2's medical NER:

| Category | Color | Example Terms |
|----------|-------|---------------|
| Algorithm/Model | 🔵 `#cce5ff` | BERT, GPT, transformer, GAN, diffusion model, ViT |
| Dataset | 🟢 `#d4edda` | ImageNet, SQuAD, GLUE, MS COCO, CIFAR-10 |
| Task | 🟡 `#fff3cd` | classification, detection, translation, generation |
| Framework | 🔴 `#f8d7da` | PyTorch, TensorFlow, JAX, HuggingFace |

### FAISS Index
- Each paper's `title + abstract` is a LangChain `Document`
- Split with `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)`
- **8,278 chunks** indexed from 2,475 papers
- Retrieval: `similarity_search_with_relevance_scores()` with `score_threshold=0.3`

### Concept Visualization (Plotly)
Three charts auto-generated from `arxiv_cs_sample.csv`:
1. **CS Subcategory Distribution** — horizontal bar chart of paper counts per `cs.*` category
2. **Papers by Year** — bar chart showing publication year distribution
3. **Top 25 Keywords** — word frequency from all abstracts after stopword removal

---

## Pipeline Flow
```
User query (Paper Search)
    ↓
FAISS similarity_search_with_relevance_scores(query, k=5)
    ↓
Top-5 papers rendered as cards
    ├── CS NER highlights in abstract
    └── "Summarize" → DistilBART → abstractive summary

User question (Research Q&A)
    ↓
ConversationalRetrievalChain(llm, retriever, chat_history)
    ├── LLM condenses follow-up with prior context
    ├── Retriever fetches relevant paper chunks
    └── Gemini generates grounded answer with paper citations
    ↓
Answer + source paper titles + updated chat history
```

---

## Files
| File | Role |
|------|------|
| `dataset/fetch_arxiv.py` | arXiv API fetcher — generates `arxiv_cs_sample.csv` |
| `dataset/arxiv_cs_sample.csv` | 2,475 CS papers |
| `src/arxiv_helper.py` | FAISS index builder + ConversationalRetrievalChain |
| `src/cs_ner.py` | CS domain entity extraction + HTML highlighting |
| `src/summarizer.py` | DistilBART abstractive summarization |
| `src/main.py` | 🔬 Research tab (search, Q&A, summarize, visualize) |

## Key Functions
```python
# arxiv_helper.py
create_arxiv_vector_db() -> int              # builds index, returns chunk count
search_papers(query, k=5) -> list[(doc, score)]
get_arxiv_qa_chain() -> ConversationalRetrievalChain

# cs_ner.py
extract_cs_entities(text) -> list[dict]
highlight_cs_entities(text, entities) -> str  # returns colored HTML

# summarizer.py
get_summary(text, max_length=150, min_length=40) -> str
```

---

## Test Results

### CS NER Tests
| Input | Expected | Pass |
|-------|----------|------|
| "fine-tune BERT on SQuAD" | BERT (Algorithm), SQuAD (Dataset) | ✅ |
| "using PyTorch for classification" | PyTorch (Framework), classification (Task) | ✅ |
| "GPT-4 is a large language model" | GPT-4 (Algorithm) | ✅ |
| "BERT BERT BERT" | 1 entity, no overlap | ✅ |
| Empty string | [] | ✅ |

### Dataset Tests
- `arxiv_cs_sample.csv` exists with 2,475 rows: ✅
- All 7 expected columns present: ✅
- All papers have title and abstract: ✅
- >90% of papers are `cs.*` category: ✅
- arxiv_id is unique per row: ✅

### FAISS Index Tests
- 8,278 chunks indexed: ✅
- Search returns results with scores ∈ [0.0, 1.0]: ✅
- Different queries return different top papers: ✅
- Each result has `title`, `authors`, `year` metadata: ✅

---

## How to Run Tests
```bash
pytest tests/test_task4_research.py -v
```
**Note:** FAISS index tests require "Build Research Knowledge Base" to be clicked in the app first.
Summarizer tests do **not** download the 300MB model (lazy load is tested separately).

---

## Sample Questions to Try in the App

### Paper Search
- `large language models reasoning`
- `diffusion models image synthesis`
- `vision transformers object detection`
- `federated learning privacy`

### Research Q&A (follow-up sequence)
| Turn | Question |
|------|----------|
| 1 | `What are diffusion models?` |
| 2 | `How do they generate images?` |
| 3 | `What makes them better than GANs?` |

| Turn | Question |
|------|----------|
| 1 | `Explain reinforcement learning from human feedback` |
| 2 | `How is it used to train ChatGPT?` |

---

## Architecture Decision Log
| Decision | Chosen | Reason |
|----------|--------|--------|
| Dataset source | arXiv API | Avoids 4GB Kaggle download; fully reproducible |
| Open-source LLM | DistilBART | 300MB, CPU-only, Apache 2.0, no API quota |
| Main QA LLM | Gemini 2.5 Flash | High quality grounded answers |
| Follow-up memory | ConversationalRetrievalChain | Native LangChain support, clean API |
| NER approach | Keyword/regex | No GPU, no model download, instant results |
