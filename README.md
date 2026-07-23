# AI Assistant Suite — LLM Internship Project

An end-to-end AI assistant platform built with **Google Gemini 2.5 Flash**, **LangChain**, and **Streamlit** as part of the ElevanceSkills internship. Covers 6 tasks: sentiment-aware customer service, medical Q&A, dynamic knowledge expansion, arXiv research, multi-modal visual reasoning, and multilingual conversations.

---

## Internship Tasks

| Task | Feature | Key Tech |
|------|---------|----------|
| **1** | Sentiment-aware customer service chat | VADER + Gemini RAG + FAISS |
| **2** | Medical Q&A with NER entity highlighting | MedQuAD (5,068 NIH pairs) + medical NER |
| **3** | Dynamic knowledge base expansion | URL ingestion + SHA-256 dedup + APScheduler |
| **4** | arXiv research assistant + summarization | DistilBART + ConversationalRetrievalChain |
| **5** | Multi-modal visual AI (4-stage pipeline) | Gemini Vision + evidence labeling + self-critique |
| **6** | Multilingual conversations (3-stage pipeline) | `langdetect` + cross-lingual context + 12 languages |

---

## Project Structure

```
customer_service_chatbot_LLM/
├── dataset/
│   ├── dataset.csv              # 76-row Nullclass FAQ (customer service)
│   ├── medquad.csv              # 5,068-row medical Q&A (NIH)
│   ├── arxiv_cs_sample.csv      # 2,475-row arXiv CS papers
│   ├── fetch_arxiv.py           # arXiv API fetcher
│   └── parse_medquad.py         # MedQuAD XML parser
├── src/
│   ├── main.py                  # Streamlit UI (7 tabs)
│   ├── langchain_helper.py      # Customer service RAG chain + shared LLM
│   ├── sentiment_analyzer.py    # VADER sentiment (Task 1)
│   ├── medical_helper.py        # Medical RAG pipeline (Task 2)
│   ├── medical_ner.py           # Medical entity recognition (Task 2)
│   ├── knowledge_expander.py    # URL ingestion + FAISS updates (Task 3)
│   ├── kb_scheduler.py          # Periodic refresh scheduler (Task 3)
│   ├── arxiv_helper.py          # arXiv RAG + conversation memory (Task 4)
│   ├── cs_ner.py                # CS domain NER (Task 4)
│   ├── summarizer.py            # DistilBART summarization (Task 4)
│   ├── multimodal_assistant.py  # 4-stage vision pipeline (Task 5)
│   └── multilingual_assistant.py# 3-stage multilingual pipeline (Task 6)
├── tests/
│   ├── test_task1_sentiment.py
│   ├── test_task2_medical_ner.py
│   ├── test_task3_knowledge_base.py
│   ├── test_task4_research.py
│   ├── test_task5_multimodal.py
│   └── test_task6_multilingual.py
├── docs/
│   ├── task1_sentiment_analysis.md
│   ├── task2_medical_qa.md
│   ├── task3_knowledge_expansion.md
│   ├── task4_research_assistant.md
│   ├── task5_multimodal_assistant.md
│   └── task6_multilingual_assistant.md
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- Google Gemini API key — free at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/aditya3786/customer-service-chatbot-llm.git
cd customer-service-chatbot-llm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo 'GOOGLE_API_KEY="your_key_here"' > src/.env

# 4. Run the app
cd src
USE_TF=0 USE_JAX=0 streamlit run main.py
```

The app opens at **http://localhost:8501**

---

## Usage

### 💬 Chat Tab — Customer Service (Task 1)
1. Click **"🗄 Build KB"** on first run (~10 s)
2. Type any customer service question
3. See sentiment badge (😊/😐/😟) + confidence score + tone-adapted answer
4. Rate with 👍/👎 — session satisfaction % updates live

### 🏥 Medical Q&A Tab (Task 2)
1. Click **"Build Medical Knowledge Base"** on first run (~60 s, embeds 5,068 NIH Q&A pairs)
2. Ask a medical question — detected entities are color-highlighted
3. View NIH-sourced answer + expandable source documents
4. Medical safety disclaimer shown on every response

### 🔄 Auto-Update Tab (Task 3)
1. Paste a URL or type a manual Q&A pair, choose target knowledge base
2. **"Ingest once now"** — immediate one-off ingestion with SHA-256 dedup
3. **"Register for periodic refresh"** — recurring ingestion on a configurable interval
4. Start/stop the background scheduler; view live vector counts and full ingestion history

### 🔬 Research Tab (Task 4)
1. Click **"Build Research Knowledge Base"** on first run (~90 s, indexes 2,475 papers)
2. Enter a semantic query → top-5 papers with CS entity highlighting
3. Click **"📄 Summarize"** → DistilBART generates a concise abstract summary (offline)
4. Ask a research question with follow-up support — conversation context is preserved per session
5. Scroll down for concept visualization charts (category distribution, year histogram, top keywords)

### 🖼️ Visual AI Tab (Task 5)
1. Upload a JPEG, PNG, or WebP image
2. Optionally type a question (or leave blank for a full image description)
3. Click **"🔍 Analyze"** — the 4-stage pipeline runs:
   - **Stage 1** — structured image analysis (scene type, objects, OCR, colors)
   - **Stage 2** — ambiguity detection (surfaces assumptions if the question is vague)
   - **Stage 3** — evidence-labelled answer (`[seen in image]` / `[inferred]` / `[general knowledge]`)
   - **Stage 4** — self-critique validation (confidence score 0–1, caveats, auto-correction)
4. Ask follow-up questions — the same image context is retained across turns

### 📊 Analytics Tab (Task 1)
- Live KPI tiles: total queries, average confidence, satisfaction %, positive/negative split
- Sentiment distribution pie chart + confidence bar chart per query
- Full query history table

### 🌐 Multilingual Tab (Task 6)
1. Type a message in **any language** (or mix languages)
2. Click **"Send"** — the 3-stage pipeline runs:
   - **Stage 1** — language detection (`langdetect`) + language-switch flag + mixed-language segmentation
   - **Stage 2** — cross-lingual context resolution using up to 5 prior turns across all languages
   - **Stage 3** — response generated entirely in your detected language
3. Language switch banners appear when you change languages mid-conversation
4. Prior context is always retained across language boundaries
5. "Language timeline" expander shows the language used per turn

**Supported languages:** 🇬🇧 English · 🇪🇸 Spanish · 🇫🇷 French · 🇮🇳 Hindi · 🇩🇪 German · 🇨🇳 Chinese · 🇯🇵 Japanese · 🇸🇦 Arabic · 🇵🇹 Portuguese · 🇮🇹 Italian · 🇰🇷 Korean · 🇷🇺 Russian

---

## Architecture

### Shared RAG Pipeline (Tasks 1–4)
```
User Question
    ↓
HuggingFace Embeddings (all-MiniLM-L6-v2, 384-dim, offline)
    ↓
FAISS Vector Store (cosine similarity, score threshold 0.7)
    ↓
Top-k matching chunks as context
    ↓
Gemini 2.5 Flash (context + question → grounded answer)
```

### Task 5 — 4-Stage Vision Pipeline
```
Image + Question → Gemini Vision (Stage 1: structured analysis)
                → Text LLM (Stage 2: ambiguity detection)
                → Gemini Vision (Stage 3: evidence-labelled response)
                → Text LLM (Stage 4: self-critique validation)
                → Final answer
```

### Task 6 — 3-Stage Multilingual Pipeline
```
Text (any language) → langdetect (Stage 1: language ID + switch detection)
                    → Text LLM (Stage 2: cross-lingual context resolution)
                    → Text LLM (Stage 3: response in user's language)
                    → Final answer
```

---

## Datasets

| Dataset | Source | Size | Used By |
|---------|--------|------|---------|
| `dataset/dataset.csv` | Nullclass FAQ | 76 rows | Task 1 |
| `dataset/medquad.csv` | MedQuAD (NIH) | 5,068 rows | Task 2 |
| `dataset/arxiv_cs_sample.csv` | arXiv API | 2,475 papers | Task 4 |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Orchestration | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (offline) |
| Vector Store | FAISS |
| Sentiment Analysis | NLTK VADER |
| Language Detection | `langdetect` (open-source, offline) |
| Medical / CS NER | Keyword + regex dictionary |
| Summarization (open-source LLM) | DistilBART — `sshleifer/distilbart-cnn-12-6` |
| Conversation Memory | LangChain ConversationalRetrievalChain |
| Scheduling | APScheduler |
| Web Scraping | Requests + BeautifulSoup4 |
| Image Handling | Pillow |
| UI | Streamlit |
| Visualizations | Plotly |

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Individual task
python -m pytest tests/test_task5_multimodal.py -v
python -m pytest tests/test_task6_multilingual.py -v
```

All LLM calls are mocked — no API quota is consumed by tests.
