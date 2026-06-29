# Customer Service Chatbot with Sentiment Analysis & Medical Q&A

An end-to-end LLM-powered chatbot built with **Google Gemini**, **LangChain**, and **Streamlit** as part of the ElevanceSkills internship. Extends the training project with two internship tasks: sentiment-aware customer service and a medical Q&A system.

---

## Problem Statement

Customer support teams at e-learning platforms like Nullclass receive hundreds of repetitive queries daily. Two key challenges:

1. **Volume** — human staff can't respond instantly at scale.
2. **Tone mismatch** — a frustrated customer gets the same robotic response as a happy one, reducing satisfaction.

This project addresses both by combining a RAG-based Q&A system with real-time sentiment detection. It is further extended with a specialized medical Q&A chatbot using NIH's MedQuAD dataset.

---

## Datasets

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| `dataset/dataset.csv` | Nullclass FAQ sheet | 76 rows | Customer service Q&A |
| `dataset/medquad.csv` | [MedQuAD (NIH)](https://github.com/abachaa/MedQuAD) | 5,068 rows | Medical Q&A |

---

## Methodology

### RAG Pipeline (shared by both chatbots)

```
User Question
    ↓
HuggingFace Embeddings (all-MiniLM-L6-v2, 384-dim, offline)
    ↓
FAISS Vector Store (cosine similarity, threshold=0.7)
    ↓
Top-k matching chunks as context
    ↓
Gemini 2.5 Flash (context + question → grounded answer)
```

**Why RAG over fine-tuning?**
- No retraining when data updates — just rebuild the FAISS index
- Answers grounded in source data, reducing hallucinations
- "I don't know" fallback for out-of-scope questions

---

## Task 1 — Sentiment Analysis Integration

**Objective:** Detect customer emotions (positive/negative/neutral) and respond with appropriate tone.

**Model: VADER (Valence Aware Dictionary and sEntiment Reasoner)**

| Approach | Size | Latency | API Cost |
|----------|------|---------|----------|
| VADER *(chosen)* | ~2 MB | <1 ms | Free, offline |
| Transformer (RoBERTa) | ~500 MB | ~100 ms | Free, offline |
| Gemini API | N/A | ~1 s | Uses quota |

VADER was selected because it is purpose-built for short conversational text, runs entirely offline, and requires no additional model download.

**Sentiment thresholds (VADER standard):**
- `compound ≥ 0.05` → Positive
- `compound ≤ -0.05` → Negative
- Otherwise → Neutral

**Tone adaptation via prompt engineering:**

| Sentiment | Injected instruction |
|-----------|---------------------|
| Negative | "Start by acknowledging their concern with empathy, then provide a clear and helpful answer." |
| Positive | "Match their positive energy with a warm, encouraging tone." |
| Neutral | *(no instruction — standard factual response)* |

**Features:**
- Color-coded sentiment badge (😊/😟/😐) with confidence score
- Empathy/warmth message for negative/positive queries
- Tone-adapted LLM responses via `PromptTemplate.partial()`
- 👍/👎 feedback buttons with session satisfaction tracking
- 📊 Analytics dashboard — sentiment pie chart, confidence bar chart, query history

**Sample Questions:**

| Sentiment | Question |
|-----------|----------|
| Negative | *"This is terrible, I can't find anything useful!"* |
| Positive | *"I love this course, it's absolutely amazing!"* |
| Neutral | *"Do you have a JavaScript course?"* |
| Neutral | *"Should I learn Power BI or Tableau?"* |

---

## Task 2 — Medical Q&A Chatbot (MedQuAD)

**Objective:** Build a specialized medical Q&A chatbot with retrieval and basic medical entity recognition.

**Dataset:** MedQuAD — 5,068 Q&A pairs parsed from 5 NIH sources:

| Source | Topic | Pairs |
|--------|-------|-------|
| CancerGov | Cancer types & treatments | ~729 |
| GHR (Genetics Home Reference) | Genetic conditions | ~1,500 |
| NIDDK | Diabetes, kidney, digestive | ~1,192 |
| NINDS | Neurological disorders | ~1,088 |
| NHLBI | Heart, lung, blood diseases | ~559 |

**Medical NER (Named Entity Recognition):**
Keyword/regex-based entity detection — no heavy model required, runs fully offline.

| Category | Color | Examples |
|----------|-------|---------|
| Disease/Condition | 🔴 Red | leukemia, diabetes, alzheimer |
| Symptom | 🟠 Orange | fever, headache, fatigue |
| Treatment | 🟢 Green | chemotherapy, insulin, surgery |

**Features:**
- "Build Medical Knowledge Base" button — embeds 5,068 docs into a separate FAISS index
- Real-time entity highlighting in the question text
- Entity badges showing detected terms and their category
- RAG answer sourced from NIH data
- Source document attribution (expandable)
- ⚠️ Medical safety disclaimer on every response

**Sample Questions:**

| Source | Question |
|--------|----------|
| CancerGov | *"What are the symptoms of leukemia?"* |
| NINDS | *"How is Alzheimer's disease treated?"* |
| NIDDK | *"What is diabetes?"* |
| NHLBI | *"What causes emphysema?"* |
| Multi-entity | *"What causes fever and headache in pneumonia?"* |

---

## Project Structure

```
customer_service_chatbot_LLM/
├── dataset/
│   ├── dataset.csv              # 76-row Nullclass FAQ dataset
│   ├── medquad.csv              # 5,068-row medical Q&A dataset
│   └── parse_medquad.py         # XML parser to regenerate medquad.csv
├── src/
│   ├── main.py                  # Streamlit UI (3 tabs: Chat, Medical Q&A, Analytics)
│   ├── langchain_helper.py      # Customer service RAG pipeline
│   ├── sentiment_analyzer.py    # VADER sentiment detection (Task 1)
│   ├── medical_helper.py        # Medical RAG pipeline (Task 2)
│   └── medical_ner.py           # Medical entity recognition (Task 2)
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com/app/apikey))

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/aditya3786/customer-service-chatbot-llm.git
cd customer-service-chatbot-llm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file in src/
echo 'GOOGLE_API_KEY="your_key_here"' > src/.env

# 4. Run the app
cd src
USE_TF=0 USE_JAX=0 streamlit run main.py
```

---

## Usage

### 💬 Chat Tab (Customer Service)
1. Click **"Create Knowledgebase"** on first run (~10 seconds)
2. Ask a question → see sentiment badge + tone-adapted answer
3. Rate with 👍/👎 — satisfaction % updates live

### 🏥 Medical Q&A Tab
1. Click **"Build Medical Knowledge Base"** on first run (~60 seconds)
2. Ask a medical question → entities highlighted + NIH-sourced answer
3. Expand **"Source documents"** to see retrieved passages

### 📊 Analytics Tab
- Sentiment distribution pie chart
- Confidence score bar chart per query
- Full query history table

---

## Internship Tasks Completed

| Task | Feature | Status |
|------|---------|--------|
| Task 1 | Sentiment Analysis Integration | ✅ Complete |
| Task 2 | Medical Q&A Chatbot (MedQuAD) | ✅ Complete |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Orchestration | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Sentiment Analysis | NLTK VADER |
| Medical NER | Keyword/regex dictionary |
| UI | Streamlit |
| Visualizations | Plotly |
