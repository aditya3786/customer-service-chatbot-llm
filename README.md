# Customer Service Chatbot with Sentiment Analysis

An end-to-end LLM-powered customer service chatbot built with **Google Gemini**, **LangChain**, and **Streamlit**. Extended with **sentiment analysis** (Task 1 of the ElevanceSkills internship) to detect and respond appropriately to customer emotions.

---

## Problem Statement

Customer support teams at e-learning platforms like Nullclass receive hundreds of repetitive queries daily. Two key challenges:

1. **Volume** — human staff can't respond instantly at scale.
2. **Tone mismatch** — a frustrated customer gets the same robotic response as a happy one, reducing satisfaction.

This project addresses both by combining a RAG-based Q&A system with real-time sentiment detection that adapts the chatbot's tone to the customer's emotional state.

---

## Dataset

- **Source**: `dataset/dataset.csv` — a real FAQ sheet used by Nullclass support staff
- **Size**: 76 rows, 2 columns (`prompt`, `response`)
- **Topics**: Course eligibility, pricing, EMI, tools (Power BI, Tableau), internships, certificates
- **Encoding**: Windows-1252 (cp1252) — handled in the data loader

---

## Methodology

### 1. RAG Pipeline (Retrieval-Augmented Generation)

```
User Question
    ↓
HuggingFace Embeddings (all-MiniLM-L6-v2, 384-dim)
    ↓
FAISS Vector Store (cosine similarity, threshold=0.7)
    ↓
Top-k matching FAQ chunks
    ↓
Gemini 2.5 Flash (prompt = context + sentiment instruction + question)
    ↓
Answer grounded in FAQ data
```

**Why RAG over fine-tuning?**
- No retraining needed when FAQs update — just rebuild the FAISS index
- Answers are grounded in source data, reducing hallucinations
- "I don't know" fallback for out-of-scope questions

### 2. Sentiment Analysis (Task 1)

**Model chosen: VADER (Valence Aware Dictionary and sEntiment Reasoner)**

| Approach | Accuracy | Size | Latency | API Cost |
|----------|----------|------|---------|----------|
| VADER (chosen) | Good for conversational text | ~2 MB | <1 ms | Free, offline |
| Transformer (e.g. RoBERTa) | Higher | ~500 MB | ~100 ms | Free, offline |
| Gemini API | Highest | N/A | ~1 s | Uses quota |

VADER was selected because it is purpose-built for short conversational text, runs entirely offline with zero latency overhead, and requires no additional model download.

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

---

## Features

- **Sentiment badge** with confidence score shown per query
- **Empathy/warmth message** for negative/positive queries
- **Tone-adapted LLM responses** via prompt injection
- **👍/👎 feedback buttons** per response
- **📊 Analytics dashboard** (separate tab):
  - Sentiment distribution pie chart
  - Confidence score bar chart per query
  - Full query history table
  - Session satisfaction %

---

## Project Structure

```
customer_service_chatbot_LLM/
├── dataset/
│   └── dataset.csv              # 76-row FAQ dataset
├── src/
│   ├── main.py                  # Streamlit UI + analytics dashboard
│   ├── langchain_helper.py      # RAG pipeline (FAISS + Gemini)
│   └── sentiment_analyzer.py   # VADER sentiment detection
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

1. Click **"Create Knowledgebase"** on first run (builds FAISS index from CSV — ~10 seconds)
2. Type a question in the input box
3. See the sentiment badge + confidence score
4. Read the tone-adapted answer
5. Click 👍 or 👎 to rate the response
6. Switch to the **📊 Analytics** tab to view sentiment trends

### Sample Questions

| Sentiment | Question |
|-----------|----------|
| Negative | *"This is terrible, I can't find anything useful!"* |
| Positive | *"I love this course, it's absolutely amazing!"* |
| Neutral | *"Do you have a JavaScript course?"* |
| Neutral | *"Should I learn Power BI or Tableau?"* |
| Neutral | *"I've a MAC computer. Can I use Power BI on it?"* |

---

## Internship Tasks Completed

| Task | Feature | Status |
|------|---------|--------|
| Task 1 | Sentiment Analysis Integration | ✅ Complete |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Orchestration | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Sentiment | NLTK VADER |
| UI | Streamlit |
| Visualizations | Plotly |
