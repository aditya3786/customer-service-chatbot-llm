# Task 1 — Sentiment Analysis Integration

## Objective
Detect the emotional tone of a customer's query (positive / negative / neutral) and automatically adapt the chatbot's response style to improve customer satisfaction.

## Evaluation Criteria
| Criterion | Approach |
|-----------|----------|
| Accuracy of sentiment detection | VADER compound score vs. ground truth labels |
| Appropriateness of response tone | Prompt engineering — distinct instruction per sentiment |
| Impact on customer satisfaction | Live 👍/👎 feedback with session satisfaction % |

---

## Methodology

### Sentiment Engine — VADER
**VADER (Valence Aware Dictionary and sEntiment Reasoner)** from NLTK was chosen over alternatives:

| Approach | Size | Latency | API Cost | Suitability |
|----------|------|---------|----------|-------------|
| **VADER** *(chosen)* | ~2 MB | <1 ms | Free, offline | Best for short conversational text |
| RoBERTa (transformer) | ~500 MB | ~100 ms | Free, offline | Overkill for this use case |
| Gemini API | — | ~1 s | Uses quota | Adds cost and latency |

VADER scores every word in a dictionary with a sentiment valence. It handles:
- Punctuation emphasis (`great!!!` > `great`)
- Capitalization (`GREAT` > `great`)
- Negations (`not good` → negative)
- Degree modifiers (`very bad` > `bad`)

### Thresholds
```
compound ≥  0.05  →  Positive
compound ≤ -0.05  →  Negative
otherwise         →  Neutral
```

### Tone Injection — `PromptTemplate.partial()`
Rather than building a separate chain or calling the API twice, a single `{sentiment_instruction}` placeholder is added to the RAG prompt. Before the chain runs, it is pre-filled via `PromptTemplate.partial()`:

```python
# langchain_helper.py
_SENTIMENT_INSTRUCTIONS = {
    "negative": "Start by acknowledging their concern with empathy, then provide a clear and helpful answer.",
    "positive": "Match their positive energy with a warm, encouraging tone.",
    "neutral":  "",
}

PROMPT = base_prompt.partial(sentiment_instruction=_SENTIMENT_INSTRUCTIONS[sentiment])
chain = RetrievalQA.from_chain_type(..., chain_type_kwargs={"prompt": PROMPT})
```

This means `RetrievalQA` still receives only `{context}` and `{question}` at runtime — no changes to the chain interface.

---

## Pipeline Flow
```
User question
    ↓
VADER SentimentIntensityAnalyzer (<1 ms, offline)
    ↓
label: positive | negative | neutral
    ↓
PromptTemplate.partial(sentiment_instruction=...)
    ↓
FAISS retriever (score_threshold=0.7) → top-k FAQ chunks
    ↓
Gemini 2.5 Flash (context + tone instruction + question)
    ↓
Tone-adapted answer
```

---

## Files
| File | Role |
|------|------|
| `src/sentiment_analyzer.py` | VADER wrapper — `analyze_sentiment(text)` |
| `src/langchain_helper.py` | Prompt with `{sentiment_instruction}` placeholder |
| `src/main.py` | UI: sentiment badge, empathy message, 👍/👎 feedback |

## Key Functions
```python
# sentiment_analyzer.py
analyze_sentiment(text: str) -> dict
# Returns: {"label": "positive"|"negative"|"neutral", "compound": float, "scores": dict}

# langchain_helper.py
get_qa_chain(sentiment: str = "neutral") -> RetrievalQA
# Returns a chain with tone instruction baked into the prompt
```

---

## UI Features
- **Color-coded badge**: 😊 Positive (green) / 😟 Negative (red) / 😐 Neutral (gray)
- **Confidence score**: VADER compound value displayed alongside badge
- **Empathy message**: Shown above the answer for negative/positive queries
- **Satisfaction tracker**: Session 👍/👎 ratio shown as live percentage

---

## Test Results

| Input | Expected Label | Detected Label | Pass |
|-------|----------------|----------------|------|
| "This is terrible, I can't find anything useful!" | negative | negative | ✅ |
| "I love this course, it's absolutely amazing!" | positive | positive | ✅ |
| "Do you have a JavaScript course?" | neutral | neutral | ✅ |
| "I've been waiting for days!" | negative | negative | ✅ |
| "Great support, very happy!" | positive | positive | ✅ |
| "What is the difference between Python 2 and 3?" | neutral | neutral | ✅ |

---

## How to Run Tests
```bash
cd customer-service-chatbot-llm
pytest tests/test_task1_sentiment.py -v
```

**Expected output:** All 22 tests pass. No API calls required — VADER runs entirely offline.

---

## Sample Questions to Try in the App

| Sentiment | Question |
|-----------|----------|
| Negative | *"This is frustrating, nothing works!"* |
| Positive | *"I love this platform, it's amazing!"* |
| Neutral | *"Should I learn Power BI or Tableau?"* |
| Negative | *"Why is my access blocked?"* |
| Positive | *"The instructor explains things perfectly!"* |
