# Task 2 — Medical Q&A Chatbot (MedQuAD)

## Objective
Build a specialized medical Q&A chatbot using NIH's MedQuAD dataset, with retrieval-augmented generation, basic medical entity recognition (NER), and a safety disclaimer on every response.

---

## Dataset — MedQuAD

**Source:** [MedQuAD GitHub](https://github.com/abachaa/MedQuAD) — NIH public health data

| NIH Source | Topic | Q&A Pairs |
|------------|-------|-----------|
| CancerGov | Cancer types & treatments | ~729 |
| GHR (Genetics Home Reference) | Genetic conditions | ~1,500 |
| NIDDK | Diabetes, kidney, digestive | ~1,192 |
| NINDS | Neurological disorders | ~1,088 |
| NHLBI | Heart, lung, blood diseases | ~559 |
| **Total** | | **5,068** |

**Parsing:** The raw dataset is distributed as XML files (one file per health topic). `dataset/parse_medquad.py` walks 5 NIH source folders, extracts `<QAPair>` elements, and writes to `dataset/medquad.csv`.

---

## Methodology

### RAG Pipeline
Reuses the exact FAISS + HuggingFace Embeddings + Gemini pattern from Task 1, with:
- **Separate FAISS index:** `src/medical_faiss_index/` — isolated from the customer service index
- **Lower score threshold:** `score_threshold=0.7` (stricter, to avoid unrelated medical advice)
- **Custom prompt:** Instructs Gemini to answer only from NIH context and cite the source

### Medical NER
Keyword/regex-based entity recognition — no GPU, no model download, fully offline.

| Category | Color | Example Terms |
|----------|-------|---------------|
| Disease/Condition | 🔴 Red `#ffcccc` | leukemia, diabetes, alzheimer's, asthma |
| Symptom | 🟠 Orange `#ffe0b2` | fever, headache, fatigue, nausea |
| Treatment | 🟢 Green `#c8e6c9` | chemotherapy, insulin, surgery, dialysis |

**Implementation:**
1. Build word-boundary regex for each term: `re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)`
2. Match all terms against the input text
3. Track covered character spans to prevent overlapping highlights
4. Return `[{term, category, start, end}]` sorted by position

---

## Pipeline Flow
```
Medical question
    ↓
CS NER → extract_entities(question) → [{term, category, start, end}]
    ↓
Entity badges shown in UI (Disease/Condition / Symptom / Treatment)
    ↓
highlight_entities(question, entities) → colored HTML
    ↓
medical_faiss_index retriever (score_threshold=0.7) → top-k NIH passages
    ↓
Gemini 2.5 Flash (NIH context + question)
    ↓
Answer + source document attribution + safety disclaimer
```

---

## Files
| File | Role |
|------|------|
| `dataset/parse_medquad.py` | XML → CSV parser (run once to regenerate) |
| `dataset/medquad.csv` | 5,068 NIH Q&A pairs |
| `src/medical_ner.py` | Entity extraction and HTML highlighting |
| `src/medical_helper.py` | FAISS index builder + retrieval chain |
| `src/main.py` | 🏥 Medical Q&A tab UI |

## Key Functions
```python
# medical_ner.py
extract_entities(text: str) -> list[dict]
highlight_entities(text: str, entities: list[dict]) -> str  # returns HTML

# medical_helper.py
create_medical_vector_db()  # embeds 5,068 docs → saves medical_faiss_index/
get_medical_qa_chain() -> RetrievalQA  # loads index, returns chain
```

---

## Safety Design
Every response includes:
> ⚠️ **Medical Disclaimer:** This information is for educational purposes only and is sourced from NIH public databases. Always consult a qualified healthcare provider for personal medical advice, diagnosis, or treatment.

The LLM prompt explicitly instructs Gemini:
- Answer only from the provided NIH context
- Do not fabricate medical information
- Add the disclaimer to every response

---

## Test Results

### NER Tests
| Input | Expected Entity | Category | Pass |
|-------|----------------|----------|------|
| "diagnosed with leukemia" | leukemia | Disease/Condition | ✅ |
| "fever and headache" | fever, headache | Symptom | ✅ |
| "treated with chemotherapy" | chemotherapy | Treatment | ✅ |
| "DIABETES MELLITUS" | diabetes mellitus | Disease/Condition | ✅ |
| "The weather is nice today" | (none) | — | ✅ |

### Dataset Tests
- `medquad.csv` exists: ✅
- Contains columns `question`, `answer`, `source`: ✅
- Row count ≥ 5,000: ✅

---

## How to Run Tests
```bash
pytest tests/test_task2_medical_ner.py -v
```
**All 23 tests pass.** No API calls — NER is regex-based, fully offline.

---

## Sample Questions to Try in the App

| Question | Source | Entities Detected |
|----------|--------|-------------------|
| *"What are the symptoms of leukemia?"* | CancerGov | leukemia (Disease), symptoms (→ in answer) |
| *"How is Alzheimer's disease treated?"* | NINDS | Alzheimer's (Disease), treatments in answer |
| *"What is diabetes?"* | NIDDK | diabetes (Disease) |
| *"What causes emphysema?"* | NHLBI | emphysema (Disease) |
| *"What causes fever and headache in pneumonia?"* | Multi | pneumonia (Disease), fever + headache (Symptom) |

---

## How to Rebuild the Dataset
```bash
# Clone MedQuAD XML source
git clone https://github.com/abachaa/MedQuAD dataset/MedQuAD

# Parse XML → CSV
python dataset/parse_medquad.py
# Output: dataset/medquad.csv (5,068 rows)
```
