# Test Suite

Unit and integration tests for all four internship tasks.

## Prerequisites
```bash
pip install pytest
# All other dependencies already in requirements.txt
```

## Run All Tests
```bash
cd customer-service-chatbot-llm
pytest tests/ -v
```

## Run by Task
```bash
pytest tests/test_task1_sentiment.py -v       # Task 1 — Sentiment Analysis
pytest tests/test_task2_medical_ner.py -v     # Task 2 — Medical NER
pytest tests/test_task3_knowledge_base.py -v  # Task 3 — Knowledge Expansion
pytest tests/test_task4_research.py -v        # Task 4 — Research Assistant
```

## Notes
- **No API key needed** for Tasks 1, 2, 4 (NER/sentiment are fully offline)
- Task 3 dedup test requires the Customer Service FAISS index (click "Create Knowledgebase" first)
- Task 4 FAISS tests require the Research index (click "Build Research Knowledge Base" first)
- Task 4 summarizer tests do **not** download the 300MB DistilBART model

## Test Count by Task
| Task | Tests | API Required |
|------|-------|-------------|
| Task 1 | 22 | No |
| Task 2 | 23 | No |
| Task 3 | 18 | No |
| Task 4 | 26 | No |
| **Total** | **89** | |
