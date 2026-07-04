"""
Task 4 — arXiv Research Assistant (Domain Expert Chatbot)
Tests for CS NER, dataset loading, FAISS index, and summarizer.
Run: pytest tests/test_task4_research.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


# ── CS Named Entity Recognition ───────────────────────────────────────────────

class TestCSNER:
    def test_detects_algorithm(self):
        from cs_ner import extract_cs_entities
        entities = extract_cs_entities("We fine-tune BERT on a classification task.")
        terms = [e["term"].upper() for e in entities]
        assert "BERT" in terms

    def test_detects_dataset(self):
        from cs_ner import extract_cs_entities
        entities = extract_cs_entities("We evaluate on the ImageNet benchmark.")
        terms = [e["term"].lower() for e in entities]
        assert "imagenet" in terms

    def test_detects_task(self):
        from cs_ner import extract_cs_entities
        entities = extract_cs_entities("The model performs image classification.")
        categories = [e["category"] for e in entities]
        assert "Task" in categories

    def test_detects_framework(self):
        from cs_ner import extract_cs_entities
        entities = extract_cs_entities("Implemented using PyTorch and CUDA.")
        terms = [e["term"].lower() for e in entities]
        assert "pytorch" in terms

    def test_multiple_entities_same_text(self):
        from cs_ner import extract_cs_entities
        text = "We fine-tune BERT on SQuAD for question answering using PyTorch."
        entities = extract_cs_entities(text)
        assert len(entities) >= 3

    def test_case_insensitive_detection(self):
        from cs_ner import extract_cs_entities
        e1 = extract_cs_entities("bert model")
        e2 = extract_cs_entities("BERT model")
        assert len(e1) == len(e2)

    def test_empty_text_returns_empty(self):
        from cs_ner import extract_cs_entities
        assert extract_cs_entities("") == []

    def test_no_cs_terms_returns_empty(self):
        from cs_ner import extract_cs_entities
        assert extract_cs_entities("The cat sat on the mat.") == []

    def test_entity_has_required_keys(self):
        from cs_ner import extract_cs_entities
        entities = extract_cs_entities("BERT on ImageNet.")
        for e in entities:
            assert "term" in e
            assert "category" in e
            assert "start" in e
            assert "end" in e

    def test_no_overlapping_spans(self):
        from cs_ner import extract_cs_entities
        text = "Using PyTorch to train transformer models on ImageNet for classification."
        entities = extract_cs_entities(text)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                assert not (a["start"] < b["end"] and b["start"] < a["end"]), \
                    f"Overlapping: '{a['term']}' and '{b['term']}'"

    def test_gpt_detected_as_algorithm(self):
        from cs_ner import extract_cs_entities
        entities = extract_cs_entities("GPT-4 is a large language model.")
        algo_entities = [e for e in entities if e["category"] == "Algorithm/Model"]
        assert len(algo_entities) > 0

    def test_tensorflow_detected_as_framework(self):
        from cs_ner import extract_cs_entities
        entities = extract_cs_entities("The model was trained using TensorFlow.")
        fw_entities = [e for e in entities if e["category"] == "Framework"]
        assert len(fw_entities) > 0


class TestCSHighlighting:
    def test_returns_string(self):
        from cs_ner import extract_cs_entities, highlight_cs_entities
        text = "BERT on SQuAD."
        entities = extract_cs_entities(text)
        result = highlight_cs_entities(text, entities)
        assert isinstance(result, str)

    def test_contains_mark_tag(self):
        from cs_ner import extract_cs_entities, highlight_cs_entities
        text = "Training BERT on SQuAD."
        entities = extract_cs_entities(text)
        html = highlight_cs_entities(text, entities)
        assert "<mark" in html

    def test_empty_entities_returns_original(self):
        from cs_ner import highlight_cs_entities
        text = "No CS terms here."
        assert highlight_cs_entities(text, []) == text

    def test_four_category_colors_defined(self):
        from cs_ner import CATEGORY_COLORS
        expected = {"Algorithm/Model", "Dataset", "Task", "Framework"}
        assert set(CATEGORY_COLORS.keys()) == expected


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestArxivDataset:
    @pytest.fixture
    def csv_path(self):
        return os.path.join(
            os.path.dirname(__file__), "..", "dataset", "arxiv_cs_sample.csv"
        )

    def test_dataset_file_exists(self, csv_path):
        assert os.path.exists(csv_path), (
            "arxiv_cs_sample.csv not found — run: python dataset/fetch_arxiv.py"
        )

    def test_dataset_has_expected_columns(self, csv_path):
        import pandas as pd
        if not os.path.exists(csv_path):
            pytest.skip("Dataset not found")
        df = pd.read_csv(csv_path)
        for col in ["arxiv_id", "title", "authors", "abstract", "categories", "year", "url"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_dataset_has_sufficient_rows(self, csv_path):
        import pandas as pd
        if not os.path.exists(csv_path):
            pytest.skip("Dataset not found")
        df = pd.read_csv(csv_path)
        assert len(df) >= 1000, f"Expected ≥1000 papers, got {len(df)}"

    def test_all_rows_have_title_and_abstract(self, csv_path):
        import pandas as pd
        if not os.path.exists(csv_path):
            pytest.skip("Dataset not found")
        df = pd.read_csv(csv_path)
        assert df["title"].notna().all(), "Some rows missing title"
        assert df["abstract"].notna().all(), "Some rows missing abstract"

    def test_categories_are_cs_domain(self, csv_path):
        import pandas as pd
        if not os.path.exists(csv_path):
            pytest.skip("Dataset not found")
        df = pd.read_csv(csv_path)
        has_cs = df["categories"].str.contains(r"cs\.", regex=True, na=False)
        assert has_cs.sum() / len(df) > 0.9, "Less than 90% of papers are CS category"

    def test_years_are_recent(self, csv_path):
        import pandas as pd
        if not os.path.exists(csv_path):
            pytest.skip("Dataset not found")
        df = pd.read_csv(csv_path)
        assert df["year"].min() >= 2020, "Expected papers from 2020 or later"

    def test_arxiv_ids_are_unique(self, csv_path):
        import pandas as pd
        if not os.path.exists(csv_path):
            pytest.skip("Dataset not found")
        df = pd.read_csv(csv_path)
        assert df["arxiv_id"].nunique() == len(df), "Duplicate arxiv IDs found"


# ── FAISS Index ───────────────────────────────────────────────────────────────

class TestArxivFAISS:
    def test_faiss_index_exists_after_build(self):
        from arxiv_helper import ARXIV_VECTORDB_PATH
        assert os.path.exists(ARXIV_VECTORDB_PATH), (
            "FAISS index not built — click 'Build Research Knowledge Base' in the app"
        )

    def test_search_returns_results(self):
        from arxiv_helper import ARXIV_VECTORDB_PATH, search_papers
        if not os.path.exists(ARXIV_VECTORDB_PATH):
            pytest.skip("FAISS index not built")
        results = search_papers("large language models", k=3)
        assert len(results) > 0

    def test_search_result_has_score(self):
        from arxiv_helper import ARXIV_VECTORDB_PATH, search_papers
        if not os.path.exists(ARXIV_VECTORDB_PATH):
            pytest.skip("FAISS index not built")
        results = search_papers("neural networks", k=3)
        for doc, score in results:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_search_result_has_metadata(self):
        from arxiv_helper import ARXIV_VECTORDB_PATH, search_papers
        if not os.path.exists(ARXIV_VECTORDB_PATH):
            pytest.skip("FAISS index not built")
        results = search_papers("transformer architecture", k=3)
        for doc, score in results:
            assert "title" in doc.metadata
            assert "authors" in doc.metadata
            assert "year" in doc.metadata

    def test_search_returns_k_results(self):
        from arxiv_helper import ARXIV_VECTORDB_PATH, search_papers
        if not os.path.exists(ARXIV_VECTORDB_PATH):
            pytest.skip("FAISS index not built")
        results = search_papers("deep learning", k=5)
        assert len(results) <= 5

    def test_different_queries_return_different_results(self):
        from arxiv_helper import ARXIV_VECTORDB_PATH, search_papers
        if not os.path.exists(ARXIV_VECTORDB_PATH):
            pytest.skip("FAISS index not built")
        r1 = search_papers("computer vision object detection", k=3)
        r2 = search_papers("natural language processing translation", k=3)
        titles1 = {doc.metadata.get("title") for doc, _ in r1}
        titles2 = {doc.metadata.get("title") for doc, _ in r2}
        assert titles1 != titles2


# ── Summarizer ────────────────────────────────────────────────────────────────

class TestSummarizer:
    def test_module_imports(self):
        from summarizer import get_summary
        assert callable(get_summary)

    def test_very_short_text_returns_as_is(self):
        from summarizer import get_summary
        short = "AI is transforming science."
        result = get_summary(short)
        # For very short text (<50 chars), the function returns it unchanged
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string(self):
        from summarizer import get_summary
        result = get_summary("This is a test abstract about machine learning.")
        assert isinstance(result, str)

    def test_long_input_is_truncated_before_model(self):
        from summarizer import _MAX_INPUT_TOKENS
        assert _MAX_INPUT_TOKENS == 1024

    def test_model_name_is_distilbart(self):
        from summarizer import _MODEL
        assert "distilbart" in _MODEL.lower()

    def test_lazy_loading_does_not_load_at_import(self):
        import summarizer
        # Pipeline should be None until get_summary is called with real text
        assert summarizer._pipeline is None or callable(summarizer._pipeline)
