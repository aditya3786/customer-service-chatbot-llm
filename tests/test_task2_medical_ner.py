"""
Task 2 — Medical Q&A Chatbot (MedQuAD)
Tests for medical NER entity extraction and HTML highlighting.
Run: pytest tests/test_task2_medical_ner.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from medical_ner import extract_entities, highlight_entities, MEDICAL_ENTITIES, CATEGORY_COLORS


# ── Entity Extraction ─────────────────────────────────────────────────────────

class TestEntityExtraction:
    def test_detects_disease(self):
        entities = extract_entities("The patient was diagnosed with leukemia.")
        terms = [e["term"].lower() for e in entities]
        assert "leukemia" in terms

    def test_detects_symptom(self):
        entities = extract_entities("She presented with fever and severe headache.")
        terms = [e["term"].lower() for e in entities]
        assert any(t in terms for t in ["fever", "headache"])

    def test_detects_treatment(self):
        entities = extract_entities("Treatment includes chemotherapy and surgery.")
        terms = [e["term"].lower() for e in entities]
        assert any(t in terms for t in ["chemotherapy", "surgery"])

    def test_detects_multiple_categories(self):
        text = "Diabetes is treated with insulin therapy and diet control."
        entities = extract_entities(text)
        categories = {e["category"] for e in entities}
        assert len(categories) >= 2

    def test_case_insensitive(self):
        entities_lower = extract_entities("diabetes mellitus")
        entities_upper = extract_entities("DIABETES MELLITUS")
        assert len(entities_lower) == len(entities_upper)

    def test_empty_string_returns_empty(self):
        assert extract_entities("") == []

    def test_no_medical_terms_returns_empty(self):
        entities = extract_entities("The weather is nice today.")
        assert entities == []

    def test_entity_has_required_keys(self):
        entities = extract_entities("Patient has fever.")
        assert len(entities) > 0
        for e in entities:
            assert "term" in e
            assert "category" in e
            assert "start" in e
            assert "end" in e

    def test_entity_positions_are_valid(self):
        text = "Patient has fever and diabetes."
        entities = extract_entities(text)
        for e in entities:
            assert e["start"] >= 0
            assert e["end"] <= len(text)
            assert e["start"] < e["end"]

    def test_no_overlapping_spans(self):
        text = "Alzheimer's disease causes memory loss and confusion."
        entities = extract_entities(text)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                assert not (a["start"] < b["end"] and b["start"] < a["end"]), \
                    f"Overlapping: '{a['term']}' and '{b['term']}'"

    def test_alzheimers_is_disease(self):
        entities = extract_entities("Alzheimer's disease is a progressive neurological disorder.")
        disease_entities = [e for e in entities if e["category"] == "Disease/Condition"]
        assert len(disease_entities) > 0

    def test_fatigue_is_symptom(self):
        entities = extract_entities("Symptoms include fatigue and nausea.")
        symptom_entities = [e for e in entities if e["category"] == "Symptom"]
        assert len(symptom_entities) > 0

    def test_chemotherapy_is_treatment(self):
        entities = extract_entities("Chemotherapy is administered in cycles.")
        treatment_entities = [e for e in entities if e["category"] == "Treatment"]
        assert len(treatment_entities) > 0


# ── HTML Highlighting ─────────────────────────────────────────────────────────

class TestHighlighting:
    def test_returns_string(self):
        entities = extract_entities("Patient has fever.")
        html = highlight_entities("Patient has fever.", entities)
        assert isinstance(html, str)

    def test_contains_mark_tag(self):
        text = "Patient has fever."
        entities = extract_entities(text)
        html = highlight_entities(text, entities)
        assert "<mark" in html

    def test_empty_entities_returns_original(self):
        text = "The weather is nice."
        html = highlight_entities(text, [])
        assert html == text

    def test_highlighted_text_contains_entity_term(self):
        text = "Patient has leukemia."
        entities = extract_entities(text)
        html = highlight_entities(text, entities)
        assert "leukemia" in html.lower()

    def test_color_is_applied_in_html(self):
        text = "Patient has fever."
        entities = extract_entities(text)
        html = highlight_entities(text, entities)
        # Should have some color styling
        assert "background" in html or "color" in html or "style" in html


# ── Data Structures ───────────────────────────────────────────────────────────

class TestDataStructures:
    def test_medical_entities_has_three_categories(self):
        assert "Disease/Condition" in MEDICAL_ENTITIES
        assert "Symptom" in MEDICAL_ENTITIES
        assert "Treatment" in MEDICAL_ENTITIES

    def test_each_category_has_terms(self):
        for category, terms in MEDICAL_ENTITIES.items():
            assert len(terms) > 0, f"Category '{category}' has no terms"

    def test_category_colors_covers_all_categories(self):
        for category in MEDICAL_ENTITIES:
            assert category in CATEGORY_COLORS, f"No color for '{category}'"

    def test_dataset_exists(self):
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "dataset", "medquad.csv"
        )
        assert os.path.exists(csv_path), "medquad.csv not found — run parse_medquad.py"

    def test_dataset_has_expected_columns(self):
        import pandas as pd
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "dataset", "medquad.csv"
        )
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for col in ["question", "answer", "source"]:
                assert col in df.columns

    def test_dataset_has_rows(self):
        import pandas as pd
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "dataset", "medquad.csv"
        )
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            assert len(df) > 1000, "Expected at least 1000 medical Q&A pairs"
