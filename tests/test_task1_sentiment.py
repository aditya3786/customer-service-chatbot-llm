"""
Task 1 — Sentiment Analysis Integration
Tests for VADER sentiment detection and prompt tone injection.
Run: pytest tests/test_task1_sentiment.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from sentiment_analyzer import analyze_sentiment


# ── VADER Sentiment Detection ─────────────────────────────────────────────────

class TestSentimentLabels:
    def test_clearly_negative_query(self):
        result = analyze_sentiment("This is terrible, I can't find anything useful!")
        assert result["label"] == "negative"

    def test_clearly_positive_query(self):
        result = analyze_sentiment("I love this course, it's absolutely amazing!")
        assert result["label"] == "positive"

    def test_neutral_factual_query(self):
        result = analyze_sentiment("Do you have a JavaScript course?")
        assert result["label"] == "neutral"

    def test_frustrated_customer(self):
        result = analyze_sentiment("This is so frustrating! Nobody helped me and I'm very upset!")
        assert result["label"] == "negative"

    def test_satisfied_customer(self):
        result = analyze_sentiment("Great support, very happy with the service!")
        assert result["label"] == "positive"

    def test_technical_question(self):
        result = analyze_sentiment("What is the difference between Python 2 and Python 3?")
        assert result["label"] == "neutral"


class TestSentimentOutput:
    def test_returns_dict_with_required_keys(self):
        result = analyze_sentiment("Hello")
        assert "label" in result
        assert "compound" in result
        assert "scores" in result

    def test_label_is_valid(self):
        for text in ["great!", "terrible!", "okay"]:
            result = analyze_sentiment(text)
            assert result["label"] in ("positive", "negative", "neutral")

    def test_compound_in_range(self):
        result = analyze_sentiment("This is fine.")
        assert -1.0 <= result["compound"] <= 1.0

    def test_scores_has_vader_keys(self):
        result = analyze_sentiment("Test text")
        assert "pos" in result["scores"]
        assert "neg" in result["scores"]
        assert "neu" in result["scores"]
        assert "compound" in result["scores"]

    def test_positive_threshold(self):
        result = analyze_sentiment("I love this! Amazing and wonderful!")
        assert result["compound"] >= 0.05

    def test_negative_threshold(self):
        result = analyze_sentiment("Absolutely terrible. Worst experience ever!")
        assert result["compound"] <= -0.05

    def test_empty_string(self):
        result = analyze_sentiment("")
        assert result["label"] == "neutral"
        assert result["compound"] == 0.0

    def test_single_word_positive(self):
        result = analyze_sentiment("excellent")
        assert result["label"] == "positive"

    def test_single_word_negative(self):
        result = analyze_sentiment("awful")
        assert result["label"] == "negative"


class TestSentimentThresholds:
    """VADER standard thresholds: >=0.05 positive, <=-0.05 negative"""

    def test_borderline_neutral_positive(self):
        result = analyze_sentiment("okay")
        # Should be neutral — compound close to 0
        assert -0.3 <= result["compound"] <= 0.3

    def test_strong_negative(self):
        result = analyze_sentiment("I hate this! It's broken and useless!")
        assert result["compound"] < -0.05

    def test_strong_positive(self):
        result = analyze_sentiment("Fantastic! I'm absolutely thrilled!")
        assert result["compound"] > 0.05


# ── Prompt Template Injection ─────────────────────────────────────────────────

class TestPromptTemplate:
    """Verify sentiment instructions are correctly wired into the prompt template."""

    def test_sentiment_instructions_dict_exists(self):
        import langchain_helper
        assert hasattr(langchain_helper, "_SENTIMENT_INSTRUCTIONS")

    def test_all_sentiments_covered(self):
        import langchain_helper
        instructions = langchain_helper._SENTIMENT_INSTRUCTIONS
        assert "positive" in instructions
        assert "negative" in instructions
        assert "neutral" in instructions

    def test_negative_instruction_is_empathetic(self):
        import langchain_helper
        neg_instruction = langchain_helper._SENTIMENT_INSTRUCTIONS["negative"]
        # Should mention empathy or acknowledgment
        assert any(word in neg_instruction.lower() for word in ["empathy", "acknowledg", "frustrated", "concern"])

    def test_positive_instruction_is_warm(self):
        import langchain_helper
        pos_instruction = langchain_helper._SENTIMENT_INSTRUCTIONS["positive"]
        assert any(word in pos_instruction.lower() for word in ["warm", "positive", "enthusiastic", "encouraging"])

    def test_neutral_instruction_is_empty(self):
        import langchain_helper
        assert langchain_helper._SENTIMENT_INSTRUCTIONS["neutral"] == ""

    def test_get_qa_chain_accepts_sentiment_param(self):
        import inspect
        import langchain_helper
        sig = inspect.signature(langchain_helper.get_qa_chain)
        assert "sentiment" in sig.parameters
        assert sig.parameters["sentiment"].default == "neutral"
