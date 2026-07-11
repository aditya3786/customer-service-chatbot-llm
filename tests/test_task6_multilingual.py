"""
Tests for Task 6: Multilingual Conversation Assistant
======================================================
All LLM calls are mocked — no API quota used.
"""

import importlib
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure src/ is on the path and TF/JAX are disabled before any import
# ---------------------------------------------------------------------------
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# Stub heavy modules before importing multilingual_assistant
# ---------------------------------------------------------------------------
def _make_stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for _heavy in [
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_community.document_loaders",
    "langchain_community.document_loaders.csv_loader",
    "langchain_huggingface",
    "faiss",
    "nltk",
    "nltk.sentiment",
    "nltk.sentiment.vader",
    "plotly",
    "plotly.express",
    "apscheduler",
    "apscheduler.schedulers",
    "apscheduler.schedulers.background",
    "bs4",
    "arxiv",
    "transformers",
    "PIL",
    "streamlit",
]:
    if _heavy not in sys.modules:
        _make_stub(_heavy)

# Stub langchain_helper before multilingual_assistant imports it
_lh = _make_stub("langchain_helper")
_lh.llm = MagicMock()
_lh.embeddings = MagicMock()
_lh.get_qa_chain = MagicMock()
_lh.create_vector_db = MagicMock()

import multilingual_assistant as ma


# ---------------------------------------------------------------------------
# Helper — build a fake LLM response
# ---------------------------------------------------------------------------

def _mock_llm_json(payload: dict):
    import json
    mock_resp = MagicMock()
    mock_resp.content = json.dumps(payload)
    return mock_resp


# ===========================================================================
# 1. detect_language
# ===========================================================================
class TestDetectLanguage(unittest.TestCase):

    def test_empty_string_returns_english(self):
        result = ma.detect_language("")
        self.assertEqual(result["code"], "en")
        self.assertEqual(result["confidence"], 0.0)

    def test_whitespace_only_returns_english(self):
        result = ma.detect_language("   ")
        self.assertEqual(result["code"], "en")

    def test_english_text(self):
        result = ma.detect_language("The customer service team is available to assist you with any questions you may have about our products and services.")
        self.assertEqual(result["code"], "en")
        self.assertIn("name", result)
        self.assertIn("confidence", result)
        self.assertIn("all_detected", result)

    def test_spanish_text(self):
        result = ma.detect_language("Hola, ¿cómo estás? Me llamo Juan.")
        self.assertEqual(result["code"], "es")
        self.assertGreater(result["confidence"], 0.5)

    def test_french_text(self):
        result = ma.detect_language("Bonjour, comment allez-vous aujourd'hui?")
        self.assertEqual(result["code"], "fr")

    def test_german_text(self):
        result = ma.detect_language("Guten Morgen, wie geht es Ihnen heute?")
        self.assertEqual(result["code"], "de")

    def test_hindi_text(self):
        result = ma.detect_language("नमस्ते, आप कैसे हैं?")
        self.assertEqual(result["code"], "hi")

    def test_returns_all_detected_list(self):
        result = ma.detect_language("Hello, how are you?")
        self.assertIsInstance(result["all_detected"], list)
        self.assertGreater(len(result["all_detected"]), 0)

    def test_langdetect_exception_returns_english_fallback(self):
        with patch("multilingual_assistant.detect_langs", side_effect=ma.LangDetectException(0, "err")):
            result = ma.detect_language("x")
        self.assertEqual(result["code"], "en")
        self.assertEqual(result["confidence"], 0.0)

    def test_portuguese_text(self):
        result = ma.detect_language("Olá, como você está hoje? Tudo bem?")
        self.assertEqual(result["code"], "pt")


# ===========================================================================
# 2. analyze_language (Stage 1)
# ===========================================================================
class TestAnalyzeLanguage(unittest.TestCase):

    def test_required_keys_present(self):
        result = ma.analyze_language("Hello there", history=[])
        for key in ("detected_lang", "lang_name", "confidence", "is_mixed", "segments", "is_switch", "prior_lang"):
            self.assertIn(key, result)

    def test_no_switch_on_first_turn(self):
        result = ma.analyze_language("Hello", history=[])
        self.assertFalse(result["is_switch"])
        self.assertIsNone(result["prior_lang"])

    def test_switch_detected_when_lang_changes(self):
        history = [{"detected_lang": "en", "query": "Hello", "final_response": "Hi"}]
        result = ma.analyze_language("Hola, ¿cómo estás? Me llamo María.", history=history)
        self.assertTrue(result["is_switch"])
        self.assertEqual(result["prior_lang"], "en")

    def test_no_switch_when_same_language(self):
        history = [{"detected_lang": "es", "query": "Hola", "final_response": "Hola!"}]
        result = ma.analyze_language("Buenos días, ¿cómo te va?", history=history)
        self.assertFalse(result["is_switch"])

    def test_segments_list_populated(self):
        result = ma.analyze_language("Hello. How are you?", history=[])
        self.assertIsInstance(result["segments"], list)

    def test_is_mixed_false_for_single_language(self):
        # Use longer, unambiguous English sentences — langdetect is probabilistic on short text
        text = "The customer service team is available Monday through Friday. Please contact us at any time. We are happy to assist you with your inquiry."
        result = ma.analyze_language(text, history=[])
        self.assertFalse(result["is_mixed"])

    def test_lang_name_populated(self):
        result = ma.analyze_language("Hello world", history=[])
        self.assertIsInstance(result["lang_name"], str)
        self.assertGreater(len(result["lang_name"]), 0)


# ===========================================================================
# 3. resolve_context (Stage 2)
# ===========================================================================
class TestResolveContext(unittest.TestCase):

    def _lang_analysis(self, lang="en"):
        return {
            "detected_lang": lang,
            "lang_name": ma.get_lang_name(lang),
            "confidence": 0.99,
            "is_mixed": False,
            "is_switch": False,
            "prior_lang": None,
            "segments": [],
        }

    def test_required_keys_present(self):
        payload = {
            "resolved_intent": "User wants a greeting",
            "is_ambiguous": False,
            "context_used": "none",
            "clarifying_question": None,
        }
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_json(payload)
            result = ma.resolve_context("Hello", self._lang_analysis(), [])
        for key in ("resolved_intent", "is_ambiguous", "context_used", "clarifying_question"):
            self.assertIn(key, result)

    def test_is_ambiguous_coerced_to_bool_from_string(self):
        payload = {"resolved_intent": "X", "is_ambiguous": "true", "context_used": "none", "clarifying_question": "What?"}
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_json(payload)
            result = ma.resolve_context("What?", self._lang_analysis(), [])
        self.assertIsInstance(result["is_ambiguous"], bool)
        self.assertTrue(result["is_ambiguous"])

    def test_is_ambiguous_false_when_clear_question(self):
        payload = {"resolved_intent": "User asks about price", "is_ambiguous": False, "context_used": "none", "clarifying_question": None}
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_json(payload)
            result = ma.resolve_context("What is the price?", self._lang_analysis(), [])
        self.assertFalse(result["is_ambiguous"])

    def test_history_included_in_prompt(self):
        history = [
            {"query": "Hello", "final_response": "Hi there!", "detected_lang": "en"},
            {"query": "What time is it?", "final_response": "It's noon.", "detected_lang": "en"},
        ]
        payload = {"resolved_intent": "continuation", "is_ambiguous": False, "context_used": "turns 1-2", "clarifying_question": None}
        captured = {}
        def _invoke(msgs):
            captured["msgs"] = msgs
            return _mock_llm_json(payload)
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.side_effect = _invoke
            ma.resolve_context("And?", self._lang_analysis(), history)
        prompt_text = " ".join(str(m.content) for m in captured["msgs"])
        self.assertIn("Hello", prompt_text)

    def test_defaults_applied_when_llm_returns_empty(self):
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="not json at all %%")
            result = ma.resolve_context("Hi", self._lang_analysis(), [])
        self.assertIn("resolved_intent", result)
        self.assertIn("is_ambiguous", result)


# ===========================================================================
# 4. generate_multilingual_response (Stage 3)
# ===========================================================================
class TestGenerateMultilingualResponse(unittest.TestCase):

    def _lang_analysis(self, lang="es"):
        return {
            "detected_lang": lang,
            "lang_name": ma.get_lang_name(lang),
            "confidence": 0.95,
            "is_mixed": False,
            "is_switch": False,
            "prior_lang": None,
            "segments": [],
        }

    def _context(self):
        return {
            "resolved_intent": "User wants a greeting in Spanish",
            "is_ambiguous": False,
            "context_used": "none",
            "clarifying_question": None,
        }

    def test_required_keys_present(self):
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Hola! ¿Cómo puedo ayudarte?")
            result = ma.generate_multilingual_response("Hola", self._lang_analysis(), self._context(), [])
        for key in ("response", "language_used", "cross_lingual_notes"):
            self.assertIn(key, result)

    def test_response_is_string(self):
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Bonjour!")
            result = ma.generate_multilingual_response("Bonjour", self._lang_analysis("fr"), self._context(), [])
        self.assertIsInstance(result["response"], str)

    def test_notes_extracted_when_present(self):
        raw = "Hola, puedo ayudar.\n---NOTES---\n{\"language_used\": \"es\", \"cross_lingual_notes\": \"switched from English\"}"
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content=raw)
            result = ma.generate_multilingual_response("Hola", self._lang_analysis(), self._context(), [])
        self.assertEqual(result["language_used"], "es")
        self.assertEqual(result["cross_lingual_notes"], "switched from English")
        self.assertNotIn("---NOTES---", result["response"])

    def test_cross_lingual_notes_none_when_no_notes_block(self):
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Hola! Estoy bien.")
            result = ma.generate_multilingual_response("Hola", self._lang_analysis(), self._context(), [])
        self.assertIsNone(result["cross_lingual_notes"])


# ===========================================================================
# 5. run_multilingual_pipeline (orchestrator)
# ===========================================================================
class TestRunPipeline(unittest.TestCase):

    def _full_mock(self):
        ctx_payload = {"resolved_intent": "test intent", "is_ambiguous": False, "context_used": "none", "clarifying_question": None}
        resp_payload = MagicMock(content="Test response in target language.")
        import json
        return [
            MagicMock(content=json.dumps(ctx_payload)),  # Stage 2
            resp_payload,                                  # Stage 3
        ]

    def test_required_output_keys(self):
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.side_effect = self._full_mock()
            result = ma.run_multilingual_pipeline("Hello, how are you?", history=[])
        for key in ("lang_analysis", "context", "response_data", "final_response", "detected_lang", "query", "stage_timings"):
            self.assertIn(key, result)

    def test_stage_timings_has_three_keys(self):
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.side_effect = self._full_mock()
            result = ma.run_multilingual_pipeline("Hello", history=[])
        self.assertEqual(len(result["stage_timings"]), 3)
        self.assertIn("stage1_language", result["stage_timings"])
        self.assertIn("stage2_context", result["stage_timings"])
        self.assertIn("stage3_response", result["stage_timings"])

    def test_final_response_is_string(self):
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.side_effect = self._full_mock()
            result = ma.run_multilingual_pipeline("Bonjour!", history=[])
        self.assertIsInstance(result["final_response"], str)

    def test_detected_lang_is_string(self):
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.side_effect = self._full_mock()
            result = ma.run_multilingual_pipeline("Hello world!", history=[])
        self.assertIsInstance(result["detected_lang"], str)

    def test_empty_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            ma.run_multilingual_pipeline("", history=[])

    def test_whitespace_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            ma.run_multilingual_pipeline("   ", history=[])

    def test_query_stored_in_result(self):
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.side_effect = self._full_mock()
            result = ma.run_multilingual_pipeline("Guten Tag", history=[])
        self.assertEqual(result["query"], "Guten Tag")

    def test_none_history_handled(self):
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.side_effect = self._full_mock()
            result = ma.run_multilingual_pipeline("Hi there!", history=None)
        self.assertIsNotNone(result)

    def test_history_language_switch_detected(self):
        history = [{"detected_lang": "en", "query": "Hello", "final_response": "Hi", "lang_analysis": {"is_switch": False, "is_mixed": False}}]
        with patch("multilingual_assistant.llm") as mock_llm:
            mock_llm.invoke.side_effect = self._full_mock()
            result = ma.run_multilingual_pipeline("Hola, cómo estás amigo?", history=history)
        # Spanish input after English history — switch should be detected
        self.assertEqual(result["lang_analysis"]["prior_lang"], "en")


# ===========================================================================
# 6. Utility functions
# ===========================================================================
class TestUtilityFunctions(unittest.TestCase):

    def test_get_lang_flag_known(self):
        self.assertEqual(ma.get_lang_flag("es"), "🇪🇸")
        self.assertEqual(ma.get_lang_flag("fr"), "🇫🇷")
        self.assertEqual(ma.get_lang_flag("hi"), "🇮🇳")
        self.assertEqual(ma.get_lang_flag("de"), "🇩🇪")

    def test_get_lang_flag_unknown_returns_globe(self):
        self.assertEqual(ma.get_lang_flag("xx"), "🌐")

    def test_get_lang_name_known(self):
        self.assertEqual(ma.get_lang_name("es"), "Spanish")
        self.assertEqual(ma.get_lang_name("fr"), "French")
        self.assertEqual(ma.get_lang_name("hi"), "Hindi")

    def test_get_lang_name_unknown_returns_uppercased_code(self):
        self.assertEqual(ma.get_lang_name("xx"), "XX")

    def test_supported_languages_has_at_least_four_non_english(self):
        non_english = [k for k in ma.SUPPORTED_LANGUAGES if k != "en"]
        self.assertGreaterEqual(len(non_english), 4)

    def test_supported_languages_includes_required(self):
        for lang in ("es", "fr", "hi", "de"):
            self.assertIn(lang, ma.SUPPORTED_LANGUAGES)


# ===========================================================================
# 7. main.py session state
# ===========================================================================
class TestMainSessionState(unittest.TestCase):

    def test_multilingual_history_key_in_main(self):
        import ast
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(main_path, encoding="utf-8") as f:
            source = f.read()
        self.assertIn("multilingual_history", source)

    def test_multilingual_tab_in_main(self):
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(main_path, encoding="utf-8") as f:
            source = f.read()
        self.assertIn("tab_multi", source)
        self.assertIn("🌐 Multilingual", source)

    def test_import_in_main(self):
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(main_path, encoding="utf-8") as f:
            source = f.read()
        self.assertIn("run_multilingual_pipeline", source)
        self.assertIn("SUPPORTED_LANGUAGES", source)


if __name__ == "__main__":
    unittest.main()
