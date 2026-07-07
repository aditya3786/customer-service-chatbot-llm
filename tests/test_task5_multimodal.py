"""
Tests for Task 5: Multi-Modal AI Assistant
==========================================
Tests cover the 4-stage pipeline without making live API calls.
All LLM interactions are mocked; image handling and logic are tested directly.
"""

import base64
import json
import os
import sys
import types
import importlib
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on the path (set by conftest.py)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_png() -> bytes:
    """Return a minimal 1×1 white PNG as bytes (no external file needed)."""
    import struct, zlib
    def chunk(name, data):
        c = struct.pack(">I", len(data)) + name + data
        return c + struct.pack(">I", zlib.crc32(c[4:]) & 0xFFFFFFFF)

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = b"\x00\xff\xff\xff"
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


SAMPLE_PNG = _make_tiny_png()

SAMPLE_IMAGE_ANALYSIS = {
    "description": "A bar chart showing quarterly revenue growth from Q1 to Q4.",
    "objects": ["bar chart", "x-axis", "y-axis", "legend"],
    "text_content": "Q1 $10M, Q2 $15M, Q3 $18M, Q4 $22M",
    "scene_type": "chart",
    "colors": ["blue", "white", "grey"],
    "spatial_info": "Bars increase left to right indicating upward trend.",
    "confidence": "high",
}

SAMPLE_VALIDATION = {
    "is_valid": True,
    "confidence": 0.88,
    "caveats": [],
    "corrected_answer": None,
}

SAMPLE_AMBIGUITY_CLEAR = {
    "is_ambiguous": False,
    "clarifying_question": None,
    "assumptions": [],
    "reasoning": "The question clearly asks about the chart trend.",
}

SAMPLE_AMBIGUITY_VAGUE = {
    "is_ambiguous": True,
    "clarifying_question": "Are you asking about the highest bar or the overall trend?",
    "assumptions": ["Assuming user wants overall trend description"],
    "reasoning": "The question 'what does this show?' is too vague.",
}


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str):
    mock = MagicMock()
    mock.content = content
    return mock


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_bare_json(self):
        from multimodal_assistant import _extract_json
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_prose(self):
        from multimodal_assistant import _extract_json
        text = 'Here is the analysis: {"scene_type": "chart", "confidence": "high"} That is all.'
        result = _extract_json(text)
        assert result["scene_type"] == "chart"

    def test_invalid_returns_empty_dict(self):
        from multimodal_assistant import _extract_json
        assert _extract_json("no json here at all") == {}

    def test_nested_json(self):
        from multimodal_assistant import _extract_json
        data = {"outer": {"inner": [1, 2, 3]}}
        assert _extract_json(json.dumps(data)) == data


# ---------------------------------------------------------------------------
# analyze_image
# ---------------------------------------------------------------------------

class TestAnalyzeImage:
    def test_returns_all_required_keys(self):
        from multimodal_assistant import analyze_image
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_IMAGE_ANALYSIS))
            result = analyze_image(SAMPLE_PNG, "image/png")
        for key in ("description", "objects", "text_content", "scene_type",
                    "colors", "spatial_info", "confidence"):
            assert key in result, f"Missing key: {key}"

    def test_objects_is_list(self):
        from multimodal_assistant import analyze_image
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_IMAGE_ANALYSIS))
            result = analyze_image(SAMPLE_PNG)
        assert isinstance(result["objects"], list)

    def test_defaults_fill_missing_keys(self):
        from multimodal_assistant import analyze_image
        partial = {"description": "A photo of a cat", "scene_type": "photo"}
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(partial))
            result = analyze_image(SAMPLE_PNG)
        assert result["objects"] == []
        assert result["text_content"] == ""
        assert result["confidence"] == "medium"

    def test_empty_bytes_raises_value_error(self):
        from multimodal_assistant import analyze_image
        with pytest.raises(ValueError, match="non-empty"):
            analyze_image(b"")

    def test_uses_correct_mime_type(self):
        from multimodal_assistant import analyze_image
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_IMAGE_ANALYSIS))
            analyze_image(SAMPLE_PNG, "image/webp")
            call_args = mock_llm.invoke.call_args
            messages = call_args[0][0]
            image_content = messages[0].content[0]
            assert "image/webp" in image_content["image_url"]["url"]

    def test_llm_called_once(self):
        from multimodal_assistant import analyze_image
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_IMAGE_ANALYSIS))
            analyze_image(SAMPLE_PNG)
        mock_llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# detect_ambiguity
# ---------------------------------------------------------------------------

class TestDetectAmbiguity:
    def test_returns_required_keys(self):
        from multimodal_assistant import detect_ambiguity
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_AMBIGUITY_CLEAR))
            result = detect_ambiguity("What is the trend?", SAMPLE_IMAGE_ANALYSIS)
        for key in ("is_ambiguous", "clarifying_question", "assumptions", "reasoning"):
            assert key in result

    def test_is_ambiguous_is_bool_for_clear_question(self):
        from multimodal_assistant import detect_ambiguity
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_AMBIGUITY_CLEAR))
            result = detect_ambiguity("What is the highest bar?", SAMPLE_IMAGE_ANALYSIS)
        assert isinstance(result["is_ambiguous"], bool)
        assert result["is_ambiguous"] is False

    def test_is_ambiguous_true_for_vague_question(self):
        from multimodal_assistant import detect_ambiguity
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_AMBIGUITY_VAGUE))
            result = detect_ambiguity("What does this show?", SAMPLE_IMAGE_ANALYSIS)
        assert result["is_ambiguous"] is True
        assert result["clarifying_question"] is not None

    def test_string_true_coerced_to_bool(self):
        from multimodal_assistant import detect_ambiguity
        payload = dict(SAMPLE_AMBIGUITY_VAGUE)
        payload["is_ambiguous"] = "true"
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(payload))
            result = detect_ambiguity("?", SAMPLE_IMAGE_ANALYSIS)
        assert result["is_ambiguous"] is True

    def test_assumptions_is_list(self):
        from multimodal_assistant import detect_ambiguity
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_AMBIGUITY_VAGUE))
            result = detect_ambiguity("huh?", SAMPLE_IMAGE_ANALYSIS)
        assert isinstance(result["assumptions"], list)


# ---------------------------------------------------------------------------
# generate_response
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    def test_returns_string(self):
        from multimodal_assistant import generate_response
        answer = "Revenue [seen in image] grew from Q1 to Q4."
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(answer)
            result = generate_response("What is the trend?", SAMPLE_IMAGE_ANALYSIS, [])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_incorporates_chat_history(self):
        from multimodal_assistant import generate_response
        history = [("What is shown?", "A bar chart."), ("What are the values?", "Q1=$10M Q4=$22M.")]
        captured_prompts = []

        def capture_invoke(messages):
            for m in messages:
                if hasattr(m, "content") and isinstance(m.content, str):
                    captured_prompts.append(m.content)
            return _mock_llm_response("Follow-up answer.")

        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.side_effect = capture_invoke
            generate_response("What grew the most?", SAMPLE_IMAGE_ANALYSIS, history)

        combined = " ".join(captured_prompts)
        assert "What is shown?" in combined or "PRIOR CONVERSATION" in combined

    def test_empty_history_works(self):
        from multimodal_assistant import generate_response
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response("Answer here.")
            result = generate_response("Describe the image.", SAMPLE_IMAGE_ANALYSIS, [])
        assert "Answer here." in result

    def test_assumptions_included_when_provided(self):
        from multimodal_assistant import generate_response
        captured = []
        def capture(messages):
            for m in messages:
                if hasattr(m, "content") and isinstance(m.content, str):
                    captured.append(m.content)
            return _mock_llm_response("OK")
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.side_effect = capture
            generate_response("What?", SAMPLE_IMAGE_ANALYSIS, [], assumptions=["Assuming bar chart"])
        assert any("Assuming bar chart" in p for p in captured)


# ---------------------------------------------------------------------------
# validate_response
# ---------------------------------------------------------------------------

class TestValidateResponse:
    def test_returns_required_keys(self):
        from multimodal_assistant import validate_response
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_VALIDATION))
            result = validate_response("What is the trend?", SAMPLE_IMAGE_ANALYSIS, "Revenue grew.")
        for key in ("is_valid", "confidence", "caveats", "corrected_answer"):
            assert key in result

    def test_confidence_is_float_in_range(self):
        from multimodal_assistant import validate_response
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_VALIDATION))
            result = validate_response("trend?", SAMPLE_IMAGE_ANALYSIS, "answer")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_clamped_above_1(self):
        from multimodal_assistant import validate_response
        payload = {"is_valid": True, "confidence": 1.5, "caveats": [], "corrected_answer": None}
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(payload))
            result = validate_response("?", SAMPLE_IMAGE_ANALYSIS, "answer")
        assert result["confidence"] == 1.0

    def test_confidence_clamped_below_0(self):
        from multimodal_assistant import validate_response
        payload = {"is_valid": False, "confidence": -0.5, "caveats": ["bad"], "corrected_answer": "fix"}
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(payload))
            result = validate_response("?", SAMPLE_IMAGE_ANALYSIS, "answer")
        assert result["confidence"] == 0.0

    def test_caveats_is_list(self):
        from multimodal_assistant import validate_response
        with patch("multimodal_assistant.llm") as mock_llm:
            mock_llm.invoke.return_value = _mock_llm_response(json.dumps(SAMPLE_VALIDATION))
            result = validate_response("?", SAMPLE_IMAGE_ANALYSIS, "answer")
        assert isinstance(result["caveats"], list)


# ---------------------------------------------------------------------------
# run_pipeline (orchestrator)
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def _mock_all_stages(self):
        """Return a patcher context that mocks all 4 stage functions."""
        answer_text = "Revenue [seen in image] grew steadily from Q1 to Q4."
        return (
            patch("multimodal_assistant.analyze_image", return_value=SAMPLE_IMAGE_ANALYSIS),
            patch("multimodal_assistant.detect_ambiguity", return_value=SAMPLE_AMBIGUITY_CLEAR),
            patch("multimodal_assistant.generate_response", return_value=answer_text),
            patch("multimodal_assistant.validate_response", return_value=SAMPLE_VALIDATION),
        )

    def test_returns_all_expected_keys(self):
        from multimodal_assistant import run_pipeline
        p1, p2, p3, p4 = self._mock_all_stages()
        with p1, p2, p3, p4:
            result = run_pipeline("What is the trend?", SAMPLE_PNG, "image/png")
        for key in ("image_analysis", "ambiguity", "answer", "validation",
                    "final_answer", "stage_timings"):
            assert key in result, f"Missing key: {key}"

    def test_stage_timings_has_4_entries(self):
        from multimodal_assistant import run_pipeline
        p1, p2, p3, p4 = self._mock_all_stages()
        with p1, p2, p3, p4:
            result = run_pipeline("trend?", SAMPLE_PNG)
        assert len(result["stage_timings"]) == 4

    def test_final_answer_uses_corrected_when_invalid(self):
        from multimodal_assistant import run_pipeline
        invalid_validation = {
            "is_valid": False,
            "confidence": 0.4,
            "caveats": ["unsupported claim"],
            "corrected_answer": "Corrected: revenue actually declined.",
        }
        p1 = patch("multimodal_assistant.analyze_image", return_value=SAMPLE_IMAGE_ANALYSIS)
        p2 = patch("multimodal_assistant.detect_ambiguity", return_value=SAMPLE_AMBIGUITY_CLEAR)
        p3 = patch("multimodal_assistant.generate_response", return_value="Wrong answer")
        p4 = patch("multimodal_assistant.validate_response", return_value=invalid_validation)
        with p1, p2, p3, p4:
            result = run_pipeline("trend?", SAMPLE_PNG)
        assert result["final_answer"] == "Corrected: revenue actually declined."

    def test_chat_history_passed_to_generate(self):
        from multimodal_assistant import run_pipeline
        history = [("prev q", "prev a")]
        captured_history = []
        def fake_generate(q, ia, ch, assumptions=None):
            captured_history.extend(ch)
            return "answer"
        p1 = patch("multimodal_assistant.analyze_image", return_value=SAMPLE_IMAGE_ANALYSIS)
        p2 = patch("multimodal_assistant.detect_ambiguity", return_value=SAMPLE_AMBIGUITY_CLEAR)
        p3 = patch("multimodal_assistant.generate_response", side_effect=fake_generate)
        p4 = patch("multimodal_assistant.validate_response", return_value=SAMPLE_VALIDATION)
        with p1, p2, p3, p4:
            run_pipeline("new q", SAMPLE_PNG, chat_history=history)
        assert ("prev q", "prev a") in captured_history

    def test_empty_question_defaults_to_describe(self):
        from multimodal_assistant import run_pipeline
        captured_questions = []
        def fake_detect(q, ia):
            captured_questions.append(q)
            return SAMPLE_AMBIGUITY_CLEAR
        p1 = patch("multimodal_assistant.analyze_image", return_value=SAMPLE_IMAGE_ANALYSIS)
        p2 = patch("multimodal_assistant.detect_ambiguity", side_effect=fake_detect)
        p3 = patch("multimodal_assistant.generate_response", return_value="desc")
        p4 = patch("multimodal_assistant.validate_response", return_value=SAMPLE_VALIDATION)
        with p1, p2, p3, p4:
            run_pipeline("", SAMPLE_PNG)
        assert captured_questions[0] == "Describe this image in detail."


# ---------------------------------------------------------------------------
# Session state key in main.py
# ---------------------------------------------------------------------------

class TestMainSessionState:
    def test_multimodal_history_key_in_main(self):
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(main_path) as f:
            source = f.read()
        assert "multimodal_history" in source

    def test_visual_ai_tab_in_main(self):
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(main_path) as f:
            source = f.read()
        assert "Visual AI" in source or "tab_multimodal" in source

    def test_run_pipeline_imported_in_main(self):
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(main_path) as f:
            source = f.read()
        assert "from multimodal_assistant import run_pipeline" in source
