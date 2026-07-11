"""
Multilingual Conversation Assistant
=====================================
3-stage cross-lingual reasoning pipeline using Gemini 2.5 Flash + langdetect.

Stage 1  analyze_language         Language detection, switch detection, mixed-input flagging
Stage 2  resolve_context          Cross-lingual context resolution using conversation history
Stage 3  generate_multilingual_response  Response generation in user's detected language

The pipeline is intentionally multi-pass (3 separate LLM/logic stages) rather than
a single-shot inference, satisfying the "intelligent decision-making" requirement.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

import json
import re
import time

from langdetect import detect, detect_langs, LangDetectException
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_helper import llm  # shared Gemini 2.5 Flash — no re-init


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "de": "German",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
}

# Flag emojis for UI display
_LANG_FLAGS: dict[str, str] = {
    "en": "🇬🇧", "es": "🇪🇸", "fr": "🇫🇷", "hi": "🇮🇳",
    "de": "🇩🇪", "zh-cn": "🇨🇳", "zh-tw": "🇹🇼", "ja": "🇯🇵",
    "ar": "🇸🇦", "pt": "🇵🇹", "it": "🇮🇹", "ko": "🇰🇷", "ru": "🇷🇺",
}


def get_lang_flag(code: str) -> str:
    """Return a flag emoji for a language code, or 🌐 if unknown."""
    return _LANG_FLAGS.get(code, "🌐")


def get_lang_name(code: str) -> str:
    """Return a human-readable language name, or the code itself if unknown."""
    return SUPPORTED_LANGUAGES.get(code, code.upper())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Parse JSON from an LLM response that may contain extra prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _normalise_lang_code(code: str) -> str:
    """Normalise langdetect output to our registry keys."""
    code = code.lower()
    # langdetect returns 'zh-cn' or 'zh-tw'; keep as-is
    # Map common aliases
    aliases = {"zh": "zh-cn", "zhs": "zh-cn", "zht": "zh-tw"}
    return aliases.get(code, code)


# ---------------------------------------------------------------------------
# Stage 1: Language detection & analysis
# ---------------------------------------------------------------------------

def detect_language(text: str) -> dict:
    """
    Detect the primary language of a text using langdetect (open-source).

    Returns
    -------
    dict with keys: code (str), name (str), confidence (float 0–1),
                    all_detected (list[dict])
    """
    if not text or not text.strip():
        return {"code": "en", "name": "English", "confidence": 0.0, "all_detected": []}

    try:
        detected_list = detect_langs(text)
        primary = detected_list[0]
        code = _normalise_lang_code(str(primary.lang))
        return {
            "code": code,
            "name": get_lang_name(code),
            "confidence": round(primary.prob, 3),
            "all_detected": [
                {"code": _normalise_lang_code(str(d.lang)), "prob": round(d.prob, 3)}
                for d in detected_list
            ],
        }
    except LangDetectException:
        return {"code": "en", "name": "English", "confidence": 0.0, "all_detected": []}


def analyze_language(text: str, history: list[dict]) -> dict:
    """
    Stage 1: Full language analysis — detection, switch detection, mixed-input flagging.

    Parameters
    ----------
    text    : Current user input (may be in any language or mixed)
    history : Prior pipeline results (each entry has a 'detected_lang' key)

    Returns
    -------
    dict with keys:
        detected_lang  — primary language code
        lang_name      — human-readable language name
        confidence     — langdetect confidence (0–1)
        is_mixed       — True if sentences differ in detected language
        segments       — list of {text, lang_code} per sentence segment
        is_switch      — True if lang differs from most recent history entry
        prior_lang     — language code from last history entry, or None
    """
    primary = detect_language(text)

    # Segment-level mixed-language detection: split by sentence boundaries
    sentences = re.split(r"(?<=[.!?।。！？])\s+", text.strip())
    segments = []
    lang_codes_seen: set[str] = set()

    for sentence in sentences:
        if not sentence.strip():
            continue
        seg_lang = detect_language(sentence)
        lang_codes_seen.add(seg_lang["code"])
        segments.append({"text": sentence, "lang_code": seg_lang["code"]})

    is_mixed = len(lang_codes_seen) > 1

    # Switch detection: compare to previous turn
    prior_lang = None
    is_switch = False
    if history:
        prior_lang = history[-1].get("detected_lang")
        if prior_lang and prior_lang != primary["code"]:
            is_switch = True

    return {
        "detected_lang": primary["code"],
        "lang_name": primary["name"],
        "confidence": primary["confidence"],
        "is_mixed": is_mixed,
        "segments": segments,
        "is_switch": is_switch,
        "prior_lang": prior_lang,
    }


# ---------------------------------------------------------------------------
# Stage 2: Cross-lingual context resolution
# ---------------------------------------------------------------------------

_CONTEXT_RESOLUTION_TEMPLATE = """You are an expert multilingual assistant. Your job is to understand what a user is asking, \
even if they have switched languages or mixed languages within their message.

CONVERSATION HISTORY (last {n_turns} turns):
{history_section}

CURRENT USER INPUT: "{query}"
DETECTED LANGUAGE: {lang_name} ({lang_code})
LANGUAGE SWITCH DETECTED: {is_switch}
MIXED-LANGUAGE INPUT: {is_mixed}

Given the full conversation history and the language context above:
1. Determine the user's clear intent in plain English.
2. If the query is ambiguous, flag it and provide a clarifying question in {lang_name}.
3. Note which prior context (if any) is relevant to answering this query.

Return ONLY a valid JSON object:
{{
  "resolved_intent": "<clear English description of what the user wants>",
  "is_ambiguous": <true or false>,
  "context_used": "<brief note on which prior turns inform this intent, or 'none'>",
  "clarifying_question": "<question in {lang_name} to resolve ambiguity, or null if not ambiguous>"
}}"""


def resolve_context(query: str, lang_analysis: dict, history: list[dict]) -> dict:
    """
    Stage 2: Cross-lingual context resolution using conversation history.

    Reasons about what the user means, considering language switches and
    mixed-language inputs, using up to 5 prior turns of context.

    Returns
    -------
    dict with keys: resolved_intent, is_ambiguous, context_used, clarifying_question
    """
    # Build history section (last 5 turns)
    history_lines = []
    for i, entry in enumerate(history[-5:], 1):
        q = entry.get("query", "")
        r = entry.get("final_response", "")
        lang = entry.get("detected_lang", "?")
        lang_name = get_lang_name(lang)
        history_lines.append(f"Turn {i} [{lang_name}] — User: {q}")
        history_lines.append(f"Turn {i} [{lang_name}] — Assistant: {r[:200]}{'…' if len(r) > 200 else ''}")

    history_section = "\n".join(history_lines) if history_lines else "(no prior turns)"

    prompt = _CONTEXT_RESOLUTION_TEMPLATE.format(
        n_turns=min(5, len(history)),
        history_section=history_section,
        query=query,
        lang_name=lang_analysis.get("lang_name", "Unknown"),
        lang_code=lang_analysis.get("detected_lang", "?"),
        is_switch=lang_analysis.get("is_switch", False),
        is_mixed=lang_analysis.get("is_mixed", False),
    )

    raw = llm.invoke([
        SystemMessage(content="You are a precise multilingual reasoning assistant."),
        HumanMessage(content=prompt),
    ]).content

    result = _extract_json(raw)

    defaults = {
        "resolved_intent": query,
        "is_ambiguous": False,
        "context_used": "none",
        "clarifying_question": None,
    }
    for key, default in defaults.items():
        result.setdefault(key, default)

    if isinstance(result["is_ambiguous"], str):
        result["is_ambiguous"] = result["is_ambiguous"].lower() == "true"

    return result


# ---------------------------------------------------------------------------
# Stage 3: Multilingual response generation
# ---------------------------------------------------------------------------

_RESPONSE_TEMPLATE = """You are a helpful multilingual assistant. The user is communicating in {lang_name}.

USER'S RESOLVED INTENT: {resolved_intent}
ORIGINAL QUERY: "{query}"
RESPONSE LANGUAGE: {lang_name} ({lang_code})

CROSS-LINGUAL CONVERSATION CONTEXT:
{history_section}

Instructions:
1. Respond ENTIRELY in {lang_name}. Do NOT mix languages in your response.
2. Use the resolved intent to give a precise, helpful answer.
3. Reference relevant prior conversation turns if they inform the answer.
4. If you draw from context established in a different language, note it briefly.
5. Keep the response natural, clear, and appropriately concise.

After your main response, append a JSON block (only one, at the very end):
---NOTES---
{{"language_used": "{lang_code}", "cross_lingual_notes": "<brief English note on any cross-lingual reasoning you applied, or null>"}}"""


def generate_multilingual_response(
    query: str,
    lang_analysis: dict,
    context: dict,
    history: list[dict],
) -> dict:
    """
    Stage 3: Generate a response in the user's detected language with full
    cross-lingual context.

    Returns
    -------
    dict with keys: response (str), language_used (str), cross_lingual_notes (str|None)
    """
    history_lines = []
    for i, entry in enumerate(history[-3:], 1):
        q = entry.get("query", "")
        r = entry.get("final_response", "")
        lang = get_lang_name(entry.get("detected_lang", "?"))
        history_lines.append(f"[{lang}] User: {q}")
        history_lines.append(f"[{lang}] Assistant: {r[:200]}{'…' if len(r) > 200 else ''}")

    history_section = "\n".join(history_lines) if history_lines else "(no prior context)"

    prompt = _RESPONSE_TEMPLATE.format(
        lang_name=lang_analysis.get("lang_name", "English"),
        lang_code=lang_analysis.get("detected_lang", "en"),
        resolved_intent=context.get("resolved_intent", query),
        query=query,
        history_section=history_section,
    )

    raw = llm.invoke([HumanMessage(content=prompt)]).content

    # Split out the ---NOTES--- JSON block if present
    notes_match = re.search(r"---NOTES---\s*(\{[\s\S]*\})", raw)
    cross_lingual_notes = None
    language_used = lang_analysis.get("detected_lang", "en")

    if notes_match:
        notes_json = _extract_json(notes_match.group(1))
        cross_lingual_notes = notes_json.get("cross_lingual_notes")
        language_used = notes_json.get("language_used", language_used)
        response_text = raw[: notes_match.start()].strip()
    else:
        response_text = raw.strip()

    return {
        "response": response_text,
        "language_used": language_used,
        "cross_lingual_notes": cross_lingual_notes,
    }


# ---------------------------------------------------------------------------
# Orchestrator: full 3-stage pipeline
# ---------------------------------------------------------------------------

def run_multilingual_pipeline(
    query: str,
    history: list[dict] | None = None,
) -> dict:
    """
    Run the full 3-stage multilingual reasoning pipeline.

    Parameters
    ----------
    query   : User's text input (any language)
    history : List of prior pipeline result dicts from this conversation

    Returns
    -------
    dict with keys:
        lang_analysis   — Stage 1 dict
        context         — Stage 2 dict
        response_data   — Stage 3 dict
        final_response  — The assistant's response string (in user's language)
        detected_lang   — Language code of detected language (convenience key)
        stage_timings   — dict of elapsed seconds per stage
    """
    if history is None:
        history = []

    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    timings: dict[str, float] = {}

    # Stage 1
    t0 = time.time()
    lang_analysis = analyze_language(query, history)
    timings["stage1_language"] = round(time.time() - t0, 2)

    # Stage 2
    t0 = time.time()
    context = resolve_context(query, lang_analysis, history)
    timings["stage2_context"] = round(time.time() - t0, 2)

    # Stage 3
    t0 = time.time()
    response_data = generate_multilingual_response(query, lang_analysis, context, history)
    timings["stage3_response"] = round(time.time() - t0, 2)

    return {
        "lang_analysis": lang_analysis,
        "context": context,
        "response_data": response_data,
        "final_response": response_data["response"],
        "detected_lang": lang_analysis["detected_lang"],
        "query": query,
        "stage_timings": timings,
    }
