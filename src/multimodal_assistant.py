"""
Multi-Modal AI Assistant
========================
4-stage reasoning pipeline over text + image inputs using Gemini 2.5 Flash Vision.

Stage 1  analyze_image      Structured JSON extraction (objects, OCR, scene type, …)
Stage 2  detect_ambiguity   Identifies vague questions; surfaces assumptions
Stage 3  generate_response  Evidence-labelled answer using image context + chat history
Stage 4  validate_response  Self-critique pass that assigns confidence and flags caveats

The pipeline is intentionally multi-pass (4 separate LLM calls) rather than a
single-shot inference, satisfying the "intelligent decision-making" requirement.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

import base64
import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_helper import llm  # shared Gemini 2.5 Flash — no re-init


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Parse JSON from an LLM response that may contain extra prose."""
    # Try bare parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract first JSON object/array
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _image_message(image_bytes: bytes, mime_type: str, text: str) -> HumanMessage:
    """Build a LangChain HumanMessage with an inline base64 image + text."""
    return HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{_b64(image_bytes)}"},
        },
        {"type": "text", "text": text},
    ])


# ---------------------------------------------------------------------------
# Stage 1: Structured image analysis
# ---------------------------------------------------------------------------

_IMAGE_ANALYSIS_PROMPT = """Analyze this image thoroughly and return ONLY a valid JSON object with exactly these keys:

{
  "description": "<one-paragraph plain-English description of the overall image>",
  "objects": ["<list of identified objects, entities, or people>"],
  "text_content": "<all visible text / OCR content, or empty string if none>",
  "scene_type": "<one of: chart, photo, diagram, screenshot, document, medical, nature, product, artwork, code, other>",
  "colors": ["<list of dominant colors>"],
  "spatial_info": "<brief note on notable spatial relationships, layout, or composition>",
  "confidence": "<one of: high, medium, low>"
}

Return ONLY the JSON object — no markdown fences, no extra text."""


def analyze_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Stage 1: Extract structured information from an image via Gemini Vision.

    Returns
    -------
    dict with keys: description, objects, text_content, scene_type,
                    colors, spatial_info, confidence
    """
    if not image_bytes:
        raise ValueError("image_bytes must be non-empty")

    msg = _image_message(image_bytes, mime_type, _IMAGE_ANALYSIS_PROMPT)
    raw = llm.invoke([msg]).content
    result = _extract_json(raw)

    # Ensure all required keys are present with safe defaults
    defaults = {
        "description": "",
        "objects": [],
        "text_content": "",
        "scene_type": "other",
        "colors": [],
        "spatial_info": "",
        "confidence": "medium",
    }
    for key, default in defaults.items():
        result.setdefault(key, default)

    return result


# ---------------------------------------------------------------------------
# Stage 2: Ambiguity detection
# ---------------------------------------------------------------------------

_AMBIGUITY_PROMPT_TEMPLATE = """You are evaluating whether a user's question is ambiguous given the image analysis below.

IMAGE ANALYSIS SUMMARY:
Scene type: {scene_type}
Description: {description}
Objects detected: {objects}
Text visible: {text_content}

USER QUESTION: "{question}"

Decide if the question is ambiguous (i.e., it could mean multiple things, lacks necessary specificity, or cannot be answered confidently without making assumptions).

Return ONLY a valid JSON object:
{{
  "is_ambiguous": <true or false>,
  "clarifying_question": "<a single clarifying question to resolve ambiguity, or null if not ambiguous>",
  "assumptions": ["<list of assumptions being made to proceed, empty list if not ambiguous>"],
  "reasoning": "<one sentence explaining your decision>"
}}"""


def detect_ambiguity(question: str, image_analysis: dict) -> dict:
    """
    Stage 2: Determine if the user's question is ambiguous in context of the image.

    Returns
    -------
    dict with keys: is_ambiguous (bool), clarifying_question (str|None),
                    assumptions (list[str]), reasoning (str)
    """
    prompt = _AMBIGUITY_PROMPT_TEMPLATE.format(
        scene_type=image_analysis.get("scene_type", "unknown"),
        description=image_analysis.get("description", ""),
        objects=", ".join(image_analysis.get("objects", [])) or "none detected",
        text_content=image_analysis.get("text_content", "") or "none",
        question=question,
    )
    raw = llm.invoke([SystemMessage(content="You are a precise analytical assistant."),
                      HumanMessage(content=prompt)]).content
    result = _extract_json(raw)

    defaults = {
        "is_ambiguous": False,
        "clarifying_question": None,
        "assumptions": [],
        "reasoning": "",
    }
    for key, default in defaults.items():
        result.setdefault(key, default)

    # Coerce is_ambiguous to bool in case LLM returned a string
    if isinstance(result["is_ambiguous"], str):
        result["is_ambiguous"] = result["is_ambiguous"].lower() == "true"

    return result


# ---------------------------------------------------------------------------
# Stage 3: Context-aware response generation
# ---------------------------------------------------------------------------

_RESPONSE_PROMPT_TEMPLATE = """You are an intelligent multi-modal assistant. You have been given an image to analyze and a question about it.

IMAGE ANALYSIS (pre-extracted):
Scene type: {scene_type}
Description: {description}
Objects/entities: {objects}
Visible text (OCR): {text_content}
Spatial layout: {spatial_info}

{history_section}

ASSUMPTIONS MADE (if any): {assumptions}

USER QUESTION: "{question}"

Instructions:
1. Answer the question based on the image analysis and your own visual reasoning.
2. Label every piece of evidence with one of:
   - [seen in image] — directly observable in the image
   - [inferred] — logically derived from what is seen
   - [general knowledge] — background knowledge not specific to this image
3. If you cannot determine something from the image, say so explicitly.
4. Keep the answer focused, clear, and factual.
5. Do NOT repeat the image analysis verbatim — synthesise it into a useful answer.

Provide your response now:"""


def generate_response(
    question: str,
    image_analysis: dict,
    chat_history: list[tuple[str, str]],
    assumptions: list[str] | None = None,
) -> str:
    """
    Stage 3: Generate an evidence-labelled answer grounded in image analysis + history.

    Parameters
    ----------
    question       : Current user question
    image_analysis : Output from analyze_image()
    chat_history   : List of (user_question, assistant_answer) tuples from prior turns
    assumptions    : Assumptions surfaced by detect_ambiguity()

    Returns
    -------
    str  — the assistant's answer with [seen in image] / [inferred] / [general knowledge] labels
    """
    # Build conversation history section
    if chat_history:
        history_lines = ["PRIOR CONVERSATION (for context):"]
        for i, (q, a) in enumerate(chat_history[-3:], 1):  # last 3 turns
            history_lines.append(f"Turn {i} — User: {q}")
            history_lines.append(f"Turn {i} — Assistant: {a[:300]}{'…' if len(a) > 300 else ''}")
        history_section = "\n".join(history_lines)
    else:
        history_section = ""

    prompt = _RESPONSE_PROMPT_TEMPLATE.format(
        scene_type=image_analysis.get("scene_type", "unknown"),
        description=image_analysis.get("description", ""),
        objects=", ".join(image_analysis.get("objects", [])) or "none detected",
        text_content=image_analysis.get("text_content", "") or "none",
        spatial_info=image_analysis.get("spatial_info", ""),
        history_section=history_section,
        assumptions=", ".join(assumptions or []) or "none",
        question=question,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ---------------------------------------------------------------------------
# Stage 4: Response validation (self-critique)
# ---------------------------------------------------------------------------

_VALIDATION_PROMPT_TEMPLATE = """You are a critical reviewer checking an AI assistant's answer for accuracy and reliability.

ORIGINAL QUESTION: "{question}"

IMAGE ANALYSIS USED:
Scene type: {scene_type}
Description: {description}
Objects: {objects}
Visible text: {text_content}

ANSWER TO REVIEW:
{answer}

Evaluate the answer and return ONLY a valid JSON object:
{{
  "is_valid": <true if the answer is factually consistent with the image analysis, false otherwise>,
  "confidence": <float between 0.0 and 1.0 — how confident you are in the answer's accuracy>,
  "caveats": ["<list of specific concerns, unsupported claims, or limitations>"],
  "corrected_answer": "<a corrected version ONLY if is_valid is false, otherwise null>"
}}

Criteria for low confidence:
- Claims not supported by the image analysis
- Over-interpretation of ambiguous visual elements
- Incorrect OCR reading
- Assumptions presented as facts"""


def validate_response(question: str, image_analysis: dict, answer: str) -> dict:
    """
    Stage 4: Self-critique the generated answer for accuracy and reliability.

    Returns
    -------
    dict with keys: is_valid (bool), confidence (float 0–1),
                    caveats (list[str]), corrected_answer (str|None)
    """
    prompt = _VALIDATION_PROMPT_TEMPLATE.format(
        question=question,
        scene_type=image_analysis.get("scene_type", "unknown"),
        description=image_analysis.get("description", ""),
        objects=", ".join(image_analysis.get("objects", [])) or "none",
        text_content=image_analysis.get("text_content", "") or "none",
        answer=answer,
    )
    raw = llm.invoke([SystemMessage(content="You are a rigorous fact-checker."),
                      HumanMessage(content=prompt)]).content
    result = _extract_json(raw)

    defaults = {
        "is_valid": True,
        "confidence": 0.7,
        "caveats": [],
        "corrected_answer": None,
    }
    for key, default in defaults.items():
        result.setdefault(key, default)

    # Coerce types
    if isinstance(result["is_valid"], str):
        result["is_valid"] = result["is_valid"].lower() == "true"
    try:
        result["confidence"] = float(result["confidence"])
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))
    except (ValueError, TypeError):
        result["confidence"] = 0.7

    return result


# ---------------------------------------------------------------------------
# Orchestrator: full 4-stage pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    question: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    chat_history: list[tuple[str, str]] | None = None,
) -> dict:
    """
    Run the full 4-stage multi-modal reasoning pipeline.

    Parameters
    ----------
    question      : User's text question (may be empty string for pure image analysis)
    image_bytes   : Raw image bytes (JPEG, PNG, or WebP)
    mime_type     : MIME type string, e.g. 'image/jpeg'
    chat_history  : List of (user_q, assistant_ans) tuples from previous turns

    Returns
    -------
    dict with keys:
        image_analysis  — Stage 1 structured dict
        ambiguity       — Stage 2 structured dict
        answer          — Stage 3 answer string (with evidence labels)
        validation      — Stage 4 validation dict
        final_answer    — corrected_answer if invalid, else answer
        stage_timings   — dict of elapsed seconds per stage
    """
    if chat_history is None:
        chat_history = []

    timings: dict[str, float] = {}

    # Default question if none provided
    effective_question = question.strip() if question.strip() else "Describe this image in detail."

    # Stage 1
    t0 = time.time()
    image_analysis = analyze_image(image_bytes, mime_type)
    timings["stage1_analyze"] = round(time.time() - t0, 2)

    # Stage 2
    t0 = time.time()
    ambiguity = detect_ambiguity(effective_question, image_analysis)
    timings["stage2_ambiguity"] = round(time.time() - t0, 2)

    # Stage 3
    t0 = time.time()
    answer = generate_response(
        effective_question,
        image_analysis,
        chat_history,
        assumptions=ambiguity.get("assumptions", []),
    )
    timings["stage3_response"] = round(time.time() - t0, 2)

    # Stage 4
    t0 = time.time()
    validation = validate_response(effective_question, image_analysis, answer)
    timings["stage4_validate"] = round(time.time() - t0, 2)

    # Use corrected answer if validation flagged the original as invalid
    final_answer = validation.get("corrected_answer") or answer

    return {
        "image_analysis": image_analysis,
        "ambiguity": ambiguity,
        "answer": answer,
        "validation": validation,
        "final_answer": final_answer,
        "stage_timings": timings,
    }
