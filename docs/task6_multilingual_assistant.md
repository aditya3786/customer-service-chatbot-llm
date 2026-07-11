# Task 6: Multilingual Conversation Assistant

## Overview
A multilingual conversation assistant that auto-detects language, handles mid-conversation language switches, resolves mixed-language inputs, and maintains cross-lingual context across turns. Built as a 3-stage pipeline using `langdetect` (open-source) for language detection and Gemini 2.5 Flash for cross-lingual reasoning.

**Model:** Google Gemini 2.5 Flash (natively multilingual — reused from all prior tasks)  
**Language detection:** `langdetect` (open-source Python library, based on Google's language-detection algorithm)

---

## 3-Stage Pipeline

```
User Input (any language)
         │
         ▼
┌─────────────────────────────────┐
│  Stage 1: Language Analysis     │  → {detected_lang, lang_name, confidence,
│  (langdetect + sentence split)  │     is_mixed, segments, is_switch, prior_lang}
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Stage 2: Cross-Lingual Context │  → {resolved_intent, is_ambiguous,
│  Resolution (text LLM)          │     context_used, clarifying_question}
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Stage 3: Multilingual Response │  → {response (in user's language),
│  Generation (text LLM)          │     language_used, cross_lingual_notes}
└─────────────────────────────────┘
         │
         ▼
    Final Response (in detected language)
```

---

## Supported Languages

| Code | Language | Flag |
|------|----------|------|
| `en` | English | 🇬🇧 |
| `es` | Spanish | 🇪🇸 |
| `fr` | French | 🇫🇷 |
| `hi` | Hindi | 🇮🇳 |
| `de` | German | 🇩🇪 |
| `zh-cn` | Chinese (Simplified) | 🇨🇳 |
| `ja` | Japanese | 🇯🇵 |
| `ar` | Arabic | 🇸🇦 |
| `pt` | Portuguese | 🇵🇹 |
| `it` | Italian | 🇮🇹 |
| `ko` | Korean | 🇰🇷 |
| `ru` | Russian | 🇷🇺 |

---

## Language Switch Detection

Stage 1 compares the detected language of the current turn against `history[-1]["detected_lang"]`. If they differ, `is_switch = True` and `prior_lang` is set. The UI shows a warning banner when a switch is detected, with the prior language and new language named explicitly. Conversation context is still carried forward — the switch does not reset history.

---

## Mixed-Language Input Handling

Stage 1 splits input by sentence boundaries (`re.split(r"(?<=[.!?।。！？])\s+", text)`) and runs `langdetect` on each sentence independently. If more than one language code is detected across segments, `is_mixed = True` and a per-segment breakdown is returned. Example:

```
Input: "What is the precio of this?"
Segments:
  "What is the precio of this?"  → en (0.72) / es (0.20)
is_mixed: True  (two language codes seen across segments)
```

---

## Cross-Lingual Context Retention

Stage 2 receives the last 5 turns of conversation history, **regardless of language**, with each turn labeled with its language:

```
Turn 1 [English] — User: Hello, what are your services?
Turn 1 [English] — Assistant: We offer customer support, ...
Turn 2 [Spanish] — User: ¿Cuánto cuesta el servicio premium?
Turn 2 [Spanish] — Assistant: El servicio premium cuesta...
```

The LLM reasons across all prior turns to understand the user's current intent, even if the languages differ. The `resolved_intent` is always expressed in English internally, then Stage 3 translates the intent into a response in the user's detected language.

---

## Files

| File | Purpose |
|------|---------|
| `src/multilingual_assistant.py` | Core 3-stage pipeline (`detect_language`, `analyze_language`, `resolve_context`, `generate_multilingual_response`, `run_multilingual_pipeline`) |
| `src/main.py` | Added `🌐 Multilingual` tab (7th tab) and `multilingual_history` session state |
| `tests/test_task6_multilingual.py` | 30 unit tests across 7 classes (all LLM calls mocked) |
| `requirements.txt` | Added `langdetect` |

---

## Sample Conversation

**Turn 1 (English):**
> **User:** Hello! What kind of support do you offer?
>
> **Stage 1:** 🇬🇧 English (99% confidence)
> **Stage 2:** Intent — User asks about available support services
> **Assistant:** We offer 24/7 customer support for billing, technical issues, and account management...

**Turn 2 (Spanish — language switch):**
> **User:** ¿Cuánto cuesta el plan premium?
>
> **Stage 1:** 🇪🇸 Spanish (97% confidence) | ⬆ Language switch: English → Spanish
> **Stage 2:** Intent — User asks about premium plan pricing (cross-lingual note: follows up on services mentioned in Turn 1)
> **Assistant:** El plan premium cuesta $29.99 al mes e incluye soporte prioritario y acceso a todas las funciones avanzadas...

**Turn 3 (Hindi — second switch, context retained):**
> **User:** क्या मुझे कोई छूट मिल सकती है?
>
> **Stage 1:** 🇮🇳 Hindi (95% confidence) | ⬆ Language switch: Spanish → Hindi
> **Stage 2:** Intent — User asks about discounts (context: premium plan discussed in Turn 2)
> **Assistant:** हाँ! यदि आप वार्षिक योजना चुनते हैं, तो आपको 20% की छूट मिलेगी...

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| `langdetect` for language ID | Open-source (BSD), no API key, runs fully offline, supports 55+ languages |
| Reuse `langchain_helper.llm` | No re-initialization; same Gemini 2.5 Flash instance used by all other tasks |
| 3 separate LLM/logic calls | Satisfies "intelligent decision-making, not simple inference" — each stage has a distinct purpose |
| `resolved_intent` in English (Stage 2) | Decouples language understanding from response generation; allows cross-lingual reasoning without translating the full prompt |
| Last 5 turns in Stage 2, last 3 in Stage 3 | Stage 2 needs more history for intent disambiguation; Stage 3 needs less (focused on current response) |
| Sentence-level segmentation for mixed-language | More accurate than word-level; preserves sentence boundaries for natural language |
| `---NOTES---` block in Stage 3 response | Lets the model append metadata (language_used, cross_lingual_notes) without polluting the user-facing response |
