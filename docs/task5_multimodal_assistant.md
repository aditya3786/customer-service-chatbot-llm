# Task 5: Multi-Modal AI Assistant

## Overview
A multi-modal AI assistant that accepts both text and image inputs and reasons over them through a 4-stage pipeline. Rather than a single model inference call, the system makes four separate LLM passes with distinct purposes — satisfying the requirement for intelligent decision-making over simple model inference.

**Model:** Google Gemini 2.5 Flash (vision-capable, reused from all prior tasks — no new model downloads)

---

## 4-Stage Pipeline

```
User Input (image + question)
         │
         ▼
┌─────────────────────────────┐
│  Stage 1: Image Analysis    │  → Structured JSON: description, objects,
│  (Gemini Vision)            │    text/OCR, scene_type, colors, spatial_info,
└────────────┬────────────────┘    confidence
             │
             ▼
┌─────────────────────────────┐
│  Stage 2: Ambiguity         │  → {is_ambiguous, clarifying_question,
│  Detection (text LLM)       │     assumptions, reasoning}
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 3: Response          │  → Evidence-labelled answer using
│  Generation (vision LLM)    │    image context + chat history
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 4: Validation        │  → {is_valid, confidence (0-1),
│  Self-Critique (text LLM)   │     caveats, corrected_answer}
└─────────────────────────────┘
         │
         ▼
    Final Answer
```

---

## Evidence Labeling Convention

Every claim in the generated answer is labeled with one of three evidence tags:

| Tag | Meaning | Example |
|-----|---------|---------|
| `[seen in image]` | Directly observable in the image | "The chart shows Q4 revenue of $22M [seen in image]" |
| `[inferred]` | Logically derived from what is visible | "Revenue growth accelerated in H2 [inferred]" |
| `[general knowledge]` | Background knowledge not from this image | "Bar charts are used to compare discrete categories [general knowledge]" |

---

## Ambiguity Handling

When a question is deemed ambiguous (Stage 2), the system:
1. States the ambiguity and its reasoning
2. Surfaces explicit **assumptions** it is making to proceed
3. Offers a **clarifying question** for the user's awareness
4. Continues to answer using those assumptions (does not block)

**Example:**
- Question: `"What does this mean?"`
- Detection: ambiguous — could refer to the chart title, the trend, or a specific data point
- Assumption: `"Assuming user wants an overall description of the chart's meaning"`
- Clarifying Q: `"Are you asking about the title text, the overall trend, or a specific bar?"`

---

## Gemini Vision Content Block Format

```python
from langchain_core.messages import HumanMessage
import base64

msg = HumanMessage(content=[
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
        },
    },
    {"type": "text", "text": prompt},
])
response = llm.invoke([msg])
```

Supported MIME types: `image/jpeg`, `image/png`, `image/webp`

---

## Files

| File | Purpose |
|------|---------|
| `src/multimodal_assistant.py` | Core 4-stage pipeline (`analyze_image`, `detect_ambiguity`, `generate_response`, `validate_response`, `run_pipeline`) |
| `src/main.py` | Added `🖼️ Visual AI` tab (6th tab) and `multimodal_history` session state |
| `tests/test_task5_multimodal.py` | 28 unit tests (all LLM calls mocked) |
| `requirements.txt` | Added `Pillow` |

---

## Sample Conversation

**Upload:** A bar chart of quarterly sales

> **User:** What is the trend shown in this chart?
>
> **Assistant (Stage 1):** Scene: chart. Objects: bar chart, axes, legend. Text: Q1 $10M, Q2 $15M, Q3 $18M, Q4 $22M.
>
> **Assistant (Stage 2):** Question is clear — no ambiguity detected.
>
> **Assistant (Stage 3):** The chart shows a consistent upward trend in revenue [seen in image]. Revenue increased from $10M in Q1 to $22M in Q4 [seen in image], representing 120% growth over the year [inferred]. This type of growth pattern often indicates successful product-market fit [general knowledge].
>
> **Validation:** Confidence 0.92, valid, no caveats.

**Follow-up (same image, history remembered):**

> **User:** Which quarter had the biggest jump?
>
> **Assistant:** Based on the prior conversation where Q1=$10M and Q4=$22M [seen in image], the largest single-quarter increase was Q3 to Q4 (+$4M) [inferred from text content].

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Reuse `langchain_helper.llm` | No re-initialization; same Gemini 2.5 Flash instance used by all other tasks |
| 4 separate LLM calls | Satisfies "intelligent decision-making, not simple inference" — each call has a distinct, specialized prompt |
| `HumanMessage` with base64 image | LangChain-native way to pass images to Gemini Vision via `langchain-google-genai` |
| Mocked tests | Stage functions accept bytes; tests validate logic, JSON parsing, type coercion, and defaults without API quota |
| `[seen in image]` labels | Explicit evidence attribution makes reasoning transparent and auditable |
| `chat_history` capped at 3 turns | Keeps prompt size bounded; sufficient for most follow-up chains |
