"""
Medical Named Entity Recognition (NER)
=======================================
Keyword/regex-based entity recognition for three medical categories:
  - Disease/Condition
  - Symptom
  - Treatment

No neural network or heavy model required — runs fully offline.
Entities are matched case-insensitively using word-boundary regex.
"""

import re

# Color mapping for UI highlighting
CATEGORY_COLORS = {
    "Disease/Condition": "#ffcccc",   # light red
    "Symptom":           "#ffe0b2",   # light orange
    "Treatment":         "#c8e6c9",   # light green
}

MEDICAL_ENTITIES = {
    "Disease/Condition": [
        "leukemia", "lymphoma", "cancer", "tumor", "carcinoma", "melanoma",
        "diabetes", "diabetic", "alzheimer", "dementia", "parkinson",
        "multiple sclerosis", "epilepsy", "stroke", "aneurysm",
        "heart disease", "hypertension", "high blood pressure",
        "asthma", "copd", "emphysema", "pneumonia", "tuberculosis",
        "hiv", "aids", "hepatitis", "cirrhosis", "kidney disease",
        "renal failure", "arthritis", "osteoporosis", "lupus",
        "crohn", "colitis", "ibs", "celiac", "anemia", "hemophilia",
        "sickle cell", "cystic fibrosis", "down syndrome",
        "autism", "adhd", "depression", "anxiety", "schizophrenia",
        "bipolar", "ptsd", "eating disorder", "obesity",
        "hypothyroidism", "hyperthyroidism", "graves disease",
        "psoriasis", "eczema", "glaucoma", "macular degeneration",
        "cataracts", "fibromyalgia", "chronic fatigue",
        "sepsis", "meningitis", "encephalitis", "malaria",
        "covid", "influenza", "ebola", "cholera", "dengue",
    ],
    "Symptom": [
        "fever", "pain", "headache", "fatigue", "nausea", "vomiting",
        "dizziness", "shortness of breath", "chest pain", "cough",
        "rash", "itching", "swelling", "inflammation", "bleeding",
        "bruising", "weight loss", "weight gain", "appetite loss",
        "night sweats", "chills", "tremor", "seizure", "paralysis",
        "numbness", "tingling", "weakness", "blurred vision",
        "hearing loss", "tinnitus", "difficulty swallowing",
        "abdominal pain", "diarrhea", "constipation", "bloating",
        "joint pain", "muscle pain", "back pain", "neck pain",
        "sore throat", "runny nose", "congestion", "sneezing",
        "insomnia", "confusion", "memory loss", "mood swings",
        "anxiety", "palpitations", "irregular heartbeat", "edema",
    ],
    "Treatment": [
        "chemotherapy", "radiation", "surgery", "transplant",
        "immunotherapy", "hormone therapy", "targeted therapy",
        "dialysis", "physical therapy", "occupational therapy",
        "psychotherapy", "cognitive behavioral therapy", "cbt",
        "medication", "antibiotic", "antiviral", "antifungal",
        "vaccine", "vaccination", "insulin", "steroids",
        "anti-inflammatory", "painkiller", "analgesic",
        "blood transfusion", "bone marrow transplant",
        "stem cell therapy", "gene therapy", "laser therapy",
        "biopsy", "endoscopy", "colonoscopy", "mri", "ct scan",
        "x-ray", "ultrasound", "ecg", "eeg", "blood test",
        "rehabilitation", "palliative care", "hospice",
        "oxygen therapy", "ventilator", "pacemaker", "stent",
        "bypass surgery", "angioplasty", "mastectomy", "amputation",
    ],
}

# Pre-compile regex patterns for efficiency
_PATTERNS = {
    category: [
        (term, re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE))
        for term in terms
    ]
    for category, terms in MEDICAL_ENTITIES.items()
}


def extract_entities(text: str) -> list:
    """
    Find all medical entities in text.

    Returns a list of dicts: {term, category, start, end}
    Sorted by start position; overlapping matches are deduplicated.
    """
    matches = []
    covered = set()  # track character positions already matched

    for category, term_patterns in _PATTERNS.items():
        for term, pattern in term_patterns:
            for m in pattern.finditer(text):
                span = range(m.start(), m.end())
                if not any(pos in covered for pos in span):
                    matches.append({
                        "term": m.group(),
                        "category": category,
                        "start": m.start(),
                        "end": m.end(),
                    })
                    covered.update(span)

    matches.sort(key=lambda x: x["start"])
    return matches


def highlight_entities(text: str, entities: list) -> str:
    """
    Wrap recognized entities in colored <mark> HTML spans.
    Returns an HTML string safe for st.markdown(unsafe_allow_html=True).
    """
    if not entities:
        return text

    result = []
    cursor = 0
    for ent in entities:
        result.append(text[cursor:ent["start"]])
        color = CATEGORY_COLORS.get(ent["category"], "#e0e0e0")
        result.append(
            f'<mark style="background:{color};padding:1px 4px;border-radius:3px;'
            f'font-weight:500" title="{ent["category"]}">{ent["term"]}</mark>'
        )
        cursor = ent["end"]
    result.append(text[cursor:])
    return "".join(result)
