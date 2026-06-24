"""
Sentiment Analysis Module
=========================
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) from NLTK to
classify customer messages as positive, negative, or neutral.

Why VADER over a transformer model:
- Designed for short conversational text (ideal for customer queries)
- Runs fully offline — no API quota consumed
- ~2 MB footprint vs ~500 MB for transformer-based models
- Compound score gives a continuous confidence measure in [-1, 1]

Threshold convention (VADER standard):
    compound >= 0.05  → positive
    compound <= -0.05 → negative
    otherwise         → neutral
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

import nltk
nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Single shared instance — avoids reloading the lexicon on every call
_sia = SentimentIntensityAnalyzer()

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a customer message.

    Parameters
    ----------
    text : str
        Raw customer query string.

    Returns
    -------
    dict with keys:
        label    : 'positive', 'negative', or 'neutral'
        compound : float in [-1.0, 1.0] — overall sentiment strength
        scores   : full VADER output (neg, neu, pos, compound)
    """
    scores = _sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= POSITIVE_THRESHOLD:
        label = "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "compound": compound, "scores": scores}
