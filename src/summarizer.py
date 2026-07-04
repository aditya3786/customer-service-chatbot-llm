import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

# Lazy-loaded so the ~300MB model is only downloaded when first used.
_pipeline = None
_MODEL = "sshleifer/distilbart-cnn-12-6"
_MAX_INPUT_TOKENS = 1024


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("summarization", model=_MODEL)
    return _pipeline


def get_summary(text: str, max_length: int = 150, min_length: int = 40) -> str:
    """Return an abstractive summary of text using DistilBART (open-source, offline)."""
    # Truncate to model's max token window (rough word-based proxy)
    words = text.split()
    if len(words) > _MAX_INPUT_TOKENS:
        text = " ".join(words[:_MAX_INPUT_TOKENS])

    if len(text.strip()) < 50:
        return text.strip()

    pipe = _get_pipeline()
    result = pipe(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]["summary_text"]
