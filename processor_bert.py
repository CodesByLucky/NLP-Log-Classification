"""Tier 2: sentence embeddings + Logistic Regression.

Handles log sources with enough labeled data to train a classifier.
Models are loaded lazily on first use, so importing this module is cheap.
"""
from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent / "models" / "log_classifier.joblib"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim embeddings, fast on CPU
CONFIDENCE_THRESHOLD = 0.5  # below this, return "Unclassified" instead of guessing


@lru_cache(maxsize=1)
def _load_models():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL), joblib.load(MODEL_PATH)


def classify_with_bert_batch(log_messages):
    """Classify a list of messages with a single embedding pass."""
    if not log_messages:
        return []
    embedder, classifier = _load_models()
    embeddings = embedder.encode(list(log_messages))
    probabilities = classifier.predict_proba(embeddings)

    labels = []
    for probs in probabilities:
        if probs.max() < CONFIDENCE_THRESHOLD:
            labels.append("Unclassified")
        else:
            labels.append(str(classifier.classes_[probs.argmax()]))
    return labels


def classify_with_bert(log_message):
    return classify_with_bert_batch([log_message])[0]


if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer",
    ]
    for log, label in zip(logs, classify_with_bert_batch(logs)):
        print(log, "->", label)
