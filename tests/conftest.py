import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEmbedder:
    """Stands in for SentenceTransformer so tests need no model download."""

    def encode(self, messages):
        return np.zeros((len(messages), 384), dtype=np.float32)


class FakeClassifier:
    """Always predicts the first class with a configurable confidence."""

    classes_ = np.array(["Critical Error", "HTTP Status", "Security Alert"])

    def __init__(self, confidence=0.9):
        self.confidence = confidence

    def predict_proba(self, embeddings):
        p = self.confidence
        rest = (1 - p) / 2
        return np.tile([p, rest, rest], (len(embeddings), 1))


@pytest.fixture
def fake_models(monkeypatch):
    import processor_bert

    def install(confidence=0.9):
        monkeypatch.setattr(processor_bert, "_load_models", lambda: (FakeEmbedder(), FakeClassifier(confidence)))

    install()
    return install
