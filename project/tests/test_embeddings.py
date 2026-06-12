"""Embedding and FAISS tests."""

import numpy as np

from src.embeddings.chunking import chunk_text


def test_chunk_text_fixed():
    text = " ".join(["word"] * 200)
    chunks = chunk_text(text, strategy="fixed", chunk_size=50, max_tokens=50)
    assert len(chunks) >= 2


def test_chunk_text_semantic():
    text = "First sentence. Second sentence. Third sentence here."
    chunks = chunk_text(text, strategy="semantic", chunk_size=5, max_tokens=3)
    assert len(chunks) >= 1


def test_encode_single_with_mock(monkeypatch):
    from src.embeddings.encoder import EmbeddingEncoder

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return np.array([[1.0, 0.0, 0.0]])

    encoder = EmbeddingEncoder(model_name="test-model")
    monkeypatch.setattr(encoder, "_model", FakeModel())
    vec = encoder.encode_single("hello world", review_id=None)
    assert vec.shape[0] == 3
