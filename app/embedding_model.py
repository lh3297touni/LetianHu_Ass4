# app/embedding_model.py
from __future__ import annotations
from typing import List
import spacy
import numpy as np

class EmbeddingModel:

    def __init__(self, model_name: str = "en_core_web_md"):
        self.nlp = spacy.load(model_name)

    def get_vector(self, word: str) -> List[float]:
        doc = self.nlp(word)
        return doc[0].vector.tolist() if len(doc) else []

    def similarity(self, w1: str, w2: str) -> float:
        d1, d2 = self.nlp(w1), self.nlp(w2)
        if not len(d1) or not len(d2):
            return 0.0
        return float(d1.similarity(d2))

    def embed_text(self, text: str) -> List[float]:
        doc = self.nlp(text)
        vecs = [t.vector for t in doc if t.has_vector]
        if not vecs:
            return []
        return np.mean(vecs, axis=0).tolist()
