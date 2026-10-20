"""TF-IDF retrieval over the document set - the "R" in this RAG pipeline.
Sentence-level chunks so the hallucination checker gets focused context
rather than whole documents."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from documents import DOCUMENTS


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def build_chunks() -> list[str]:
    chunks = []
    for doc in DOCUMENTS.values():
        chunks.extend(_split_sentences(doc))
    return chunks


class Retriever:
    def __init__(self):
        self.chunks = build_chunks()
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in top_idx if scores[i] > 0]
