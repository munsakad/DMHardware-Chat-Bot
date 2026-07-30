"""
retriever.py
------------
Retrieval logic that anchors Gemini's answers in the curated knowledge base.

Uses scikit-learn's TF-IDF vectorizer + cosine similarity to find the most
relevant records for a user's question. This is fast, dependency-light,
runs fully offline/locally, and scales fine to tens of thousands of rows —
no vector database required for a dataset this size.
"""

import re
from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")

# sklearn's default English stop-word list drops a few words that are
# genuine hardware-engineering terms in this domain (e.g. "via" as in a
# PCB via, not "via email"). Keep those out of the stop-word set so they
# stay searchable.
_DOMAIN_ALLOWLIST = {"via"}


def _simple_stem(word: str) -> str:
    """
    Very small rule-based stemmer (no extra dependency / no corpus download
    needed). Good enough to fold common plural/verb-form variants together
    for retrieval purposes, e.g. vias -> via, gates -> gate, controllers -> controller.
    """
    w = word.lower()
    for suffix, replacement in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)] + replacement
    return w


def _stemming_tokenizer(text: str) -> List[str]:
    return [_simple_stem(tok) for tok in _WORD_RE.findall(text)]


# Stem the stop-word list itself (minus the domain allowlist) so it lines
# up with tokens produced by the stemming tokenizer above.
_STOP_WORDS = list({
    _simple_stem(w) for w in ENGLISH_STOP_WORDS if w not in _DOMAIN_ALLOWLIST
})


@dataclass
class RetrievedRecord:
    title: str
    category: str
    subcategory: str
    content: str
    score: float


class KnowledgeRetriever:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            tokenizer=_stemming_tokenizer,
            stop_words=_STOP_WORDS,
            ngram_range=(1, 2),
            max_df=0.9,
            token_pattern=None,
        )
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"])

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.05) -> List[RetrievedRecord]:
        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for i in top_indices:
            if scores[i] < min_score:
                continue
            row = self.df.iloc[i]
            results.append(
                RetrievedRecord(
                    title=row["title"],
                    category=row["category"],
                    subcategory=row["subcategory"],
                    content=row["content"],
                    score=float(scores[i]),
                )
            )
        return results

    @staticmethod
    def format_context(records: List[RetrievedRecord]) -> str:
        """Formats retrieved records into a context block for the Gemini prompt."""
        if not records:
            return "No directly relevant records were found in the knowledge base."

        chunks = []
        for r in records:
            chunks.append(f"[{r.category} / {r.subcategory}] {r.title}: {r.content}")
        return "\n".join(chunks)
