import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.split()) >= 6]


def extractive(text: str, n_sentences: int = 3) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= n_sentences:
        return text

    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:n_sentences])

    scores = np.asarray(tfidf.sum(axis=1)).flatten()
    top_indices = sorted(
        sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:n_sentences]
    )
    return " ".join(sentences[i] for i in top_indices)
