import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def deduplicate(articles: list[dict], threshold: float = 0.82) -> tuple[list[dict], int]:
    """
    Remove near-duplicate articles by title cosine similarity.
    The same story often appears across multiple BBC feeds.

    Returns the deduplicated list and the count of articles removed.
    """
    if len(articles) <= 1:
        return articles, 0

    titles = [a["title"] for a in articles]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(titles)

    dropped: set[int] = set()
    for i in range(len(articles)):
        if i in dropped:
            continue
        sims = cosine_similarity(matrix[i : i + 1], matrix).flatten()
        for j in range(i + 1, len(articles)):
            if j not in dropped and sims[j] >= threshold:
                dropped.add(j)

    kept = [articles[i] for i in range(len(articles)) if i not in dropped]
    return kept, len(dropped)
