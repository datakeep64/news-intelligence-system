import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ArticleSearch:
    def __init__(self) -> None:
        # Bigrams capture phrases like "interest rates", "climate change"
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8000,
        )
        self._matrix = None
        self._articles: list[dict] = []

    def index(self, articles: list[dict]) -> None:
        self._articles = articles
        texts = [f"{a['title']} {a.get('summary', '')}" for a in articles]
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._matrix is None or not query.strip():
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self._articles[i], "score": round(float(scores[i]), 4)}
            for i in top_indices
            if scores[i] > 0
        ]

    @property
    def indexed(self) -> bool:
        return self._matrix is not None
