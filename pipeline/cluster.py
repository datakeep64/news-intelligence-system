"""
Cross-source event clustering.

An editor watching the wires doesn't want 40 cards for one story — they want
"here are the distinct events today, who's covering each, what they agree on, and
where coverage diverges." This groups articles by underlying *event* (not just
near-duplicate titles), then for each event surfaces the sources involved, the
terms they share, and whether outlets cover it with notably different tone.

Method, kept consistent with the rest of the pipeline:
  - TF-IDF (unigram+bigram) vectors, as in search.py / classify.py
  - Agglomerative clustering on cosine distance with a distance threshold, so the
    number of events is discovered from the data rather than fixed in advance
  - VADER sentiment (nltk) to measure tone spread across sources within an event

Naming is deliberate: this measures *tone divergence*, not logical contradiction.
Claim-level contradiction detection would need an NLI model; calling a sentiment
spread a "contradiction" would overstate what the signal actually is.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# VADER is loaded lazily and defensively: if its lexicon isn't available (e.g. no
# network on first run), clustering still works — tone features just degrade to None.
_SIA = None
_SIA_TRIED = False


def _get_sia():
    global _SIA, _SIA_TRIED
    if not _SIA_TRIED:
        _SIA_TRIED = True
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment import SentimentIntensityAnalyzer
            _SIA = SentimentIntensityAnalyzer()
        except Exception:
            _SIA = None
    return _SIA


def _article_text(a: dict) -> str:
    return f"{a.get('title', '')} {a.get('summary', '')}".strip()


def _compound(text: str) -> float | None:
    sia = _get_sia()
    if sia is None:
        return None
    return round(float(sia.polarity_scores(text)["compound"]), 3)


def _consensus_terms(matrix, indices: list[int], feature_names, top_n: int = 6) -> list[str]:
    """Terms shared across the cluster, ranked by total weight.

    A consensus term must appear in at least two articles in the cluster — a term
    from a single article describes that article, not the shared event.
    """
    sub = matrix[indices]
    doc_freq = np.asarray((sub > 0).sum(axis=0)).ravel()
    weight = np.asarray(sub.sum(axis=0)).ravel()
    shared = np.where(doc_freq >= 2)[0]
    if shared.size == 0:
        shared = np.argsort(weight)[::-1][:top_n]
    ranked = shared[np.argsort(weight[shared])[::-1]][:top_n]
    return [feature_names[i] for i in ranked]


def _tone_spread(members: list[dict]) -> dict:
    """Per-source mean sentiment and the spread across sources for one event."""
    per_article = [(m["source"], _compound(_article_text(m))) for m in members]
    if any(c is None for _, c in per_article):
        return {"available": False}

    by_source: dict[str, list[float]] = {}
    for source, comp in per_article:
        by_source.setdefault(source, []).append(comp)
    per_source = {s: round(float(np.mean(v)), 3) for s, v in by_source.items()}

    values = list(per_source.values())
    spread = round(max(values) - min(values), 3) if len(values) > 1 else 0.0
    # >=0.5 on VADER's -1..1 compound scale is a substantial tone gap.
    return {
        "available": True,
        "per_source": per_source,
        "spread": spread,
        "divergent": spread >= 0.5 and len(per_source) > 1,
    }


def cluster_events(
    articles: list[dict],
    distance_threshold: float = 0.7,
    min_cluster_size: int = 2,
) -> dict:
    """
    Group articles into events.

    Returns ``{"events": [...], "singletons": [...]}`` where each event has the
    member articles, the sources covering it, a representative (most central)
    article, consensus terms, and a tone-spread summary. ``singletons`` are
    stories covered by only one article (below ``min_cluster_size``).
    """
    n = len(articles)
    if n < 2:
        return {"events": [], "singletons": list(articles)}

    texts = [_article_text(a) for a in articles]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    sim = cosine_similarity(matrix)
    distance = 1.0 - sim
    np.fill_diagonal(distance, 0.0)
    distance[distance < 0] = 0.0  # guard against tiny floating-point negatives

    model = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = model.fit_predict(distance)

    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)

    events: list[dict] = []
    singletons: list[dict] = []

    for indices in groups.values():
        if len(indices) < min_cluster_size:
            singletons.extend(articles[i] for i in indices)
            continue

        # Representative = most central article (highest total similarity to peers).
        sub_sim = sim[np.ix_(indices, indices)]
        centrality = sub_sim.sum(axis=1)
        representative = articles[indices[int(np.argmax(centrality))]]

        members = [articles[i] for i in indices]
        source_counts: dict[str, int] = {}
        for m in members:
            source_counts[m["source"]] = source_counts.get(m["source"], 0) + 1

        events.append(
            {
                "size": len(indices),
                "sources": sorted(source_counts),
                "source_counts": source_counts,
                "representative": representative,
                "members": members,
                "consensus_terms": _consensus_terms(matrix, indices, feature_names),
                "tone": _tone_spread(members),
            }
        )

    # Most-covered events first; ties broken by number of distinct sources.
    events.sort(key=lambda e: (e["size"], len(e["sources"])), reverse=True)
    return {"events": events, "singletons": singletons}
