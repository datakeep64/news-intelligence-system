import spacy

_NLP = None

# Labels we care about for news
_KEEP = {"PERSON", "ORG", "GPE", "NORP"}
_LABELS = {
    "PERSON": "People",
    "ORG":    "Organisations",
    "GPE":    "Places",
    "NORP":   "Groups",
}


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extract_entities(text: str) -> dict[str, list[str]]:
    nlp = _get_nlp()
    # Cap at 1000 chars — sufficient for a headline + summary, avoids slow processing
    doc = nlp(text[:1000])
    entities: dict[str, list[str]] = {}
    seen: set[str] = set()
    for ent in doc.ents:
        if ent.label_ not in _KEEP:
            continue
        key = ent.text.strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        entities.setdefault(ent.label_, []).append(key)
    return entities


def trending_entities(articles: list[dict], top_n: int = 8) -> dict[str, list[tuple[str, int]]]:
    """Count entity frequency across all articles. Returns top_n per type."""
    counts: dict[str, dict[str, int]] = {}
    for article in articles:
        text = f"{article['title']} {article.get('summary', '')}"
        for label, names in extract_entities(text).items():
            for name in names:
                counts.setdefault(label, {})
                counts[label][name] = counts[label].get(name, 0) + 1

    result = {}
    for label, freq in counts.items():
        sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        if sorted_items:
            result[_LABELS.get(label, label)] = sorted_items
    return result
