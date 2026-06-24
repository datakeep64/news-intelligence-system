CATEGORIES: dict[str, list[str]] = {
    "Technology": [
        "technology", "ai", "software", "digital", "computer", "data",
        "internet", "tech", "robot", "algorithm", "cyber", "innovation",
        "semiconductor", "smartphone", "app", "cloud", "chip", "quantum",
    ],
    "Business": [
        "business", "economy", "market", "stock", "trade", "company",
        "investment", "profit", "revenue", "bank", "finance", "gdp",
        "inflation", "merger", "acquisition", "shares", "nasdaq", "ftse",
    ],
    "Politics": [
        "government", "election", "parliament", "president", "minister",
        "political", "senate", "congress", "vote", "policy", "law",
        "treaty", "democracy", "opposition", "prime minister", "party",
    ],
    "Health": [
        "health", "medical", "disease", "vaccine", "hospital", "treatment",
        "patient", "drug", "pandemic", "surgery", "doctor", "nhs", "cancer",
        "virus", "clinical", "mental health", "therapy", "pharmaceutical",
    ],
    "Science": [
        "science", "research", "study", "discovery", "climate", "environment",
        "space", "nasa", "physics", "biology", "chemistry", "experiment",
        "fossil", "planet", "carbon", "emission", "asteroid", "genome",
    ],
    "Sports": [
        "sport", "football", "soccer", "basketball", "cricket", "tennis",
        "olympic", "championship", "player", "team", "match", "league",
        "tournament", "athlete", "coach", "goal", "transfer", "world cup",
    ],
    "World": [
        "war", "conflict", "military", "troops", "refugee", "humanitarian",
        "united nations", "diplomat", "sanctions", "protest", "crisis",
        "attack", "border", "aid", "ceasefire", "nato", "invasion",
    ],
}


def classify(text: str) -> tuple[str, dict[str, float]]:
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for category, keywords in CATEGORIES.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = round(hits / len(keywords), 4)

    if max(scores.values()) == 0:
        return "General", scores

    return max(scores, key=scores.get), scores
