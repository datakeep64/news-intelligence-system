import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Richer seed documents per category — gives TF-IDF enough vocabulary to compute
# meaningful IDF weights and capture bigrams like "artificial intelligence"
_SEEDS = {
    "Technology": (
        "technology software artificial intelligence machine learning deep learning "
        "digital computer data internet tech robot algorithm cyber innovation "
        "semiconductor smartphone app cloud chip quantum computing neural network "
        "programming coding startup cybersecurity encryption autonomous"
    ),
    "Business": (
        "business economy market stock trade company investment profit revenue bank "
        "finance gdp inflation merger acquisition shares nasdaq ftse corporate "
        "earnings quarterly interest rates supply chain logistics retail consumer "
        "spending venture capital ipo startup valuation"
    ),
    "Politics": (
        "government election parliament president prime minister political senate "
        "congress vote policy law treaty democracy opposition party legislation "
        "cabinet foreign policy referendum diplomat bilateral summit coalition "
        "campaign ballot candidate incumbent"
    ),
    "Health": (
        "health medical disease vaccine hospital treatment patient drug pandemic "
        "surgery doctor nhs cancer virus clinical mental health therapy "
        "pharmaceutical trial outbreak mortality prescription antibiotics "
        "public health epidemic chronic illness"
    ),
    "Science": (
        "science research study discovery climate environment space nasa physics "
        "biology chemistry experiment fossil planet carbon emission asteroid "
        "genome genetic renewable energy solar wind ocean temperature species "
        "extinction ecology atmosphere gravity"
    ),
    "Sports": (
        "sport football soccer basketball cricket tennis olympic championship "
        "player team match league tournament athlete coach goal transfer world cup "
        "medal race marathon boxing swimming cycling formula one rugby stadium"
    ),
    "World": (
        "war conflict military troops refugee humanitarian united nations diplomat "
        "sanctions protest crisis attack border aid ceasefire nato invasion "
        "occupation casualty peacekeeping foreign affairs geopolitical tension "
        "international relations"
    ),
}

_LABELS = list(_SEEDS.keys())

# Fit once at module load — reused for every classify() call
_VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
_CATEGORY_MATRIX = _VECTORIZER.fit_transform(list(_SEEDS.values()))


def classify(text: str) -> tuple[str, dict[str, float]]:
    if not text.strip():
        return "General", {label: 0.0 for label in _LABELS}

    query_vec = _VECTORIZER.transform([text])
    raw_scores = cosine_similarity(query_vec, _CATEGORY_MATRIX).flatten()

    total = raw_scores.sum()
    if total == 0:
        return "General", {label: 0.0 for label in _LABELS}

    # Normalise to proportions so scores across categories are comparable
    norm = raw_scores / total
    scores = {label: round(float(s), 4) for label, s in zip(_LABELS, norm)}
    best = _LABELS[int(np.argmax(raw_scores))]
    return best, scores
