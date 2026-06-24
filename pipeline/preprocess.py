import re
import string
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_PUNCT = set(string.punctuation)


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return word_tokenize(text.lower())


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOP_WORDS and t not in _PUNCT and len(t) > 2]


def lemmatize(tokens: list[str]) -> list[str]:
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def run(text: str) -> dict:
    cleaned = clean(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    lemmatized = lemmatize(filtered)
    return {
        "original_length": len(text.split()),
        "cleaned": cleaned,
        "tokens": tokens,
        "filtered": filtered,
        "lemmatized": lemmatized,
        "processed_text": " ".join(lemmatized),
    }
