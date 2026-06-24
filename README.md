# News Intelligence System

An end-to-end NLP pipeline for processing news content — from multi-outlet RSS collection through to topic classification, named-entity extraction, deduplication, extractive summarisation, vector search, and **cross-source event clustering**.

## Pipeline stages

| Stage | What it does | Module |
|-------|-------------|--------|
| **Collect** | Fetches articles from multiple outlets (BBC, Guardian, Sky, NPR) | `pipeline/collect.py` |
| **Preprocess** | Tokenise, remove stopwords, lemmatise | `pipeline/preprocess.py` |
| **Classify** | Keyword-weighted TF-IDF topic classification | `pipeline/classify.py` |
| **NER** | spaCy named-entity extraction + trending entities | `pipeline/ner.py` |
| **Deduplicate** | Remove near-duplicate articles (title cosine similarity) | `pipeline/deduplicate.py` |
| **Cluster** | Group articles into events; consensus terms + tone divergence | `pipeline/cluster.py` |
| **Summarise** | TF-IDF sentence scoring (extractive) | `pipeline/summarise.py` |
| **Search** | TF-IDF cosine similarity retrieval | `pipeline/search.py` |

### Cross-source event clustering

A reader watching the wires doesn't want forty cards for one story — they want the distinct events, who's covering each, and where coverage diverges. The clustering stage (`pipeline/cluster.py`):

1. **Groups** articles by underlying event using agglomerative clustering on TF-IDF cosine distance, with a distance threshold so the number of events is discovered from the data rather than fixed in advance.
2. **Surfaces consensus terms** per event — terms shared across at least two articles, ranked by weight — to show the common ground.
3. **Flags tone divergence** using VADER sentiment: when outlets cover the same event with a substantially different tone (compound-score spread ≥ 0.5), the event is flagged.

The signal is named *tone divergence*, not "contradiction": VADER measures sentiment, not logical entailment, and claim-level contradiction detection would require an NLI model. The naming reflects what the method can actually support.

## Run locally

```bash
git clone https://github.com/datakeep64/news-intelligence-system
cd news-intelligence-system
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
news-intelligence-system/
├── app.py              # Streamlit UI (Feed / Events / Search / Analyse tabs)
├── pipeline/
│   ├── collect.py      # Multi-outlet RSS collection
│   ├── preprocess.py   # NLTK text cleaning
│   ├── classify.py     # Topic classification
│   ├── ner.py          # spaCy named-entity extraction
│   ├── deduplicate.py  # Near-duplicate removal
│   ├── cluster.py      # Cross-source event clustering
│   ├── summarise.py    # Extractive summarisation
│   └── search.py       # TF-IDF vector search
├── tests/
│   └── test_cluster.py # Unit tests for event clustering
└── requirements.txt
```

## Tests

```bash
pip install pytest
pytest tests/ -q
```

## Part of

[datakeep64.github.io](https://datakeep64.github.io) — Applied AI portfolio
