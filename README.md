# News Intelligence System

An end-to-end NLP pipeline for processing news content — from live RSS collection through to topic classification, extractive summarisation, and embedding-based search.

## Pipeline stages

| Stage | What it does | Module |
|-------|-------------|--------|
| **Collect** | Fetches articles from BBC RSS feeds | `pipeline/collect.py` |
| **Preprocess** | Tokenise, remove stopwords, lemmatise | `pipeline/preprocess.py` |
| **Classify** | Keyword-weighted TF-IDF topic classification | `pipeline/classify.py` |
| **Summarise** | TF-IDF sentence scoring (extractive) | `pipeline/summarise.py` |
| **Search** | TF-IDF cosine similarity retrieval | `pipeline/search.py` |

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
├── app.py              # Streamlit UI (Feed / Search / Analyse tabs)
├── pipeline/
│   ├── collect.py      # RSS feed collection
│   ├── preprocess.py   # NLTK text cleaning
│   ├── classify.py     # Topic classification
│   ├── summarise.py    # Extractive summarisation
│   └── search.py       # TF-IDF vector search
└── requirements.txt
```

## Part of

[datakeep64.github.io](https://datakeep64.github.io) — Applied AI portfolio
