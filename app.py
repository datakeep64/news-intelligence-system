import streamlit as st
from pipeline.collect import fetch_articles
from pipeline.classify import classify
from pipeline.summarise import extractive
from pipeline.preprocess import run as preprocess_run
from pipeline.search import ArticleSearch

st.set_page_config(
    page_title="News Intelligence System",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0d1117; }
    .topic-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-family: monospace;
        padding: 2px 10px;
        border-radius: 20px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .article-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .article-title { font-size: 0.95rem; font-weight: 600; color: #e6edf3; margin-bottom: 4px; }
    .article-summary { font-size: 0.85rem; color: #8b949e; line-height: 1.5; }
    .article-meta { font-size: 0.72rem; color: #6e7681; margin-top: 6px; font-family: monospace; }
    .score-bar { height: 4px; border-radius: 2px; background: #58a6ff; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

TOPIC_COLOURS = {
    "Technology": ("#58a6ff", "rgba(88,166,255,0.12)"),
    "Business":   ("#f0883e", "rgba(240,136,62,0.12)"),
    "Politics":   ("#d2a8ff", "rgba(210,168,255,0.12)"),
    "Health":     ("#3fb950", "rgba(63,185,80,0.12)"),
    "Science":    ("#79c0ff", "rgba(121,192,255,0.12)"),
    "Sports":     ("#ffa657", "rgba(255,166,87,0.12)"),
    "World":      ("#ff7b72", "rgba(255,123,114,0.12)"),
    "General":    ("#8b949e", "rgba(139,148,158,0.12)"),
}


def badge(topic: str) -> str:
    colour, bg = TOPIC_COLOURS.get(topic, ("#8b949e", "rgba(139,148,158,0.12)"))
    return (
        f'<span class="topic-badge" style="color:{colour};background:{bg};'
        f'border:1px solid {colour}44">{topic}</span>'
    )


@st.cache_data(ttl=600, show_spinner=False)
def _fetch(max_per_feed: int) -> list[dict]:
    return fetch_articles(max_per_feed=max_per_feed)


def _build_searcher(articles: list[dict]) -> ArticleSearch:
    searcher = ArticleSearch()
    searcher.index(articles)
    return searcher


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 News Intelligence System")
    st.caption("End-to-end NLP pipeline for news processing")
    st.divider()
    max_articles = st.slider("Articles per feed", 3, 10, 6)
    fetch_clicked = st.button("Fetch Latest Articles", type="primary", use_container_width=True)
    if fetch_clicked:
        with st.spinner("Fetching from BBC RSS feeds..."):
            fetched = _fetch(max_articles)
        st.session_state["articles"] = fetched
        st.session_state["searcher"] = _build_searcher(fetched)
        st.success(f"Fetched {len(fetched)} articles")
    st.divider()
    st.caption("Built by [Adhithi M](https://datakeep64.github.io) · [GitHub](https://github.com/datakeep64/news-intelligence-system)")

articles: list[dict] = st.session_state.get("articles", [])
searcher: ArticleSearch | None = st.session_state.get("searcher", None)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📰  Live Feed", "🔍  Search", "🔬  Analyse Text"])

# ── Tab 1: Live Feed ──────────────────────────────────────────────────────────

with tab1:
    st.subheader("Live Feed")
    st.caption("Articles fetched from BBC RSS feeds, classified by topic using TF-IDF cosine similarity.")

    if not articles:
        st.info("Click **Fetch Latest Articles** in the sidebar to load the feed.", icon="📡")
    else:
        col_info, col_filter = st.columns([1, 2])
        with col_info:
            st.caption(f"{len(articles)} articles loaded")
        with col_filter:
            topic_filter = st.multiselect(
                "Filter by topic",
                options=list(TOPIC_COLOURS.keys()),
                default=[],
                placeholder="All topics",
                label_visibility="collapsed",
            )

        shown = 0
        for article in articles:
            topic, _ = classify(f"{article['title']} {article.get('summary', '')}")
            if topic_filter and topic not in topic_filter:
                continue
            summary_text = article.get("summary", "")
            shown += 1
            st.markdown(
                f'<div class="article-card">'
                f'{badge(topic)}'
                f'<div class="article-title">{article["title"]}</div>'
                f'<div class="article-summary">{summary_text[:280]}{"…" if len(summary_text) > 280 else ""}</div>'
                f'<div class="article-meta">{article["source"]} · '
                f'<a href="{article["link"]}" target="_blank" style="color:#58a6ff">Read →</a></div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        if shown == 0:
            st.caption("No articles match the selected topics.")

# ── Tab 2: Search ─────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Semantic Search")
    st.caption("Unigram + bigram TF-IDF vector search across all fetched articles.")

    if not articles:
        st.info("Fetch articles first using the sidebar.", icon="📡")
    else:
        query = st.text_input(
            "Search query",
            placeholder="e.g. interest rates, climate change, AI regulation",
        )
        top_k = st.slider("Results to show", 1, 10, 5)

        if query and searcher:
            results = searcher.search(query, top_k=top_k)
            if not results:
                st.caption("No matching articles found for that query.")
            else:
                st.caption(f"{len(results)} results")
                for r in results:
                    topic, _ = classify(f"{r['title']} {r.get('summary', '')}")
                    score_pct = int(r["score"] * 100)
                    st.markdown(
                        f'<div class="article-card">'
                        f'{badge(topic)}'
                        f'<div class="article-title">{r["title"]}</div>'
                        f'<div class="article-summary">{r.get("summary", "")[:220]}</div>'
                        f'<div class="article-meta">{r["source"]} · relevance {score_pct}% · '
                        f'<a href="{r["link"]}" target="_blank" style="color:#58a6ff">Read →</a></div>'
                        f'<div class="score-bar" style="width:{min(score_pct * 4, 100)}%"></div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ── Tab 3: Analyse Text ───────────────────────────────────────────────────────

with tab3:
    st.subheader("Analyse Any Article")
    st.caption("Paste any article text — the pipeline runs preprocess → classify → summarise and shows each stage.")

    sample = (
        "Researchers at MIT have developed a new artificial intelligence system capable of "
        "predicting protein structures with unprecedented accuracy, surpassing existing benchmarks "
        "by a significant margin. The model, trained on over 200 million protein sequences, uses a "
        "transformer architecture combined with a novel attention mechanism that accounts for "
        "evolutionary relationships between amino acids. Scientists believe this breakthrough could "
        "accelerate drug discovery by enabling faster identification of protein targets for new "
        "treatments. The research, published in Nature, has already attracted interest from major "
        "pharmaceutical companies including Pfizer and AstraZeneca, who are exploring applications "
        "in oncology and rare genetic diseases. The team plans to release the model weights publicly "
        "by the end of the year, allowing researchers worldwide to build on their findings."
    )

    text = st.text_area("Article text", value=sample, height=200)
    n_sent = st.slider("Summary sentences", 1, 5, 3)

    if st.button("Run Pipeline", type="primary"):
        if not text.strip():
            st.warning("Paste some article text first.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**① Preprocessing**")
                result = preprocess_run(text)
                st.metric("Original word count", result["original_length"])
                st.metric("Tokens after filtering", len(result["filtered"]))
                with st.expander("Show lemmatized tokens"):
                    st.write(", ".join(result["lemmatized"][:30]))

            with col2:
                st.markdown("**② Classification**")
                topic, scores = classify(text)
                st.markdown(badge(topic), unsafe_allow_html=True)
                with st.expander("Category similarity scores"):
                    for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                        st.progress(float(score), text=f"{cat}: {score:.3f}")

            st.markdown("**③ Extractive Summary** *(position-weighted TF-IDF)*")
            summary = extractive(text, n_sentences=n_sent)
            st.info(summary)
