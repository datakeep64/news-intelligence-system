import feedparser

FEEDS = [
    ("BBC Top Stories", "http://feeds.bbci.co.uk/news/rss.xml"),
    ("BBC Technology", "http://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("BBC Business", "http://feeds.bbci.co.uk/news/business/rss.xml"),
    ("BBC Science & Environment", "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
    ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
]


def fetch_articles(max_per_feed: int = 8) -> list[dict]:
    articles = []
    for source, url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "").strip()
                summary = entry.get("summary", "").strip()
                if title:
                    articles.append({
                        "title": title,
                        "summary": summary,
                        "link": entry.get("link", ""),
                        "source": source,
                        "published": entry.get("published", ""),
                    })
        except Exception:
            continue
    return articles
