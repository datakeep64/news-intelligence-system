import socket
import feedparser

# Multiple outlets (not just BBC sections) so cross-source event clustering can
# surface genuine multi-outlet coverage of the same story. Feeds that fail to
# fetch are skipped gracefully, so the list can grow without breaking the app.
FEEDS = [
    ("BBC Top Stories", "http://feeds.bbci.co.uk/news/rss.xml"),
    ("BBC Technology", "http://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("BBC Business", "http://feeds.bbci.co.uk/news/business/rss.xml"),
    ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("The Guardian World", "https://www.theguardian.com/world/rss"),
    ("The Guardian Business", "https://www.theguardian.com/uk/business/rss"),
    ("The Guardian Technology", "https://www.theguardian.com/uk/technology/rss"),
    ("Sky News", "https://feeds.skynews.com/feeds/rss/home.xml"),
    ("NPR News", "https://feeds.npr.org/1001/rss.xml"),
]

_TIMEOUT_SECONDS = 10


def fetch_articles(max_per_feed: int = 8) -> list[dict]:
    articles = []
    original_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_TIMEOUT_SECONDS)

    try:
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
    finally:
        socket.setdefaulttimeout(original_timeout)

    return articles
