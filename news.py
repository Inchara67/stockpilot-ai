import logging
import yfinance as yf

logger = logging.getLogger(__name__)


def _parse(raw_news: list, max_articles: int) -> list:
    filtered = []
    for item in raw_news[:max_articles]:
        content = item.get("content", {})
        title = content.get("title") or item.get("title", "")
        source = content.get("provider", {}).get("displayName") \
            or item.get("publisher", "")
        url = (
            content.get("canonicalUrl", {}).get("url")
            or item.get("link")
            or item.get("url", "")
        )
        if title and url:
            filtered.append({"title": title, "url": url, "source": source})
    return filtered


def fetch_news(ticker: str, max_articles: int = 10, query: str = None) -> list:
    """
    Fetch news from yfinance for a ticker. If nothing comes back and a
    search query is given (e.g. an index name), fall back to a news search.
    """
    articles = []

    try:
        raw_news = yf.Ticker(ticker).news
        if raw_news:
            articles = _parse(raw_news, max_articles)
    except Exception as e:
        logger.exception("[news] ticker news failed for %s: %s", ticker, e)

    # Fallback: search by name (needed for indices like ^NSEI)
    if not articles and query:
        try:
            raw_news = yf.Search(query, news_count=max_articles).news
            if raw_news:
                articles = _parse(raw_news, max_articles)
        except Exception as e:
            logger.exception(
                "[news] search fallback failed for %s: %s", query, e)

    if not articles:
        logger.warning("[news] No news found for %s", ticker)

    logger.info("[news] %d articles fetched for %s", len(articles), ticker)
    return articles
