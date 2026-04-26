import logging
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_news(ticker: str, max_articles: int = 10) -> list:
    """
    Fetch stock-specific news directly from yfinance.
    No API key needed. Always relevant to the ticker.
    """
    try:
        t = yf.Ticker(ticker)
        raw_news = t.news  # already filtered to this stock
    except Exception as e:
        logger.exception(
            "[news] yfinance news fetch failed for %s: %s", ticker, e)
        return []

    if not raw_news:
        logger.warning("[news] No news returned by yfinance for %s", ticker)
        return []

    filtered = []
    for item in raw_news[:max_articles]:
        # yfinance news structure
        content = item.get("content", {})
        title = content.get("title") or item.get("title", "")
        source = content.get("provider", {}).get("displayName") \
            or item.get("publisher", "")

        # grab the canonical URL
        url = (
            content.get("canonicalUrl", {}).get("url")
            or item.get("link")
            or item.get("url", "")
        )

        if title and url:
            filtered.append({
                "title":  title,
                "url":    url,
                "source": source
            })

    logger.info("[news] %d articles fetched for %s", len(filtered), ticker)
    return filtered
