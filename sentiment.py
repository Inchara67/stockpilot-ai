from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def analyze_news(headlines: list) -> float:
    """
    Returns average compound VADER sentiment score across all headlines.
    Score range: -1.0 (very negative) to +1.0 (very positive).
    Returns 0 if no headlines provided.
    """
    if not headlines:
        return 0.0

    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    return sum(scores) / len(scores)
