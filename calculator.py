import numpy as np
import pandas as pd
from data_loader import get_monthly_history


def run_monte_carlo(
    stock: str,
    investment: float,
    months: int,
    simulations: int = 1000
) -> dict:
    """
    Runs a Monte Carlo simulation to estimate the range of portfolio outcomes.

    Uses historical monthly returns (mean + std) to generate `simulations`
    random price paths over `months` holding period.

    Returns a dict with:
        - paths:        np.array of shape (simulations, months+1) — all price paths
        - final_values: np.array of final portfolio values across all simulations
        - percentiles:  dict with p10, p25, p50, p75, p90 final values
        - prob_profit:  probability (0–1) of ending in profit
        - mean_return:  expected mean return %
        - investment:   original investment amount (for reference)
        - monthly_returns_used: the historical monthly returns used
    """
    data = get_monthly_history(stock)

    if data.empty or len(data) < 12:
        return None

    # Calculate historical monthly returns
    monthly_returns = data['Close'].pct_change().dropna()

    mu = float(monthly_returns.mean())    # average monthly return
    sigma = float(monthly_returns.std())     # monthly volatility

    # Monte Carlo simulation — O(simulations × months)
    # Each path starts at investment and compounds randomly each month
    rng = np.random.default_rng(seed=42)
    paths = np.zeros((simulations, months + 1))
    paths[:, 0] = investment

    for m in range(1, months + 1):
        random_returns = rng.normal(mu, sigma, simulations)
        paths[:, m] = paths[:, m - 1] * (1 + random_returns)

    final_values = paths[:, -1]

    percentiles = {
        "p10": float(np.percentile(final_values, 10)),
        "p25": float(np.percentile(final_values, 25)),
        "p50": float(np.percentile(final_values, 50)),   # median
        "p75": float(np.percentile(final_values, 75)),
        "p90": float(np.percentile(final_values, 90)),
    }

    prob_profit = float(np.mean(final_values > investment))
    mean_return = float(
        (np.mean(final_values) - investment) / investment * 100)

    return {
        "paths":                 paths,
        "final_values":          final_values,
        "percentiles":           percentiles,
        "prob_profit":           prob_profit,
        "mean_return":           mean_return,
        "investment":            investment,
        "mu":                    mu,
        "sigma":                 sigma,
        "monthly_returns_used":  monthly_returns,
    }


def format_currency(value: float, currency: str) -> str:
    """Format a number in Indian lakh/crore style for ₹, or standard for $."""
    if currency == "₹":
        if value >= 1_00_00_000:
            return f"₹{value / 1_00_00_000:.2f} Cr"
        elif value >= 1_00_000:
            return f"₹{value / 1_00_000:.2f} L"
        else:
            return f"₹{value:,.2f}"
    else:
        if value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        else:
            return f"${value:,.2f}"
