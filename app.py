import streamlit as st
import os
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import time

from data_loader import get_live, get_daily_recent
from utils import predict
from trainer import start_training, get_training_progress
from sentiment import analyze_news
from news import fetch_news
from calculator import run_monte_carlo, format_currency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PAGE CONFIG

st.set_page_config(
    page_title="AI Stock Advisor",
    page_icon="📈",
    layout="wide"
)
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');

.stApp {
    background-color: #0a0a0a;
}

.block-container {
    padding-top: 2rem !important;
}

/* Container */
.container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 40vh;  
    flex-direction: column;
    gap: 10px;
    padding-top: 60px;  
}

/* Main Text */
.glitch-text {
    font-family: 'Orbitron', sans-serif;
    font-size: 70px;
    color: #dbeafe;
    text-shadow: 
        0 0 8px rgba(59,130,246,0.3),
        0 0 20px rgba(30,58,138,0.2);
    position: relative;
    letter-spacing: 2px;

    animation: smoothGlitch 1.4s ease-out forwards;
}

/* Smooth glitch */
@keyframes smoothGlitch {
    0% {
        filter: blur(8px);
        opacity: 0;
        transform: scaleX(1.2);
    }
    30% {
        filter: blur(4px);
        opacity: 0.7;
    }
    60% {
        filter: blur(2px);
    }
    100% {
        filter: blur(0.6px);
        opacity: 1;
        transform: scaleX(1);
    }
}

/* Smear layers */
.glitch-text::before,
.glitch-text::after {
    content: "AI Stock Advisor";
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    opacity: 0.5;
}

.glitch-text::before {
    color: #3b82f6;
    filter: blur(2px);
    clip-path: inset(30% 0 50% 0);
    animation: smear 1.2s ease-out forwards;
}

.glitch-text::after {
    color: #93c5fd;
    filter: blur(3px);
    clip-path: inset(60% 0 20% 0);
    animation: smear 1.4s ease-out forwards;
}

@keyframes smear {
    0% {
        transform: translateX(-40px);
        opacity: 0;
    }
    50% {
        opacity: 0.6;
    }
    100% {
        transform: translateX(0);
        opacity: 0.2;
    }
}

/* STOCK DIVIDER */
.divider {
    position: relative;
    width: 60%;
    height: 2px;
    overflow: hidden;
}

/* line growth */
.divider::before {
    content: "";
    position: absolute;
    width: 0%;
    height: 100%;
    background: linear-gradient(90deg, #888, #ccc, #888);
    animation: stockGrow 1.5s ease-out forwards;
}

/* moving highlight */
.divider::after {
    content: "";
    position: absolute;
    top: -1px;
    left: -20%;
    width: 20%;
    height: 4px;
    background: rgba(255,255,255,0.4);
    filter: blur(2px);
    animation: sweep 1.5s ease-out;
}

@keyframes stockGrow {
    0% {
        width: 0%;
        transform: translateY(0);
        opacity: 0;
    }
    30% {
        width: 30%;
        transform: translateY(-2px);
    }
    60% {
        width: 60%;
        transform: translateY(2px);
    }
    100% {
        width: 100%;
        transform: translateY(0);
        opacity: 1;
    }
}

@keyframes sweep {
    0% {
        left: -20%;
        opacity: 0;
    }
    50% {
        opacity: 1;
    }
    100% {
        left: 100%;
        opacity: 0;
    }
}

/* subtitle */
.subtitle {
    text-align: center;
    font-size: 14px;
    color: gray;
    margin: 0;
}
/* STOCK TREND DIVIDER */
.trend-divider {
    width: 60%;
    margin: 30px auto;
    position: relative;
}

/* SVG line animation */
.trend-divider svg {
    width: 100%;
    height: 60px;
}

/* line path */
.trend-path {
    fill: none;
    stroke: #60a5fa;
    stroke-width: 2;

    stroke-dasharray: 300;
    stroke-dashoffset: 300;

    animation: drawLine 1.8s ease-out forwards;
}

/* moving dot */
.trend-dot {
    fill: #bfdbfe;
    r: 4;
    filter: drop-shadow(0 0 6px rgba(96,165,250,0.6));

    animation: moveDot 1.8s ease-out forwards;
}

/* draw animation */
@keyframes drawLine {
    to {
        stroke-dashoffset: 0;
    }
}

/* dot follows path manually */
@keyframes moveDot {
    0%   { transform: translate(0px, 40px); opacity: 0; }
    20%  { transform: translate(60px, 25px); opacity: 1; }
    40%  { transform: translate(120px, 35px); }
    60%  { transform: translate(180px, 15px); }
    80%  { transform: translate(240px, 25px); }
    100% { transform: translate(300px, 10px); }
}
.divider {
    position: relative;
    width: 60%;
    height: 1px;
    background: #333;
    margin: 40px auto;
    overflow: hidden;
}

.divider::after {
    content: "";
    position: absolute;
    top: 0;
    left: -50%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, #ffffff, transparent);
    animation: sweep 2s ease-out;
}

@keyframes sweep {
    0% {
        left: -50%;
        opacity: 0;
    }
    50% {
        opacity: 1;
    }
    100% {
        left: 100%;
        opacity: 0;
    }
}
</style>

<div class="container">
    <div class="glitch-text">AI Stock Advisor</div>
    <div class="trend-divider">
    <svg viewBox="0 0 300 60" preserveAspectRatio="none">
        <!-- zig-zag stock line -->
        <path class="trend-path"
              d="M0 40 L60 25 L120 35 L180 15 L240 25 L300 10" />
              <circle class="trend-dot" cx="0" cy="0"></circle>
    </svg>
</div>
    <p class="subtitle">
        Real-time predictions · News sentiment · Investment calculator
    </p>
</div>

""", unsafe_allow_html=True)


st.markdown("""
<style>

.divider {
    position: relative;
    width: 60%;
    height: 1px;
    background: #333;
    margin: 40px auto;
    overflow: hidden;
}

.divider::after {
    content: "";
    position: absolute;
    top: 0;
    left: -50%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, #ffffff, transparent);
    animation: sweep 2s ease-out;
}

@keyframes sweep {
    0% {
        left: -50%;
        opacity: 0;
    }
    50% {
        opacity: 1;
    }
    100% {
        left: 100%;
        opacity: 0;
    }
}

</style>

<div class="divider"></div>

""", unsafe_allow_html=True)

# MARKET + EXCHANGE

col_market, col_exchange, col_ticker = st.columns([1, 1, 2])

with col_market:
    market = st.selectbox("🌐 Market", ["US 🇺🇸", "India 🇮🇳"])

with col_exchange:
    if "India" in market:
        exchange = st.selectbox("🏦 Exchange", ["NSE", "BSE"])
        currency = "₹"
    else:
        exchange = None
        currency = "$"

with col_ticker:
    placeholder_hint = "e.g. RELIANCE" if "India" in market else "e.g. AAPL"

    stock_input = st.text_input(
        "🔠 Stock Symbol",
        value="",
        placeholder=placeholder_hint
    )
    if not stock_input.strip():
        st.info("👆🏼 Enter a stock symbol above to get started.")
        st.stop()

# BUILD FULL TICKER

if "India" in market:
    suffix = ".NS" if exchange == "NSE" else ".BO"
    stock = stock_input + suffix
else:
    stock = stock_input

# STOCK VALIDATION


@st.cache_data(ttl=0)
def validate_stock(ticker: str) -> bool:
    try:
        data = yf.download(ticker, period="5d", progress=False)
        return not data.empty
    except Exception as e:
        logger.exception("Failed to validate ticker %s: %s", ticker, e)
        return False


if not stock_input:
    st.info("👆 Enter a stock symbol above to get started.")
    st.stop()

with st.spinner(f"Validating {stock}..."):
    valid = validate_stock(stock)

if not valid:
    st.error(
        f"🚫 **{stock_input}** not found in {'Indian' if 'India' in market else 'US'} market. Check the symbol and try again.")
    st.stop()

# STOCK HEADER

try:
    info = yf.Ticker(stock).info
    company_name = info.get("longName") or info.get("shortName") or stock_input
    sector = info.get("sector", "")
    industry = info.get("industry", "")
except Exception as e:
    logger.exception("Failed to fetch company info for %s: %s", stock, e)
    company_name = stock_input
    sector = ""
    industry = ""

st.subheader(f"{company_name}  `{stock}`")
if sector:
    st.caption(f"📂 {sector}   ·  {industry}")

st.divider()

# MODEL TRAINING

model_path = f"models/{stock}.pkl"
start_training(stock)

progress = get_training_progress(stock)

if not os.path.exists(model_path):
    st.warning("⏳ Training model for the first time — this may take a minute...")
    prog_bar = st.progress(5)
    st.caption("Fetching 5 years of historical data and training RandomForest...")

    while True:
        progress = get_training_progress(stock)
        if progress == -1:
            st.error("❌ Model training failed. Please try again.")
            st.stop()
        prog_bar.progress(min(max(progress, 0), 100))
        if progress >= 100:
            break

        time.sleep(1)

    st.success("✅ Model ready!")
    st.rerun()

# PREDICTION METRICS

pred = predict(stock)
live = get_live(stock)
current = None

st.subheader("📊 Price Prediction")

m1, m2, m3, m4 = st.columns(4)

if not live.empty:
    current = float(live['Close'].iloc[-1])
    m1.metric("Current Price", format_currency(current, currency))

if pred is not None and current is not None:
    change = pred - current
    change_pct = (change / current) * 100
    m2.metric(
        "Predicted Next Close",
        format_currency(pred, currency),
        delta=f"{'+' if change >= 0 else ''}{change_pct:.2f}%"
    )

    # Intraday high/low
    today_data = live[live.index.date == live.index[-1].date()]
    if not today_data.empty:
        m3.metric("Today High",  format_currency(
            float(today_data['High'].max()), currency))
        m4.metric("Today Low",   format_currency(
            float(today_data['Low'].min()), currency))

    # Trend signal
    if change > 0:
        st.success("📈 **Uptrend Expected** — Predicted price is above current")
    else:
        st.error("📉 **Downtrend Expected** — Predicted price is below current")
else:
    st.warning("⚠️ Not enough data to generate prediction yet.")

st.divider()

# INTRADAY CHART

st.subheader("📉 Intraday Price (5-Day · 5-Min)")

if live.empty:
    st.info("📭 Market is closed. Chart will appear during trading hours.")
else:
    chart_data = live.copy()
    if isinstance(chart_data.columns, pd.MultiIndex):
        chart_data.columns = chart_data.columns.get_level_values(0)
    chart_data.index = pd.to_datetime(chart_data.index, utc=True)
    chart_data.index = chart_data.index.tz_localize(None)
    chart_data = chart_data[~chart_data.index.duplicated(
        keep='last')].sort_index()
    chart_data = chart_data[chart_data['Volume'] > 0]
    chart_data = chart_data.dropna(subset=['Open', 'High', 'Low', 'Close'])

    if len(chart_data) < 10:
        st.info("📭 Market is closed. Chart will appear during trading hours.")
    else:
        time_range = (chart_data.index[-1] -
                      chart_data.index[0]).total_seconds()

        if time_range < 600:
            st.info("📭 Market is closed. Chart will appear during trading hours.")
        else:
            candle_colors = [
                '#26a69a' if row['Close'] >= row['Open'] else '#ef5350'
                for _, row in chart_data.iterrows()
            ]

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                increasing_fillcolor='#26a69a',
                decreasing_fillcolor='#ef5350',
            ))

            fig.add_trace(go.Bar(
                x=chart_data.index,
                y=chart_data['Volume'],
                name='Volume',
                marker_color=candle_colors,
                opacity=0.4,
                yaxis='y2'
            ))

            today = chart_data[chart_data.index.date ==
                               chart_data.index[-1].date()]
            if not today.empty:
                open_price = float(today['Open'].iloc[0])
                fig.add_hline(
                    y=open_price,
                    line_dash="dot",
                    line_color="rgba(255,255,255,0.3)",
                    annotation_text=f"Today's Open: {currency}{open_price:.2f}",
                    annotation_position="top left",
                    annotation_font_color="rgba(255,255,255,0.5)"
                )

            fig.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False,
                    rangeslider=dict(visible=False),
                    type='date',
                    tickformat='%b %d\n%H:%M',
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.07)',
                    side='right',
                    tickprefix=currency,
                ),
                yaxis2=dict(
                    overlaying='y',
                    side='left',
                    showgrid=False,
                    showticklabels=False,
                    range=[0, chart_data['Volume'].max() * 4]
                ),
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            s1, s2, s3, s4 = st.columns(4)
            last = chart_data.iloc[-1]
            first = chart_data.iloc[0]
            day_change = float(last['Close'] - first['Open'])
            day_change_pct = day_change / float(first['Open']) * 100

            s1.metric(
                "Open",     f"{currency}{float(today['Open'].iloc[0]):.2f}" if not today.empty else "—")
            s2.metric(
                "Day High", f"{currency}{float(chart_data['High'].max()):.2f}")
            s3.metric(
                "Day Low",  f"{currency}{float(chart_data['Low'].min()):.2f}")
            s4.metric(
                "Change",   f"{currency}{day_change:.2f}", delta=f"{day_change_pct:.2f}%")

st.divider()

# NEWS + SENTIMENT

col_news, col_sentiment = st.columns([2, 1])

with col_news:
    st.subheader("📰 Market News")
    news = fetch_news(stock)

    if not news:
        st.warning("No relevant financial news found.")
    else:
        for article in news:
            st.markdown(f"""
            <div style="
                padding:12px 16px;
                border-radius:10px;
                margin-bottom:10px;
                background-color:rgba(255,255,255,0.03);
                border:1px solid rgba(255,255,255,0.1);
            ">
                <b>{article['title']}</b><br>
                <span style="color:gray;font-size:0.85em;">{article['source']}</span> &nbsp;
                <a href="{article['url']}" target="_blank" style="font-size:0.85em;">Read more →</a>
            </div>
            """, unsafe_allow_html=True)

with col_sentiment:
    st.subheader("🎭 Sentiment")
    headlines = [n["title"] for n in news] if news else []
    sentiment_score = analyze_news(headlines)

    if sentiment_score > 0.2:
        st.success(f"📈 Positive\n\n**Score: {sentiment_score:.2f}**")
    elif sentiment_score < -0.2:
        st.error(f"📉 Negative\n\n**Score: {sentiment_score:.2f}**")
    else:
        st.info(f"⚖️ Neutral\n\n**Score: {sentiment_score:.2f}**")

    # AI Decision
    st.subheader("🤖 AI Decision")
    if pred is not None and current is not None:
        if pred > current and sentiment_score > 0.2:
            st.success("📈 **STRONG BUY**")
        elif pred < current and sentiment_score < -0.2:
            st.error("📉 **STRONG SELL**")
        elif pred > current:
            st.info("📈 **BUY** *(Weak Signal)*")
        elif pred < current:
            st.info("📉 **SELL** *(Weak Signal)*")
        else:
            st.warning("⚖️ **HOLD**")

        confidence = min(abs(pred - current) / current * 100, 10)
        st.caption(f"Signal confidence: {confidence:.2f}%")
        st.progress(confidence / 10)
    else:
        st.warning("Not enough data for decision.")

st.divider()

# INVESTMENT CALCULATOR

st.subheader("🧮 Investment Calculator")
st.caption(
    "Monte Carlo simulation using historical volatility — not financial advice")

with st.expander("📖 How does this work?", expanded=False):
    st.markdown("""
    This calculator uses **Monte Carlo simulation** to model thousands of possible
    futures based on the stock's historical monthly return and volatility.

    - It runs **1,000 simulations** of your holding period
    - Each simulation uses a random monthly return drawn from historical data
    - The output shows a **probability range**, not a single guarantee
    - The wider the range, the more volatile the stock
    """)

calc_col1, calc_col2, calc_col3 = st.columns(3)

with calc_col1:
    investment_raw = st.number_input(
        f"💰 Investment Amount ({currency})",
        min_value=100.0,
        max_value=10_000_000.0,
        value=10000.0,
        step=500.0,
        format="%.2f"
    )

with calc_col2:
    holding_months = st.slider(
        "📅 Holding Period (months)",
        min_value=1,
        max_value=60,
        value=12,
        step=1
    )

with calc_col3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_calc = st.button("💡 Predict", use_container_width=True)

if run_calc:
    with st.spinner("Running 1,000 Monte Carlo simulations..."):
        result = run_monte_carlo(stock, investment_raw, holding_months)

    if result is None:
        st.error("Not enough historical data to run simulation.")
    else:
        p = result["percentiles"]
        inv = result["investment"]

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Worst Case (10th %)",  format_currency(p["p10"], currency),
                  delta=f"{(p['p10']-inv)/inv*100:.1f}%")
        r2.metric("Conservative (25th %)", format_currency(p["p25"], currency),
                  delta=f"{(p['p25']-inv)/inv*100:.1f}%")
        r3.metric("Median Outcome",        format_currency(p["p50"], currency),
                  delta=f"{(p['p50']-inv)/inv*100:.1f}%")
        r4.metric("Optimistic (75th %)",   format_currency(p["p75"], currency),
                  delta=f"{(p['p75']-inv)/inv*100:.1f}%")
        r5.metric("Best Case (90th %)",    format_currency(p["p90"], currency),
                  delta=f"{(p['p90']-inv)/inv*100:.1f}%")

        prob_pct = result["prob_profit"] * 100
        mean_ret = result["mean_return"]

        info_col1, info_col2 = st.columns(2)
        info_col1.info(f"📊 **Probability of Profit:** {prob_pct:.1f}%")
        info_col2.info(f"📈 **Expected Mean Return:** {mean_ret:+.2f}%")

        # Monte Carlo fan chart
        st.markdown("#### Simulation Paths")

        paths = result["paths"]
        month_axis = list(range(holding_months + 1))

        fig2 = go.Figure()

        # Plot a sample of paths (show 100 for performance)
        sample_idx = np.random.choice(
            len(paths), size=min(100, len(paths)), replace=False)
        for i in sample_idx:
            fig2.add_trace(go.Scatter(
                x=month_axis,
                y=paths[i],
                mode='lines',
                line=dict(width=0.4, color='rgba(100, 180, 255, 0.15)'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Percentile bands
        for label, pct_key, color in [
            ("10th–90th %ile", None, None),
        ]:
            fig2.add_trace(go.Scatter(
                x=month_axis + month_axis[::-1],
                y=[float(np.percentile(paths[:, m], 90)) for m in range(holding_months + 1)] +
                  [float(np.percentile(paths[:, m], 10))
                   for m in range(holding_months + 1)][::-1],
                fill='toself',
                fillcolor='rgba(100, 180, 255, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='10–90th %ile range'
            ))

        # Median line
        fig2.add_trace(go.Scatter(
            x=month_axis,
            y=[float(np.median(paths[:, m]))
               for m in range(holding_months + 1)],
            mode='lines',
            name='Median',
            line=dict(color='#00C49F', width=2.5)
        ))

        # Investment line
        fig2.add_hline(
            y=inv,
            line_dash="dash",
            line_color="rgba(255,100,100,0.7)",
            annotation_text=f"Break-even ({format_currency(inv, currency)})",
            annotation_position="top left"
        )

        fig2.update_layout(
            xaxis_title="Months",
            yaxis_title=f"Portfolio Value ({currency})",
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.15)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Final value distribution histogram
        st.markdown("#### Final Value Distribution")

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=result["final_values"],
            nbinsx=50,
            marker_color='rgba(100, 180, 255, 0.7)',
            name='Simulation outcomes'
        ))
        fig3.add_vline(x=inv,    line_dash="dash",
                       line_color="red",    annotation_text="Break-even")
        fig3.add_vline(x=p["p50"], line_dash="dot",
                       line_color="#00C49F", annotation_text="Median")

        fig3.update_layout(
            xaxis_title=f"Final Portfolio Value ({currency})",
            yaxis_title="Number of Simulations",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            f"⚠️ Based on {len(result['monthly_returns_used'])} months of historical data · "
            f"Monthly μ={result['mu']*100:.2f}%  σ={result['sigma']*100:.2f}% · "
            "Not financial advice."
        )

# MARKET HOURS

st.divider()
if "India" in market:
    st.caption(
        "🕐 NSE/BSE trading hours: 9:15 AM – 3:30 PM IST (Mon–Fri). Outside hours, prices reflect last close.")
else:
    st.caption(
        "🕐 NYSE/NASDAQ trading hours: 9:30 AM – 4:00 PM ET (Mon–Fri). Outside hours, prices reflect last close.")
