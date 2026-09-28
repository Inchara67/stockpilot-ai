# StockPilot AI

**StockPilot AI** is an interactive stock analysis and forecasting application built with **Python and Streamlit**. It combines machine learning, real-time market data, financial news sentiment, and Monte Carlo simulation to provide a single interface for exploring stock price trends and potential investment outcomes.

The application supports both **US stocks** and **Indian stocks listed on NSE/BSE**.

> **Disclaimer:** StockPilot AI is an experimental analytical tool and does not provide financial advice or guarantee future market performance.

---

## Features

### AI Price Prediction

StockPilot AI uses a **Random Forest Regressor** trained on historical daily stock data to predict the next closing price.

The model uses technical features including:

* Closing Price
* 10-day Moving Average
* 50-day Moving Average
* Daily Return
* 10-day Volatility
* 10-day Momentum
* 10-day Average Volume

These features are generated consistently during both training and prediction to avoid feature-scale mismatches.

The Random Forest model uses **100 estimators** and is persisted using Joblib for reuse.

---

### Real-Time Market Data

Market data is retrieved through **yfinance**.

The application supports:

* US stocks
* NSE stocks
* BSE stocks
* 5-minute intraday data
* Daily historical data
* Monthly historical data
* 5 years of historical data for model training

For Indian stocks, the application automatically maps symbols to `.NS` for NSE and `.BO` for BSE.

---

### Intraday Candlestick Dashboard

The application provides an interactive **5-day / 5-minute candlestick chart** with:

* Open price
* High price
* Low price
* Closing price
* Trading volume
* Today's opening price
* Day high/low
* Price change

Charts are rendered using **Plotly**.

---

### Financial News & Sentiment Analysis

Stock-specific financial news is retrieved using **yfinance**, eliminating the need for a separate news API key.

The application analyzes article headlines using **VADER Sentiment Analysis** and calculates an average compound sentiment score ranging from **-1 to +1**.

The interface categorizes the resulting sentiment as:

* Positive
* Neutral
* Negative

The prediction and sentiment signals are then combined to generate an application-level market signal.

---

### AI Decision Signal

StockPilot AI compares the predicted next closing price with the current price and combines that signal with news sentiment.

Possible outputs include:

* Strong Buy
* Buy
* Hold
* Sell
* Strong Sell

A signal confidence value is also displayed based on the relative difference between the predicted and current prices.

---

### Monte Carlo Investment Simulator

The built-in investment calculator performs **1,000 Monte Carlo simulations** using historical monthly returns and volatility.

Users can specify:

* Investment amount
* Holding period from 1–60 months

The simulator generates:

* 10th percentile outcome
* 25th percentile outcome
* Median outcome
* 75th percentile outcome
* 90th percentile outcome
* Probability of profit
* Expected mean return
* Simulated portfolio paths
* Final value distribution

The simulation uses the historical mean monthly return and standard deviation to generate random monthly return paths.

---

## Model Training Pipeline

StockPilot AI automatically trains a model when a stock is analyzed for the first time.

The training workflow is:

```text
Historical Market Data
        ↓
Feature Engineering
        ↓
Technical Indicators
        ↓
Random Forest Regressor
        ↓
Saved .pkl Model
        ↓
Next Closing Price Prediction
```

Training runs in a **background thread**, allowing the Streamlit interface to remain responsive. Models are retrained only when the existing model is missing or older than the configured retraining interval of **24 hours**.

---

## Architecture

```text
                     ┌─────────────────────┐
                     │      Streamlit      │
                     │         UI          │
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
       Market Data         News Data       User Inputs
        (yfinance)         (yfinance)      Investment
              │                 │                 │
              ▼                 ▼                 ▼
       Feature Engine      VADER NLP       Monte Carlo
              │            Sentiment         Engine
              ▼                 │                 │
       Random Forest            │                 │
          Model                 │                 │
              │                 │                 │
              └────────────┬────┴─────────────────┘
                           ▼
                  Analysis Dashboard
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        Prediction     Sentiment     Simulation
          Signal        Signal        Outcomes
```

---

## Project Structure

```text
StockPilot-AI/
│
├── app.py                  # Streamlit application and dashboard
├── data_loader.py          # Market data retrieval
├── features.py             # Feature engineering
├── model.py                # Random Forest training/loading
├── trainer.py              # Background model training
├── utils.py                # Price prediction utilities
├── news.py                 # Financial news retrieval
├── sentiment.py            # VADER sentiment analysis
├── calculator.py           # Monte Carlo investment simulation
│
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Git exclusions
│
└── models/                 # Generated trained models
    ├── *.pkl
    └── *_progress.txt
```

The project dependencies include Streamlit, yfinance, scikit-learn, Joblib, NumPy, Pandas, Plotly, Requests, and VADER Sentiment.

---

## Tech Stack

| Category             | Technology              |
| -------------------- | ----------------------- |
| Language             | Python                  |
| Frontend / Dashboard | Streamlit               |
| Machine Learning     | Scikit-learn            |
| ML Model             | Random Forest Regressor |
| Market Data          | yfinance                |
| Data Processing      | Pandas, NumPy           |
| Visualization        | Plotly                  |
| Sentiment Analysis   | VADER                   |
| Model Persistence    | Joblib                  |
| Simulation           | Monte Carlo             |
| Version Control      | Git / GitHub            |

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/stockpilot-ai.git
cd stockpilot-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Example Workflow

```text
Select Market
     ↓
Select Exchange (NSE/BSE for India)
     ↓
Enter Stock Symbol
     ↓
Validate Stock
     ↓
Fetch Historical Data
     ↓
Train / Load Random Forest Model
     ↓
Generate Next-Close Prediction
     ↓
Fetch Intraday Data
     ↓
Fetch Financial News
     ↓
Calculate Sentiment
     ↓
Generate Signal
     ↓
Run Monte Carlo Simulation
```

---

## Important Notes

* The machine learning model predicts the **next closing price**, not long-term stock performance.
* The Monte Carlo simulator is based on historical monthly returns and volatility and represents a range of simulated outcomes rather than guaranteed results.
* Historical market behavior does not guarantee future performance.
* The application is intended for **educational and analytical purposes** and should not be treated as professional financial advice.

---

## Future Improvements

Potential extensions for the project include:

* Additional technical indicators
* Model comparison between Random Forest, XGBoost and other forecasting approaches
* Backtesting prediction performance
* Historical prediction accuracy tracking
* Portfolio-level analysis
* Risk-adjusted performance metrics
* More advanced NLP models for financial sentiment
* Automated model evaluation and retraining
* Cloud deployment and persistent model storage

---

## Disclaimer

**StockPilot AI is an experimental machine-learning project for educational and analytical purposes. Predictions and simulations are generated from historical data and statistical assumptions and should not be considered financial advice.**
