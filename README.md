# Quant Dashboard — Python / Git / Linux for Finance

Streamlit dashboard oriented toward quant/finance use cases:

- analyze a single asset (stocks/indices/crypto)
- build and backtest a multi-asset portfolio
- compare multiple simple strategies (SMA, RSI, MACD, Bollinger, Golden Cross, Buy & Hold)
- generate predictions (simple models + uncertainty intervals)
- produce a daily report via a `cron/` script

> The project mainly relies on yfinance for historical prices and Streamlit for the interface.

---

## Features Overview

### Page 1 — Home (`app.py`)
- Project overview, pages, and modules

### Page 2 — Single Asset (`pages/2_Single_Asset.py`)
- Ticker selection (US equities, indices, crypto)
- “Live” price (latest available price via yfinance)
- Download and display of historical data (OHLC)
- Strategy application:
  - Buy & Hold
  - SMA (moving averages)
  - RSI
  - MACD
  - Bollinger Bands
  - Golden Cross
- Strategy vs Buy & Hold comparison
- Metrics table (Sharpe, Sortino, annualized volatility, max drawdown)
- Multi-strategy comparison + summary table
- “AI reco” module: search for optimal parameters (simple optimization, e.g. maximize Sortino)

### Page 3 — Portfolio (`pages/3_Portfolio.py`)
- Portfolio creation (allocation per ticker, duplicate merging, optional equal weighting)
- Construction of a portfolio “pseudo-asset” (Close series = portfolio value)
- Global window + non-overlapping segments (entry/exit periods)
- Buy & Hold portfolio backtest vs selected strategies
- Gated/segmented equity curves + metrics (Sharpe, Sortino, Vol, MaxDD, performance)

### Cron — Daily Report (`cron/daily_report.py`)
- Generates a text report per ticker (stored in `cron/data/`)
- Scheduling example provided in `cron/crontab.txt`

---

## Repository Structure & Usage
```
Python-Git-Linux-for-Finance/
├── app.py                          # Streamlit home page
├── pages/
│   ├── 2_Single_Asset.py           # Single asset analysis
│   └── 3_Portfolio.py              # Portfolio analysis
├── modules/
│   ├── data_loader.py              # Price loading (yfinance) + Streamlit cache
│   ├── preprocessing.py            # Cleaning/date slicing, gated equity
│   ├── strategy_single.py          # Strategies + metrics (Sharpe/Sortino/…)
│   ├── plots.py                    # Plotly charts for equity/segments
│   ├── prediction.py               # Single asset prediction (RF/linear + CI)
│   ├── portfolio.py                # Portfolio construction + segments
│   ├── predictions_portfolio.py    # Portfolio prediction models
│   └── ai_reco.py                  # Parameter recommendation (simple grid, e.g. Sortino)
├── .streamlit/
│   ├── config.toml                 # Theme + server.runOnSave
│   └── secrets.toml                # secrets (do not commit)
├── cron/
│   ├── daily_report.py             # Reporting script
│   ├── crontab.txt                 # Crontab example
│   └── data/                       # Generated reports
├── requirements.txt                # Runtime dependencies
└── requirements-dev.txt            # Dev tools (optional)

Usage:

Single Asset:
1. Select a ticker (e.g. AAPL, ^GSPC, BTC-USD)
2. Choose a time period
3. Select a strategy and parameters
4. Analyze performance and metrics, compare strategies
5. Test predictions (linear model / random forest)

Portfolio:
1. Add multiple tickers with allocations
2. Generate the portfolio time series
3. Define a global window and segments
4. Compare Buy & Hold vs strategies
```

---

## Installation (Local)

### Prerequisites
- Python 3.10+
- up-to-date pip

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate          # Windows PowerShell
python -m pip install -U pip

Installation:

1) Create virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate          # Windows PowerShell
python -m pip install -U pip

2) Install dependencies
pip install -r requirements.txt

Optional (dev):
pip install -r requirements-dev.txt

Run app:
streamlit run app.py
```
