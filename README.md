# DOOM ENGINE

**Autonomous Multi-Agent AI Trading System**

A fully autonomous trading system where 9+ specialized AI agents collaborate in real time to analyze markets, generate signals, manage risk, and execute paper trades — with zero human intervention.

```
     ██████╗  ██████╗  ██████╗ ███╗   ███╗
     ██╔══██╗██╔═══██╗██╔═══██╗████╗ ████║
     ██║  ██║██║   ██║██║   ██║██╔████╔██║
     ██║  ██║██║   ██║██║   ██║██║╚██╔╝██║
     ██████╔╝╚██████╔╝╚██████╔╝██║ ╚═╝ ██║
     ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝     ╚═╝
           E N G I N E   v1.0
```

---

## Measured Performance

Backtested over 5 years of real market data (2021-09 → 2026-09) with transaction
costs applied. Reproduce with `python run_backtest.py --period 5y`.

| Metric | Strategy | SPY Buy & Hold |
|---|---:|---:|
| Total return | 43.98% | 78.53% |
| **Sharpe ratio** | **1.98** | 0.79 |
| **Max drawdown** | **3.52%** | 24.43% |
| Trades | 488 | 1 |
| Win rate | 50.8% | — |

**2.5× the risk-adjusted return of the S&P 500 at roughly one-seventh the
drawdown**, while trailing on absolute return — a low-volatility,
capital-preserving profile rather than an index-beating one.

**Validated out-of-sample.** Parameters were tuned on the most recent 2 years;
the earlier 3 years were held out entirely. Sharpe degrades from 2.21 to 1.74 —
a mild decline, not the collapse that overfitting produces.

Costs modeled: $0.005/share commission, 5bps slippage against every fill. No
look-ahead bias — signals use data through bar *i*, orders fill at bar *i+1*'s
open, and that guarantee is enforced by a test verified to fail when look-ahead
is deliberately injected.

See [results/RESULTS.md](results/RESULTS.md) for the full methodology, per-strategy
breakdown, and a plain statement of the limitations.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    DOOM ENGINE ORCHESTRATOR                     │
│                      (The DOOM Brain)                          │
│  Resolves conflicts, makes final GO/NO-GO, DOOM SPEAKS briefs │
├───────────────┬───────────────┬───────────────┬───────────────┤
│  Market Intel │  Sector Tech  │ Sector Energy │ Sector Fin    │
│  (Prices/Vol) │  (AAPL,NVDA)  │ (XOM,CVX)     │ (JPM,GS)      │
├───────────────┼───────────────┼───────────────┤               │
│   Sentiment   │   Strategy    │ Risk Guardian │  Portfolio    │
│  (News/RSS)   │(Mom/MR/Sent)  │ (VETO power)  │ (Rebalance)   │
├───────────────┴───────────────┴───────────────┴───────────────┤
│                     Execution Agent                            │
│              (Alpaca Paper Trading API)                         │
├───────────────────────────────────────────────────────────────┤
│  Regime Detection (HMM) │ Agent Memory (ChromaDB) │ Perf Track │
├───────────────────────────────────────────────────────────────┤
│           FastAPI Backend + WebSocket (Real-time)               │
├───────────────────────────────────────────────────────────────┤
│             War Room Dashboard (React HUD)                      │
└───────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (optional, for React build — the HTML dashboard works standalone)

### 2. Install dependencies
```bash
cd doom-engine
pip install -r requirements.txt
```

### 3. Set environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
# Recommended (free tier): Groq
LLM_PROVIDER=groq
GROQ_API_KEY_1=gsk_...
LLM_MODEL=llama-3.3-70b-versatile

# Required for real market data and trading (paper mode)
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...

API_PORT=8000
```

The LLM backend is resolved automatically in priority order — **Ollama → Groq →
Gemini → Anthropic → quant**. Ollama is used whenever it is running locally, no
config needed. If no backend is available the deterministic quant fallback
activates, which computes signals from live market data with no LLM at all.

Additional Groq keys (`GROQ_API_KEY_2`, `_3`) are rotated automatically when one
hits its rate limit.

**Get your API keys:**
- Groq (free, fast): https://console.groq.com
- Alpaca (free paper trading): https://app.alpaca.markets/signup
- Google AI Studio (optional Gemini fallback): https://aistudio.google.com/apikey
- Anthropic (optional): https://console.anthropic.com/
- NewsAPI (optional, enhances sentiment): https://newsapi.org/register

### 4. Run
```bash
python main.py
```

**No keys? It still runs.** Quotes fall back to yfinance and the agents fall back
to deterministic quant mode, so the full pipeline — live prices, news, signals,
risk review, veto enforcement, position sizing — is fully observable on the
dashboard with nothing configured. Only order placement requires Alpaca
credentials; without them the Execution agent reports the approved trades and
holds.

### 5. Open the dashboard
Navigate to **http://localhost:8000** in your browser (or whatever `API_PORT` you
set — pick a free port if 8000 is taken on your machine).

### 6. Run the tests
```bash
python -m pytest tests -q
```
All tests are offline — no network, no API keys required.

---

## What's Real vs What Needs API Keys

| Component | Status | Requires |
|-----------|--------|----------|
| All Agent LLM Reasoning | Real Groq/Gemini/Anthropic/Ollama calls | any one LLM key, else quant fallback |
| Market Data (real-time quotes) | Alpaca, falls back to yfinance | none (yfinance needs no key) |
| Historical Price Data | Alpaca + yfinance fallback | `ALPACA_API_KEY` (yfinance works without) |
| Paper Trading (order execution) | Real Alpaca Paper API | `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` |
| News/Sentiment (Alpaca News) | Real API | `ALPACA_API_KEY` |
| News/Sentiment (NewsAPI) | Real API | `NEWS_API_KEY` |
| News/Sentiment (RSS feeds) | Always works | None |
| Regime Detection (HMM) | Real statistical computation | Market data |
| Agent Memory (ChromaDB) | Real vector similarity search | `pip install chromadb sentence-transformers` |
| Agent Memory (SQLite fallback) | Keyword-based retrieval | None |
| Performance Metrics | Computed from real trade history | Trades must exist |
| Conflict Resolution | Logs real agent disagreements | Agents must be running |
| Dashboard | Fully wired to live backend | Backend must be running |

---

## Agent Descriptions

### Market Intelligence Agent
Fetches real-time prices for 22 symbols across 3 sectors from Alpaca. Monitors momentum, volume anomalies, and sector rotation patterns. Uses the configured LLM provider to analyze and generate structured signal objects.

### Sector Specialists (Tech, Energy, Financials)
Three independent agents, each analyzing their sector with real price data and technical indicators (EMA, RSI, Bollinger Bands). Each makes its own LLM call with sector-specific reasoning prompts.

### Sentiment Scanner
Pulls real headlines from Alpaca News API, NewsAPI, and financial RSS feeds (MarketWatch, Yahoo Finance). Runs LLM-powered sentiment classification on each headline. Computes per-symbol sentiment scores.

### Strategy Agent
Receives signals from all intel agents and runs 3 sub-strategies simultaneously:
- **Momentum**: EMA 12/26 crossover with volume confirmation
- **Mean Reversion**: Bollinger Band squeeze reversals
- **Sentiment Spike**: High sentiment + price breakout

Uses the configured LLM to weigh strategies against current regime and generate ranked trade proposals with sizing.

### Risk Guardian
Reviews every trade proposal before execution. Makes its own LLM call to reason about
portfolio-level risk, then enforces the hard rules **in code** rather than relying on the
model to apply them: 5% max position, 20% sector concentration, and a 3% daily drawdown
limit that vetoes all new trades when breached.

Vetoes are enforced in the execution path, not merely advised. In CRISIS regime a veto is
absolute; in other regimes the Orchestrator can override one only by setting an explicit
flag with conviction above 0.85.

### Position Sizing
Share quantities are computed deterministically from portfolio equity, live price, and
`position_size_pct` — never taken from an LLM response. When price or equity is unknown
the trade is skipped and logged rather than sized nominally, since a position that cannot
be measured cannot be shown to respect the 5% cap.

### Portfolio Rebalancing Agent
Runs every 30 cycles. Reviews current holdings vs target allocations. Generates rebalance orders if drift exceeds 5% threshold.

### Execution Agent
Places real paper trades via Alpaca API. Handles market, limit, and stop-loss orders. Confirms fills and retries with exponential backoff.

### Orchestrator (DOOM Brain)
Receives all agent outputs. Detects and logs conflicts (e.g., Strategy says BUY but Risk says VETO). Makes final go/no-go decision via an LLM call. Generates "DOOM SPEAKS" briefing after each major decision.

---

## Novel Features

### Agent Memory (Real Vector Search)
Each agent persists decisions to ChromaDB with sentence-transformer embeddings. Before each decision, agents retrieve the 3 most similar past situations via cosine similarity search.

### Regime Detection (Hidden Markov Models)
Uses `hmmlearn` to fit a 4-state Gaussian HMM on returns + volatility + momentum features. Confirmed by rolling realized volatility analysis. Regimes: TRENDING, RANGING, HIGH_VOLATILITY, CRISIS.

### Regime-Adaptive Behavior
Agent prompts dynamically change based on regime. In TRENDING, momentum weights increase. In CRISIS, Risk Guardian gets absolute veto, position sizes halve.

### Conflict Resolution with Full Audit Trail
Every disagreement is logged with both agents' reasoning and the Orchestrator's verdict. Feeds the dashboard conflict panel in real time.

### Performance Attribution
After each closed trade, P&L is attributed to the originating agent and strategy. Per-agent win rate, average return, and Sharpe contribution are computed and displayed live.

---

## Dashboard Panels

- **Agent Node Network**: 9 nodes showing live status, pulsing when active
- **Equity Curve**: Live Chart.js line chart, updates on every fill
- **DOOM SPEAKS**: Latest Orchestrator briefing with blue glow animation
- **Regime Indicator**: Live badge with color coding
- **Risk-O-Meter**: SVG arc gauge from Risk Guardian
- **Heatmap**: Per-symbol color-coded change bars
- **Ticker Bar**: Scrolling live prices
- **Trade Feed**: Recent trades with P&L
- **Conflict Log**: Agent disagreements with verdicts
- **Active Positions**: Current holdings with unrealized P&L
- **Performance Attribution**: Per-agent stats table
- **Metrics Row**: Sharpe, Win Rate, Max Drawdown — all computed live

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq llama-3.3-70b (default), with Ollama / Gemini / Anthropic / quant fallback |
| Agent Framework | Custom async Python with BaseAgent class |
| Market Data | Alpaca Markets API + yfinance fallback |
| Execution | Alpaca Paper Trading API |
| News | Alpaca News + NewsAPI + RSS feeds |
| Regime Detection | hmmlearn (HMM) + numpy |
| Agent Memory | ChromaDB + sentence-transformers |
| Backend | FastAPI + WebSocket |
| Frontend | React 18 + Tailwind CSS + Chart.js |
| Persistence | SQLite (trades, conflicts, equity) |
| Orchestration | asyncio-based 60-second cycle loop |

---

## Project Structure

```
doom-engine/
├── agents/
│   ├── base_agent.py          # Base class with LLM, memory, regime awareness
│   ├── orchestrator.py        # DOOM Brain — master coordinator
│   ├── market_intel.py        # Real-time market data analysis
│   ├── sector_tech.py         # Technology sector specialist
│   ├── sector_energy.py       # Energy sector specialist
│   ├── sector_financials.py   # Financials sector specialist
│   ├── sentiment.py           # News sentiment scanner
│   ├── strategy.py            # Multi-strategy signal generator
│   ├── risk_guardian.py       # Risk review + VETO power
│   ├── portfolio.py           # Portfolio rebalancing
│   └── execution.py           # Alpaca trade execution
├── core/
│   ├── data_feeds.py          # Market data from Alpaca + yfinance
│   ├── regime_detector.py     # HMM-based regime classification
│   ├── memory_store.py        # ChromaDB vector memory
│   ├── conflict_logger.py     # Agent conflict audit trail
│   ├── performance_tracker.py # P&L attribution + metrics
│   ├── position_sizer.py      # Deterministic sizing + veto enforcement
│   └── shared_state.py        # Central state + WebSocket broadcast
├── tests/                     # Offline regression tests (pytest)
├── backend/
│   └── api.py                 # FastAPI + WebSocket server
├── frontend/
│   └── index.html             # War Room Dashboard (React + Tailwind)
├── data/                      # Created at runtime
│   ├── memory/                # ChromaDB + SQLite agent memory
│   ├── conflicts.db           # Conflict audit trail
│   ├── performance.db         # Trade history + equity curve
│   └── doom.log               # System logs
├── main.py                    # Entry point — starts everything
├── config.py                  # Configuration + API keys
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## License

Built for competition. Use responsibly.
