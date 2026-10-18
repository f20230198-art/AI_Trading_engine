"""
DOOM ENGINE — Analytics Database

A read-only SQLite database built from the engine's own records, so the Analyst
agent can answer business questions with SQL instead of from model memory.

Sources (all real, nothing synthesized):
  - results/analytics/backtest_trades_2y.csv  181 closed trades from `run_backtest.py --period 2y`
  - results/backtest_2y.json                   daily equity curve of that same run
  - results/validation.json                    in-sample / out-of-sample / full-period metrics
  - results/analytics/spy_regime_daily.csv     SPY closes + regime label per day (see
                                               build_analytics_db.py --refresh-regime)
  - data/conflicts.db                          live agent conflicts + risk-enforcer audit trail
  - data/performance.db                        live paper trades (optional, often empty)

SCHEMA_DOCS below is the single source of truth for the schema. It is used three
ways: as documentation, as the data dictionary the LLM sees, and as the
table/column allow-list the SQL guard enforces. A column added to the database
but not documented here is invisible to the agent by construction.
"""
import json
import logging
import os
import sqlite3
from typing import Dict, Optional

import pandas as pd

from config import SECTOR_MAP

logger = logging.getLogger("doom.analytics")

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DATA_DIR = os.path.join(ROOT_DIR, "data")
ANALYTICS_DIR = os.path.join(RESULTS_DIR, "analytics")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "analytics.db")

# ---------------------------------------------------------------------------
# Data dictionary — what the LLM sees, and what the SQL guard allows
# ---------------------------------------------------------------------------
SCHEMA_DOCS: Dict[str, Dict] = {
    "symbols": {
        "kind": "table",
        "description": "Watchlist symbols and the sector each belongs to.",
        "columns": {
            "symbol": "Ticker, e.g. 'AAPL'.",
            "sector": "One of 'technology', 'energy', 'financials'.",
        },
    },
    "trades": {
        "kind": "table",
        "description": (
            "One row per trade. source='backtest' rows are the 181 closed trades of the "
            "2-year backtest (2024-12 to 2026-09). source='live' rows are paper trades "
            "from the running engine (may be empty)."
        ),
        "columns": {
            "trade_id": "Unique integer id.",
            "source": "'backtest' or 'live'.",
            "symbol": "Ticker traded. Join to symbols.symbol for sector.",
            "strategy": "'MEAN_REVERSION' or 'MOMENTUM' (live trades may also be 'MANUAL_ORDER').",
            "originating_agent": "Agent that proposed the trade ('backtest_engine' for backtest rows).",
            "status": "'closed' or 'open'.",
            "entry_date": "Date the position was opened, 'YYYY-MM-DD'.",
            "exit_date": "Date the position was closed, 'YYYY-MM-DD' (NULL if open).",
            "entry_price": "Fill price at entry, USD, after slippage.",
            "exit_price": "Fill price at exit, USD, after slippage.",
            "qty": "Shares.",
            "pnl": "Realized profit/loss in USD, net of costs.",
            "pnl_pct": "Realized return on the position in percent (5.0 means +5%).",
            "costs": "Commission paid on the round trip, USD.",
            "exit_reason": "'STOP_LOSS', 'TAKE_PROFIT', 'TIME_EXIT' or 'END_OF_TEST'.",
            "holding_days": "Calendar days between entry and exit.",
        },
    },
    "equity_daily": {
        "kind": "table",
        "description": "Daily portfolio equity of the 2-year backtest, starting at $100,000.",
        "columns": {
            "date": "Trading day, 'YYYY-MM-DD'.",
            "equity": "Total portfolio value in USD at that day's close.",
        },
    },
    "market_regime_daily": {
        "kind": "table",
        "description": (
            "Market regime per trading day, computed from SPY with the engine's own "
            "volatility regime classifier using only data up to that day."
        ),
        "columns": {
            "date": "Trading day, 'YYYY-MM-DD'.",
            "spy_close": "SPY adjusted close, USD.",
            "regime": "'TRENDING', 'RANGING', 'HIGH_VOLATILITY' or 'CRISIS'.",
        },
    },
    "backtest_windows": {
        "kind": "table",
        "description": (
            "Headline metrics per validation window: 'IN-SAMPLE (2y)', "
            "'OUT-OF-SAMPLE (3y)', 'FULL PERIOD (5y)'. Benchmark is SPY buy-and-hold."
        ),
        "columns": {
            "window_name": "Window label.",
            "start_date": "First day of the window, 'YYYY-MM-DD'.",
            "end_date": "Last day of the window, 'YYYY-MM-DD'.",
            "total_return_pct": "Strategy total return, percent.",
            "cagr_pct": "Strategy compound annual growth rate, percent.",
            "sharpe_ratio": "Strategy annualized Sharpe ratio.",
            "sortino_ratio": "Strategy annualized Sortino ratio.",
            "max_drawdown_pct": "Strategy maximum peak-to-trough drawdown, percent.",
            "total_trades": "Closed trades in the window.",
            "win_rate_pct": "Share of trades with pnl > 0, percent.",
            "profit_factor": "Gross profit / gross loss.",
            "benchmark_total_return_pct": "SPY buy-and-hold total return, percent.",
            "benchmark_sharpe_ratio": "SPY buy-and-hold Sharpe ratio.",
            "benchmark_max_drawdown_pct": "SPY buy-and-hold max drawdown, percent.",
        },
    },
    "backtest_strategy_breakdown": {
        "kind": "table",
        "description": "Per-strategy results inside each validation window.",
        "columns": {
            "window_name": "Window label, matches backtest_windows.window_name.",
            "strategy": "'MEAN_REVERSION' or 'MOMENTUM'.",
            "trades": "Closed trades.",
            "win_rate_pct": "Percent of winning trades.",
            "total_pnl": "Total realized P&L, USD.",
            "avg_pnl": "Average P&L per trade, USD.",
        },
    },
    "agent_conflicts": {
        "kind": "table",
        "description": (
            "Live audit trail of disagreements between agents and of risk-enforcer "
            "rejections/clamps. Small: only what the live engine has logged so far. "
            "Text columns are free text written by LLM agents."
        ),
        "columns": {
            "conflict_id": "Unique integer id.",
            "timestamp": "UTC time, ISO-8601.",
            "agent_a": "First agent, usually the proposer.",
            "agent_a_position": "What agent_a wanted, e.g. 'BUY LMT'.",
            "agent_a_reasoning": "agent_a's free-text reasoning.",
            "agent_b": "Second agent, usually the reviewer.",
            "agent_b_position": "What agent_b wanted.",
            "agent_b_reasoning": "agent_b's free-text reasoning.",
            "symbol": "Ticker in dispute.",
            "verdict": "Outcome, e.g. 'PENDING', 'REJECTED', 'CLAMPED', 'APPROVED'.",
            "verdict_reasoning": "Free-text explanation of the verdict.",
            "resolved_by": "'orchestrator' or 'risk_enforcer'.",
        },
    },
    # --- Views: pre-aggregated answers to the common business questions ---------
    "trades_enriched": {
        "kind": "view",
        "description": "trades joined with sector and the market regime on the entry date.",
        "columns": {
            "trade_id": "See trades.", "source": "See trades.", "symbol": "See trades.",
            "sector": "Sector of the symbol.", "strategy": "See trades.",
            "entry_date": "See trades.", "exit_date": "See trades.",
            "exit_month": "Exit month, 'YYYY-MM'.",
            "entry_regime": "Market regime on the entry date.",
            "pnl": "See trades.", "pnl_pct": "See trades.", "qty": "See trades.",
            "exit_reason": "See trades.", "holding_days": "See trades.",
            "is_win": "1 if pnl > 0 else 0.",
        },
    },
    "perf_by_month": {
        "kind": "view",
        "description": "Closed-trade performance grouped by exit month (all sources).",
        "columns": {
            "exit_month": "'YYYY-MM'.", "trades": "Closed trades.",
            "wins": "Trades with pnl > 0.", "win_rate_pct": "wins / trades, percent.",
            "total_pnl": "Sum of pnl, USD.", "avg_pnl_pct": "Mean pnl_pct, percent.",
        },
    },
    "perf_by_sector": {
        "kind": "view",
        "description": "Closed-trade performance grouped by sector.",
        "columns": {
            "sector": "Sector.", "trades": "Closed trades.", "wins": "Winning trades.",
            "win_rate_pct": "Percent.", "total_pnl": "USD.", "avg_pnl_pct": "Percent.",
        },
    },
    "perf_by_symbol": {
        "kind": "view",
        "description": "Closed-trade performance grouped by symbol.",
        "columns": {
            "symbol": "Ticker.", "sector": "Sector.", "trades": "Closed trades.",
            "wins": "Winning trades.", "win_rate_pct": "Percent.",
            "total_pnl": "USD.", "avg_pnl_pct": "Percent.",
        },
    },
    "perf_by_regime": {
        "kind": "view",
        "description": "Closed-trade performance grouped by the market regime on the entry date.",
        "columns": {
            "entry_regime": "Regime.", "trades": "Closed trades.", "wins": "Winning trades.",
            "win_rate_pct": "Percent.", "total_pnl": "USD.", "avg_pnl_pct": "Percent.",
        },
    },
    "trade_outcomes": {
        "kind": "view",
        "description": "Closed trades grouped by how they exited.",
        "columns": {
            "exit_reason": "Exit reason.", "trades": "Closed trades.",
            "total_pnl": "USD.", "avg_pnl": "Mean pnl, USD.",
            "avg_holding_days": "Mean holding period, days.",
        },
    },
    "drawdowns": {
        "kind": "view",
        "description": "Backtest equity with its running peak and drawdown from that peak, per day.",
        "columns": {
            "date": "'YYYY-MM-DD'.", "equity": "USD.",
            "running_peak": "Highest equity up to and including this day, USD.",
            "drawdown_pct": "(running_peak - equity) / running_peak * 100.",
        },
    },
    "risk_events": {
        "kind": "view",
        "description": "Risk-enforcer rejections and clamps (subset of agent_conflicts).",
        "columns": {
            "conflict_id": "See agent_conflicts.", "timestamp": "UTC ISO-8601.",
            "symbol": "Ticker.", "verdict": "'REJECTED' or 'CLAMPED'.",
            "reason": "Free-text reason the limit fired.",
        },
    },
    "agent_decision_outcomes": {
        "kind": "view",
        "description": (
            "Each agent conflict with any live trade opened in the same symbol on the "
            "same day, to compare decisions with outcomes. NULL trade columns mean no "
            "trade followed."
        ),
        "columns": {
            "conflict_id": "See agent_conflicts.", "decision_date": "'YYYY-MM-DD'.",
            "symbol": "Ticker.", "proposer": "agent_a.", "proposal": "agent_a_position.",
            "reviewer": "agent_b.", "verdict": "Verdict.",
            "trade_id": "Matching live trade, or NULL.", "trade_pnl": "Its pnl, or NULL.",
        },
    },
}

# Free-text columns written by LLM agents. Their values are untrusted: the
# Analyst fences them and never follows instructions found inside them.
UNTRUSTED_TEXT_COLUMNS = {
    ("agent_conflicts", "agent_a_reasoning"),
    ("agent_conflicts", "agent_b_reasoning"),
    ("agent_conflicts", "verdict_reasoning"),
    ("agent_conflicts", "agent_a_position"),
    ("agent_conflicts", "agent_b_position"),
    ("risk_events", "reason"),
}

_DDL = """
CREATE TABLE symbols (symbol TEXT PRIMARY KEY, sector TEXT NOT NULL);
CREATE TABLE trades (
    trade_id INTEGER PRIMARY KEY, source TEXT NOT NULL, symbol TEXT NOT NULL,
    strategy TEXT, originating_agent TEXT, status TEXT NOT NULL,
    entry_date TEXT, exit_date TEXT, entry_price REAL, exit_price REAL, qty REAL,
    pnl REAL, pnl_pct REAL, costs REAL, exit_reason TEXT, holding_days INTEGER
);
CREATE TABLE equity_daily (date TEXT PRIMARY KEY, equity REAL NOT NULL);
CREATE TABLE market_regime_daily (date TEXT PRIMARY KEY, spy_close REAL, regime TEXT NOT NULL);
CREATE TABLE backtest_windows (
    window_name TEXT PRIMARY KEY, start_date TEXT, end_date TEXT,
    total_return_pct REAL, cagr_pct REAL, sharpe_ratio REAL, sortino_ratio REAL,
    max_drawdown_pct REAL, total_trades INTEGER, win_rate_pct REAL, profit_factor REAL,
    benchmark_total_return_pct REAL, benchmark_sharpe_ratio REAL, benchmark_max_drawdown_pct REAL
);
CREATE TABLE backtest_strategy_breakdown (
    window_name TEXT, strategy TEXT, trades INTEGER, win_rate_pct REAL,
    total_pnl REAL, avg_pnl REAL
);
CREATE TABLE agent_conflicts (
    conflict_id INTEGER PRIMARY KEY, timestamp TEXT, agent_a TEXT, agent_a_position TEXT,
    agent_a_reasoning TEXT, agent_b TEXT, agent_b_position TEXT, agent_b_reasoning TEXT,
    symbol TEXT, verdict TEXT, verdict_reasoning TEXT, resolved_by TEXT
);

CREATE VIEW trades_enriched AS
SELECT t.trade_id, t.source, t.symbol, s.sector, t.strategy, t.entry_date, t.exit_date,
       substr(t.exit_date, 1, 7) AS exit_month, r.regime AS entry_regime,
       t.pnl, t.pnl_pct, t.qty, t.exit_reason, t.holding_days,
       CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END AS is_win
FROM trades t
LEFT JOIN symbols s ON s.symbol = t.symbol
LEFT JOIN market_regime_daily r ON r.date = t.entry_date
WHERE t.status = 'closed';

CREATE VIEW perf_by_month AS
SELECT exit_month, COUNT(*) AS trades, SUM(is_win) AS wins,
       ROUND(100.0 * SUM(is_win) / COUNT(*), 1) AS win_rate_pct,
       ROUND(SUM(pnl), 2) AS total_pnl, ROUND(AVG(pnl_pct), 3) AS avg_pnl_pct
FROM trades_enriched GROUP BY exit_month;

CREATE VIEW perf_by_sector AS
SELECT sector, COUNT(*) AS trades, SUM(is_win) AS wins,
       ROUND(100.0 * SUM(is_win) / COUNT(*), 1) AS win_rate_pct,
       ROUND(SUM(pnl), 2) AS total_pnl, ROUND(AVG(pnl_pct), 3) AS avg_pnl_pct
FROM trades_enriched GROUP BY sector;

CREATE VIEW perf_by_symbol AS
SELECT symbol, sector, COUNT(*) AS trades, SUM(is_win) AS wins,
       ROUND(100.0 * SUM(is_win) / COUNT(*), 1) AS win_rate_pct,
       ROUND(SUM(pnl), 2) AS total_pnl, ROUND(AVG(pnl_pct), 3) AS avg_pnl_pct
FROM trades_enriched GROUP BY symbol, sector;

CREATE VIEW perf_by_regime AS
SELECT entry_regime, COUNT(*) AS trades, SUM(is_win) AS wins,
       ROUND(100.0 * SUM(is_win) / COUNT(*), 1) AS win_rate_pct,
       ROUND(SUM(pnl), 2) AS total_pnl, ROUND(AVG(pnl_pct), 3) AS avg_pnl_pct
FROM trades_enriched GROUP BY entry_regime;

CREATE VIEW trade_outcomes AS
SELECT exit_reason, COUNT(*) AS trades, ROUND(SUM(pnl), 2) AS total_pnl,
       ROUND(AVG(pnl), 2) AS avg_pnl, ROUND(AVG(holding_days), 1) AS avg_holding_days
FROM trades_enriched GROUP BY exit_reason;

CREATE VIEW drawdowns AS
SELECT date, equity, running_peak,
       ROUND((running_peak - equity) / running_peak * 100, 3) AS drawdown_pct
FROM (SELECT date, equity, MAX(equity) OVER (ORDER BY date) AS running_peak FROM equity_daily);

CREATE VIEW risk_events AS
SELECT conflict_id, timestamp, symbol, verdict, verdict_reasoning AS reason
FROM agent_conflicts WHERE resolved_by = 'risk_enforcer';

CREATE VIEW agent_decision_outcomes AS
SELECT c.conflict_id, substr(c.timestamp, 1, 10) AS decision_date, c.symbol,
       c.agent_a AS proposer, c.agent_a_position AS proposal, c.agent_b AS reviewer,
       c.verdict, t.trade_id, t.pnl AS trade_pnl
FROM agent_conflicts c
LEFT JOIN trades t ON t.source = 'live' AND t.symbol = c.symbol
                  AND t.entry_date = substr(c.timestamp, 1, 10);
"""


def _load_backtest_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = "backtest"
    df["originating_agent"] = "backtest_engine"
    df["status"] = "closed"
    return df


def _load_live_trades(db_path: str) -> pd.DataFrame:
    """Live paper trades from performance.db, mapped onto the analytics schema."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM trades", conn)
        except Exception:
            return pd.DataFrame()
    if df.empty:
        return df
    return pd.DataFrame({
        "source": "live",
        "symbol": df["symbol"],
        "strategy": df["originating_strategy"],
        "originating_agent": df["originating_agent"],
        "status": df["status"],
        "entry_date": df["entry_time"].str[:10],
        "exit_date": df["exit_time"].str[:10],
        "entry_price": df["entry_price"],
        "exit_price": df["exit_price"],
        "qty": df["qty"],
        "pnl": df["pnl"],
        "pnl_pct": df["pnl_pct"],
        "costs": None,
        "exit_reason": None,
        "holding_days": None,
    })


def _load_conflicts(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM conflicts ORDER BY id", conn)
        except Exception:
            return pd.DataFrame()
    return df.rename(columns={
        "id": "conflict_id", "agent_a_name": "agent_a", "agent_b_name": "agent_b",
    })


def _load_windows(path: str):
    with open(path, encoding="utf-8") as f:
        windows = json.load(f)["windows"]
    rows, breakdown = [], []
    for name, w in windows.items():
        s, b = w["strategy"], w["benchmark"]
        rows.append({
            "window_name": name, "start_date": w["start"], "end_date": w["end"],
            "total_return_pct": s["total_return_pct"], "cagr_pct": s["cagr_pct"],
            "sharpe_ratio": s["sharpe_ratio"], "sortino_ratio": s["sortino_ratio"],
            "max_drawdown_pct": s["max_drawdown_pct"], "total_trades": s["total_trades"],
            "win_rate_pct": s["win_rate_pct"], "profit_factor": s["profit_factor"],
            "benchmark_total_return_pct": b["total_return_pct"],
            "benchmark_sharpe_ratio": b["sharpe_ratio"],
            "benchmark_max_drawdown_pct": b["max_drawdown_pct"],
        })
        for strategy, st in s.get("by_strategy", {}).items():
            breakdown.append({
                "window_name": name, "strategy": strategy, "trades": st["trades"],
                "win_rate_pct": st["win_rate"], "total_pnl": st["total_pnl"],
                "avg_pnl": st["avg_pnl"],
            })
    return pd.DataFrame(rows), pd.DataFrame(breakdown)


def _append(conn: sqlite3.Connection, table: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    columns = [c for c in SCHEMA_DOCS[table]["columns"] if c in df.columns]
    df[columns].to_sql(table, conn, if_exists="append", index=False)


def build_analytics_db(
    db_path: str = DEFAULT_DB_PATH,
    results_dir: str = RESULTS_DIR,
    data_dir: str = DATA_DIR,
) -> Dict[str, int]:
    """
    (Re)build the analytics database from source files. Writes to a temp file and
    swaps it in, so a reader never sees a half-built database. Returns row counts.
    """
    analytics_dir = os.path.join(results_dir, "analytics")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    tmp_path = db_path + ".building"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    conn = sqlite3.connect(tmp_path)
    try:
        conn.executescript(_DDL)

        _append(conn, "symbols", pd.DataFrame(
            [{"symbol": s, "sector": sector} for sector, syms in SECTOR_MAP.items() for s in syms]
        ))

        trade_frames = [
            _load_backtest_trades(os.path.join(analytics_dir, "backtest_trades_2y.csv")),
            _load_live_trades(os.path.join(data_dir, "performance.db")),
        ]
        trades = pd.concat([f for f in trade_frames if not f.empty], ignore_index=True)
        trades.insert(0, "trade_id", range(1, len(trades) + 1))
        _append(conn, "trades", trades)

        with open(os.path.join(results_dir, "backtest_2y.json"), encoding="utf-8") as f:
            _append(conn, "equity_daily", pd.DataFrame(json.load(f)["equity_curve"]))

        _append(conn, "market_regime_daily",
                pd.read_csv(os.path.join(analytics_dir, "spy_regime_daily.csv")))

        windows, breakdown = _load_windows(os.path.join(results_dir, "validation.json"))
        _append(conn, "backtest_windows", windows)
        _append(conn, "backtest_strategy_breakdown", breakdown)

        _append(conn, "agent_conflicts", _load_conflicts(os.path.join(data_dir, "conflicts.db")))

        conn.commit()
        counts = {
            name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            for name, doc in SCHEMA_DOCS.items() if doc["kind"] == "table"
        }
    finally:
        conn.close()

    os.replace(tmp_path, db_path)
    logger.info(f"Analytics DB built at {db_path}: {counts}")
    return counts


def connect_readonly(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Open the analytics DB read-only, twice over: the file is opened with
    SQLite's mode=ro (the OS-level handle cannot write), and query_only makes
    the engine itself refuse any write statement.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Analytics DB not found at {db_path}; run build_analytics_db.py")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    return conn


def ensure_analytics_db(db_path: str = DEFAULT_DB_PATH) -> str:
    """Build the DB on first use if it does not exist yet."""
    if not os.path.exists(db_path):
        build_analytics_db(db_path)
    return db_path


def render_data_dictionary(include_views: bool = True) -> str:
    """The schema as the LLM sees it: one block per table/view with column docs."""
    lines = []
    for name, doc in SCHEMA_DOCS.items():
        if doc["kind"] == "view" and not include_views:
            continue
        lines.append(f"{doc['kind'].upper()} {name}: {doc['description']}")
        for col, desc in doc["columns"].items():
            lines.append(f"  - {col}: {desc}")
        lines.append("")
    return "\n".join(lines).strip()


def compute_regime_daily(spy_close: pd.Series, lookback: Optional[int] = None) -> pd.DataFrame:
    """
    Label every day with the engine's own volatility regime classifier, using
    only the `lookback` closes ending on that day (no look-ahead).
    """
    from config import TRADING_CONFIG
    from core.regime_detector import RegimeDetector

    lookback = lookback or TRADING_CONFIG.regime_lookback_days
    detector = RegimeDetector()
    rows = []
    for i in range(lookback, len(spy_close) + 1):
        window = spy_close.iloc[i - lookback:i]
        rows.append({
            "date": window.index[-1].strftime("%Y-%m-%d"),
            "spy_close": round(float(window.iloc[-1]), 2),
            "regime": detector._volatility_regime(pd.DataFrame({"close": window.values})),
        })
    return pd.DataFrame(rows)
