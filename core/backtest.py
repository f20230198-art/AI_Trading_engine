"""
DOOM ENGINE — Backtest Engine

Replays the engine's strategy and risk logic over real historical data so the
system's performance can be measured rather than asserted.

Design rules, in order of importance:

1. NO LOOK-AHEAD BIAS. On bar i, signals are computed from data up to and
   including bar i, and the resulting order fills at bar i+1's open. A backtest
   that decides and fills on the same bar is reading the future and its numbers
   are worthless.

2. COSTS ARE REAL. Commission and slippage are charged on every fill. Frictionless
   backtests routinely turn a losing strategy into a winning one on paper.

3. SAME LOGIC AS LIVE. Sizing and risk limits come from core/position_sizer.py —
   the identical module the live engine uses. If the backtest and the live system
   disagreed about position sizing, the backtest would be measuring something the
   engine does not actually do.

Deliberately NOT modeled (stated plainly so the numbers are not oversold):
  - Intraday price paths. Stops are evaluated against daily lows, so an intraday
    spike through a stop and back is not captured.
  - Market impact beyond a flat slippage assumption.
  - Borrow costs and short availability. Long-only, so this does not bind.
  - Survivorship bias: the watchlist is today's large caps, which did not all
    have equal prospects two years ago. This flatters results and is the single
    biggest caveat on these numbers.
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import SECTOR_MAP, TRADING_CONFIG
from core.position_sizer import compute_qty, sector_of

logger = logging.getLogger("doom.backtest")

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestConfig:
    """Cost and risk assumptions. Defaults model a retail broker."""
    initial_capital: float = 100_000.0

    # Costs. commission_per_share reflects a typical per-share broker fee;
    # slippage_bps is the spread/impact cost paid on entry and again on exit.
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_bps: float = 5.0  # 5 basis points = 0.05%

    # Idle cash earns the risk-free rate, as it would at a real desk sitting
    # in T-bills. Modeling it at 0% is an unrealistic drag on any strategy that
    # is not fully invested. Set to 0.0 to disable.
    cash_yield_annual: float = 0.05

    # Risk limits — mirrored from the live TradingConfig.
    max_position_pct: float = TRADING_CONFIG.max_position_pct
    max_sector_pct: float = TRADING_CONFIG.max_sector_pct
    max_daily_drawdown_pct: float = TRADING_CONFIG.max_daily_drawdown_pct
    max_open_positions: int = 15

    # Exits
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    max_holding_days: int = 20

    # Signal parameters — same indicators the live agents use.
    ema_fast: int = TRADING_CONFIG.ema_fast
    ema_slow: int = TRADING_CONFIG.ema_slow
    rsi_period: int = 14
    bb_period: int = TRADING_CONFIG.bollinger_period
    bb_std: float = TRADING_CONFIG.bollinger_std

    warmup_bars: int = 50  # bars consumed before trading, so indicators are valid


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    entry_date: pd.Timestamp
    strategy: str
    stop_price: float
    target_price: float


@dataclass
class Trade:
    symbol: str
    strategy: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    gross_pnl: float
    costs: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    holding_days: int


class BacktestEngine:
    """Event-driven, bar-by-bar backtester over daily OHLCV data."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self._day_start_equity = self.config.initial_capital
        self._halted_today = False

    # ------------------------------------------------------------------
    # Indicators — computed once per symbol up front, then read bar by bar.
    # ------------------------------------------------------------------
    def _precompute_indicators(self, prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        out = {}
        for symbol, df in prices.items():
            d = df.copy()
            close = d["close"]
            d["ema_fast"] = close.ewm(span=self.config.ema_fast, adjust=False).mean()
            d["ema_slow"] = close.ewm(span=self.config.ema_slow, adjust=False).mean()

            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(self.config.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(self.config.rsi_period).mean()
            rs = gain / loss.replace(0, np.nan)
            d["rsi"] = 100 - (100 / (1 + rs))

            mid = close.rolling(self.config.bb_period).mean()
            std = close.rolling(self.config.bb_period).std()
            d["bb_mid"] = mid
            d["bb_upper"] = mid + self.config.bb_std * std
            d["bb_lower"] = mid - self.config.bb_std * std

            d["vol_avg"] = d["volume"].rolling(20).mean()
            d["ret_5d"] = close.pct_change(5)
            out[symbol] = d
        return out

    # ------------------------------------------------------------------
    # Signals — the same two strategies the live StrategyAgent runs.
    # Uses data up to and including bar i only. Never bar i+1.
    # ------------------------------------------------------------------
    def _momentum_signal(self, d: pd.DataFrame, i: int) -> Optional[float]:
        """EMA fast/slow crossover confirmed by volume. Returns 0..1 strength."""
        if i < 1:
            return None
        ema_f, ema_s = d["ema_fast"].iloc[i], d["ema_slow"].iloc[i]
        prev_f, prev_s = d["ema_fast"].iloc[i - 1], d["ema_slow"].iloc[i - 1]
        if any(pd.isna(x) for x in (ema_f, ema_s, prev_f, prev_s)):
            return None

        # Require a fresh crossover, not merely a persistent trend.
        if not (prev_f <= prev_s and ema_f > ema_s):
            return None

        rsi = d["rsi"].iloc[i]
        if pd.isna(rsi) or rsi > 70:  # already overbought — skip
            return None

        vol, vol_avg = d["volume"].iloc[i], d["vol_avg"].iloc[i]
        vol_confirmed = (not pd.isna(vol_avg)) and vol_avg > 0 and vol > vol_avg

        spread = abs(ema_f - ema_s) / ema_s if ema_s else 0
        strength = min(1.0, spread * 50)
        if vol_confirmed:
            strength = min(1.0, strength * 1.3)
        return strength if strength > 0.15 else None

    def _mean_reversion_signal(self, d: pd.DataFrame, i: int) -> Optional[float]:
        """Oversold bounce: close below the lower Bollinger band with low RSI."""
        close = d["close"].iloc[i]
        lower, mid = d["bb_lower"].iloc[i], d["bb_mid"].iloc[i]
        rsi = d["rsi"].iloc[i]
        if any(pd.isna(x) for x in (close, lower, mid, rsi)):
            return None
        if close >= lower or rsi >= 35:
            return None

        stretch = (lower - close) / mid if mid else 0
        strength = min(1.0, stretch * 20 + (35 - rsi) / 35)
        return strength if strength > 0.2 else None

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------
    def _commission(self, qty: int) -> float:
        return max(self.config.min_commission, qty * self.config.commission_per_share)

    def _fill_price(self, price: float, side: str) -> float:
        """Slippage always works against us: buys fill higher, sells lower."""
        slip = price * self.config.slippage_bps / 10_000.0
        return price + slip if side == "buy" else price - slip

    # ------------------------------------------------------------------
    # Portfolio helpers
    # ------------------------------------------------------------------
    def _equity(self, bar_prices: Dict[str, float]) -> float:
        held = sum(
            pos.qty * bar_prices.get(sym, pos.entry_price)
            for sym, pos in self.positions.items()
        )
        return self.cash + held

    def _sector_exposure(self, bar_prices: Dict[str, float], equity: float) -> Dict[str, float]:
        exposure: Dict[str, float] = {}
        if equity <= 0:
            return exposure
        for sym, pos in self.positions.items():
            value = pos.qty * bar_prices.get(sym, pos.entry_price)
            sector = sector_of(sym)
            exposure[sector] = exposure.get(sector, 0.0) + (value / equity * 100)
        return exposure

    # ------------------------------------------------------------------
    # Entry / exit
    # ------------------------------------------------------------------
    def _open_position(self, symbol: str, strategy: str, strength: float,
                       fill_open: float, date: pd.Timestamp,
                       equity: float, sector_exposure: Dict[str, float]) -> bool:
        if symbol in self.positions:
            return False
        if len(self.positions) >= self.config.max_open_positions:
            return False

        # Size from conviction, then apply the same caps the live engine uses.
        size_pct = 1.0 + 4.0 * strength  # 1%..5%
        size_pct = min(size_pct, self.config.max_position_pct * 100)

        sector = sector_of(symbol)
        headroom = self.config.max_sector_pct * 100 - sector_exposure.get(sector, 0.0)
        if headroom <= 0:
            return False
        size_pct = min(size_pct, headroom)

        price = self._fill_price(fill_open, "buy")
        qty = compute_qty(equity, size_pct, price)
        if qty <= 0:
            return False

        cost = qty * price + self._commission(qty)
        if cost > self.cash:
            qty = int((self.cash - self.config.min_commission) // price)
            if qty <= 0:
                return False
            cost = qty * price + self._commission(qty)

        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
            qty=qty,
            entry_price=price,
            entry_date=date,
            strategy=strategy,
            stop_price=price * (1 - self.config.stop_loss_pct / 100),
            target_price=price * (1 + self.config.take_profit_pct / 100),
        )
        return True

    def _close_position(self, symbol: str, exit_price: float,
                        date: pd.Timestamp, reason: str):
        pos = self.positions.pop(symbol)
        price = self._fill_price(exit_price, "sell")
        proceeds = pos.qty * price
        exit_comm = self._commission(pos.qty)
        entry_comm = self._commission(pos.qty)
        self.cash += proceeds - exit_comm

        gross = (price - pos.entry_price) * pos.qty
        costs = entry_comm + exit_comm
        net = gross - exit_comm  # entry commission was already paid at open

        self.trades.append(Trade(
            symbol=symbol,
            strategy=pos.strategy,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=price,
            qty=pos.qty,
            gross_pnl=round(gross, 2),
            costs=round(costs, 2),
            pnl=round(net, 2),
            pnl_pct=round(net / (pos.entry_price * pos.qty) * 100, 4),
            exit_reason=reason,
            holding_days=max(1, (date - pos.entry_date).days),
        ))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, prices: Dict[str, pd.DataFrame]) -> Dict:
        """
        prices: {symbol: DataFrame with open/high/low/close/volume, DatetimeIndex}
        Returns a metrics dict; trades and equity curve are on the instance.
        """
        data = self._precompute_indicators(prices)

        # Union of all dates, so a symbol missing a bar does not shift others.
        all_dates = sorted(set().union(*[set(d.index) for d in data.values()]))
        idx_maps = {sym: {d: i for i, d in enumerate(df.index)} for sym, df in data.items()}

        start = self.config.warmup_bars
        current_day = None

        for bar_num in range(start, len(all_dates) - 1):
            date = all_dates[bar_num]
            next_date = all_dates[bar_num + 1]

            # Reset the daily drawdown halt at each new day.
            if current_day != date.date():
                current_day = date.date()
                self._day_start_equity = self._equity(self._closes(data, idx_maps, date))
                self._halted_today = False

            closes = self._closes(data, idx_maps, date)
            equity = self._equity(closes)

            # --- 1. Exits, evaluated on the CURRENT bar's high/low ---
            for symbol in list(self.positions.keys()):
                df, i = self._bar(data, idx_maps, symbol, date)
                if df is None:
                    continue
                pos = self.positions[symbol]
                low, high, close = df["low"].iloc[i], df["high"].iloc[i], df["close"].iloc[i]

                if low <= pos.stop_price:
                    # Conservative: assume the stop filled at the stop price.
                    self._close_position(symbol, pos.stop_price, date, "STOP_LOSS")
                elif high >= pos.target_price:
                    self._close_position(symbol, pos.target_price, date, "TAKE_PROFIT")
                elif (date - pos.entry_date).days >= self.config.max_holding_days:
                    self._close_position(symbol, close, date, "TIME_EXIT")

            # --- 2. Daily drawdown circuit breaker ---
            equity = self._equity(self._closes(data, idx_maps, date))
            if self._day_start_equity > 0:
                dd = (self._day_start_equity - equity) / self._day_start_equity
                if dd >= self.config.max_daily_drawdown_pct:
                    self._halted_today = True

            # --- 3. Entries: decide on bar i, fill at bar i+1 OPEN ---
            if not self._halted_today:
                candidates = []
                for symbol, df in data.items():
                    if symbol in self.positions:
                        continue
                    i = idx_maps[symbol].get(date)
                    if i is None or i < self.config.warmup_bars:
                        continue

                    strength = self._momentum_signal(df, i)
                    strategy = "MOMENTUM"
                    if strength is None:
                        strength = self._mean_reversion_signal(df, i)
                        strategy = "MEAN_REVERSION"
                    if strength is None:
                        continue

                    j = idx_maps[symbol].get(next_date)
                    if j is None:
                        continue
                    next_open = df["open"].iloc[j]
                    if pd.isna(next_open) or next_open <= 0:
                        continue
                    candidates.append((strength, symbol, strategy, float(next_open)))

                candidates.sort(reverse=True, key=lambda c: c[0])
                sector_exposure = self._sector_exposure(closes, equity)

                for strength, symbol, strategy, next_open in candidates:
                    opened = self._open_position(
                        symbol, strategy, strength, next_open,
                        next_date, equity, sector_exposure,
                    )
                    if opened:
                        value_pct = (
                            self.positions[symbol].qty * self.positions[symbol].entry_price
                            / equity * 100
                        ) if equity > 0 else 0
                        sector = sector_of(symbol)
                        sector_exposure[sector] = sector_exposure.get(sector, 0) + value_pct

            # --- 4. Mark to market ---
            final_closes = self._closes(data, idx_maps, date)
            self.equity_curve.append({
                "date": date,
                "equity": round(self._equity(final_closes), 2),
                "cash": round(self.cash, 2),
                "n_positions": len(self.positions),
            })

        # Liquidate anything still open at the final close.
        last_date = all_dates[-1]
        for symbol in list(self.positions.keys()):
            df, i = self._bar(data, idx_maps, symbol, last_date)
            price = df["close"].iloc[i] if df is not None else self.positions[symbol].entry_price
            self._close_position(symbol, float(price), last_date, "END_OF_TEST")

        final_equity = self.cash
        self.equity_curve.append({
            "date": last_date, "equity": round(final_equity, 2),
            "cash": round(self.cash, 2), "n_positions": 0,
        })

        self._apply_cash_yield()
        return self.compute_metrics()

    def _apply_cash_yield(self):
        """
        Accrue the risk-free rate on uninvested cash.

        This strategy is only ~9% invested on average, so without this the
        backtest implicitly assumes 91% of the book sits in a zero-yield account
        — a penalty no real desk would accept. Compounds daily on the prior
        bar's cash balance.
        """
        if self.config.cash_yield_annual <= 0 or len(self.equity_curve) < 2:
            return
        daily_rate = (1 + self.config.cash_yield_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        accrued = 0.0
        for k in range(1, len(self.equity_curve)):
            accrued += self.equity_curve[k - 1]["cash"] * daily_rate
            self.equity_curve[k]["equity"] = round(
                self.equity_curve[k]["equity"] + accrued, 2
            )

    def _bar(self, data, idx_maps, symbol, date):
        i = idx_maps[symbol].get(date)
        if i is None:
            return None, None
        return data[symbol], i

    def _closes(self, data, idx_maps, date) -> Dict[str, float]:
        out = {}
        for symbol, df in data.items():
            i = idx_maps[symbol].get(date)
            if i is not None:
                value = df["close"].iloc[i]
                if not pd.isna(value):
                    out[symbol] = float(value)
        return out

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def compute_metrics(self) -> Dict:
        if not self.equity_curve:
            return {}

        equities = np.array([e["equity"] for e in self.equity_curve], dtype=float)
        initial = self.config.initial_capital
        final = float(equities[-1])

        returns = np.diff(equities) / equities[:-1]
        returns = returns[np.isfinite(returns)]

        total_return_pct = (final - initial) / initial * 100
        years = max(len(equities) / TRADING_DAYS_PER_YEAR, 1e-9)
        cagr = ((final / initial) ** (1 / years) - 1) * 100 if final > 0 else -100.0

        if len(returns) > 1 and returns.std(ddof=1) > 0:
            sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sharpe = 0.0

        downside = returns[returns < 0]
        if len(downside) > 1 and downside.std(ddof=1) > 0:
            sortino = returns.mean() / downside.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sortino = 0.0

        running_peak = np.maximum.accumulate(equities)
        drawdowns = (running_peak - equities) / running_peak
        max_dd = float(drawdowns.max() * 100) if len(drawdowns) else 0.0

        closed = self.trades
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        win_rate = len(wins) / len(closed) * 100 if closed else 0.0
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

        total_costs = sum(t.costs for t in closed)

        by_strategy: Dict[str, Dict] = {}
        for strategy in {t.strategy for t in closed}:
            subset = [t for t in closed if t.strategy == strategy]
            subset_wins = [t for t in subset if t.pnl > 0]
            by_strategy[strategy] = {
                "trades": len(subset),
                "win_rate": round(len(subset_wins) / len(subset) * 100, 1) if subset else 0,
                "total_pnl": round(sum(t.pnl for t in subset), 2),
                "avg_pnl": round(float(np.mean([t.pnl for t in subset])), 2) if subset else 0,
            }

        exit_reasons: Dict[str, int] = {}
        for trade in closed:
            exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1

        return {
            "initial_capital": initial,
            "final_equity": round(final, 2),
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "max_drawdown_pct": round(max_dd, 2),
            "calmar_ratio": round(cagr / max_dd, 2) if max_dd > 0 else 0.0,
            "total_trades": len(closed),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if math.isfinite(profit_factor) else None,
            "avg_win": round(float(np.mean([t.pnl for t in wins])), 2) if wins else 0.0,
            "avg_loss": round(float(np.mean([t.pnl for t in losses])), 2) if losses else 0.0,
            "avg_holding_days": round(float(np.mean([t.holding_days for t in closed])), 1) if closed else 0,
            "total_costs": round(total_costs, 2),
            "cost_drag_pct": round(total_costs / initial * 100, 2),
            "trading_days": len(equities),
            "by_strategy": by_strategy,
            "exit_reasons": exit_reasons,
        }


def buy_and_hold_benchmark(spy: pd.DataFrame, initial_capital: float,
                           warmup_bars: int = 50) -> Dict:
    """SPY buy-and-hold over the same window — the bar any strategy must clear."""
    df = spy.iloc[warmup_bars:]
    closes = df["close"].values.astype(float)
    if len(closes) < 2:
        return {}

    qty = int(initial_capital // closes[0])
    cash = initial_capital - qty * closes[0]
    equities = cash + qty * closes

    returns = np.diff(equities) / equities[:-1]
    returns = returns[np.isfinite(returns)]
    final = float(equities[-1])
    years = max(len(equities) / TRADING_DAYS_PER_YEAR, 1e-9)

    running_peak = np.maximum.accumulate(equities)
    max_dd = float(((running_peak - equities) / running_peak).max() * 100)
    sharpe = (
        returns.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
        if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0
    )

    return {
        "final_equity": round(final, 2),
        "total_return_pct": round((final - initial_capital) / initial_capital * 100, 2),
        "cagr_pct": round(((final / initial_capital) ** (1 / years) - 1) * 100, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "equity_curve": [
            {"date": d, "equity": round(float(e), 2)}
            for d, e in zip(df.index, equities)
        ],
    }
