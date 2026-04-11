"""
15 benchmark trading strategies for SCAF baseline comparison.

Each strategy takes a price Series and returns a log-return Series
(same index, NaN where no position can be computed).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd


class BenchmarkStrategies:
    # ── 1. Buy-and-hold ─────────────────────────────────────────────────── #
    @staticmethod
    def buy_and_hold(prices: pd.Series) -> pd.Series:
        """Always long at full position."""
        returns = np.log(prices / prices.shift(1)).fillna(0)
        return returns

    # ── 2. Momentum ─────────────────────────────────────────────────────── #
    @staticmethod
    def momentum(prices: pd.Series, window: int = 20) -> pd.Series:
        """Go long when *window*-day return > 0, else flat."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        signal = prices.pct_change(window)
        position = np.where(signal > 0, 1.0, 0.0)
        return pd.Series(position * log_ret, index=prices.index)

    # ── 3. Mean-reversion ───────────────────────────────────────────────── #
    @staticmethod
    def mean_reversion(prices: pd.Series, window: int = 20) -> pd.Series:
        """Short when price > SMA, long when price < SMA."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        sma = prices.rolling(window).mean()
        position = np.where(prices < sma, 1.0, -1.0)
        position = pd.Series(position, index=prices.index).fillna(0)
        return position * log_ret

    # ── 4. SMA crossover (50 / 200) ─────────────────────────────────────── #
    @staticmethod
    def sma_crossover(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
        """Golden cross: long when SMA_fast > SMA_slow."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        fast_ma = prices.rolling(fast).mean()
        slow_ma = prices.rolling(slow).mean()
        position = np.where(fast_ma > slow_ma, 1.0, -1.0)
        position = pd.Series(position, index=prices.index).fillna(0)
        return position * log_ret

    # ── 5. EMA crossover (12 / 26) ──────────────────────────────────────── #
    @staticmethod
    def ema_crossover(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Long when EMA_fast > EMA_slow."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        position = np.where(ema_fast > ema_slow, 1.0, -1.0)
        position = pd.Series(position, index=prices.index).fillna(0)
        return position * log_ret

    # ── 6. RSI (overbought / oversold) ──────────────────────────────────── #
    @staticmethod
    def rsi_strategy(prices: pd.Series, period: int = 14,
                     oversold: float = 30.0, overbought: float = 70.0) -> pd.Series:
        """Long below *oversold*, short above *overbought*, flat in between."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-12)
        rsi = 100 - 100 / (1 + rs)
        position = np.where(rsi < oversold, 1.0, np.where(rsi > overbought, -1.0, 0.0))
        position = pd.Series(position, index=prices.index).fillna(0)
        return position * log_ret

    # ── 7. Bollinger Band ───────────────────────────────────────────────── #
    @staticmethod
    def bollinger_band(prices: pd.Series, window: int = 20,
                       n_std: float = 2.0) -> pd.Series:
        """Long below lower band, short above upper band."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + n_std * std
        lower = sma - n_std * std
        position = np.where(prices < lower, 1.0, np.where(prices > upper, -1.0, 0.0))
        position = pd.Series(position, index=prices.index).fillna(0)
        return position * log_ret

    # ── 8. Random strategy (statistical floor) ──────────────────────────── #
    @staticmethod
    def random_strategy(prices: pd.Series, seed: int = 42) -> pd.Series:
        """Random ±1 position — serves as a floor baseline."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        rng = np.random.default_rng(seed)
        position = rng.choice([-1.0, 1.0], size=len(prices))
        return pd.Series(position * log_ret.values, index=prices.index)

    # ── 9. Volatility-scaling ───────────────────────────────────────────── #
    @staticmethod
    def volatility_scaling(prices: pd.Series, target_vol: float = 0.01,
                           window: int = 20) -> pd.Series:
        """Long with position size = target_vol / realised_vol."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        realised_vol = log_ret.rolling(window).std().shift(1).fillna(1e-4)
        position = (target_vol / realised_vol).clip(0, 2.0)
        return position * log_ret

    # ── 10. Trend-following (52-week high breakout) ──────────────────────── #
    @staticmethod
    def trend_following(prices: pd.Series, window: int = 252) -> pd.Series:
        """Long when price makes a new *window*-day high."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        rolling_high = prices.rolling(window).max().shift(1)
        position = np.where(prices >= rolling_high, 1.0, 0.0)
        position = pd.Series(position, index=prices.index).fillna(0)
        return position * log_ret

    # ── 11. MACD strategy ───────────────────────────────────────────────── #
    @staticmethod
    def macd_strategy(prices: pd.Series, fast: int = 12,
                      slow: int = 26, signal: int = 9) -> pd.Series:
        """Long when MACD line crosses above signal line."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        position = np.where(macd_line > signal_line, 1.0, -1.0)
        position = pd.Series(position, index=prices.index).fillna(0)
        return position * log_ret

    # ── 12. Donchian breakout ───────────────────────────────────────────── #
    @staticmethod
    def donchian_breakout(prices: pd.Series, window: int = 20) -> pd.Series:
        """Long at upper channel breakout, short at lower channel breakout."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        upper = prices.rolling(window).max().shift(1)
        lower = prices.rolling(window).min().shift(1)
        position = np.where(prices >= upper, 1.0,
                            np.where(prices <= lower, -1.0, 0.0))
        position = pd.Series(position, index=prices.index).fillna(0)
        return position * log_ret

    # ── 13. Risk-adjusted momentum ──────────────────────────────────────── #
    @staticmethod
    def risk_adjusted_momentum(prices: pd.Series, mom_window: int = 20,
                               vol_window: int = 20,
                               target_vol: float = 0.01) -> pd.Series:
        """Momentum signal scaled by inverse realised volatility."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        mom_signal = prices.pct_change(mom_window)
        realised_vol = log_ret.rolling(vol_window).std().shift(1).fillna(1e-4)
        position = np.sign(mom_signal) * (target_vol / realised_vol).clip(0, 2.0)
        position = position.fillna(0)
        return position * log_ret

    # ── 14. Contrarian ──────────────────────────────────────────────────── #
    @staticmethod
    def contrarian(prices: pd.Series, window: int = 5) -> pd.Series:
        """Short-term reversal: fade the previous *window*-day move."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        short_ret = prices.pct_change(window)
        position = np.where(short_ret > 0, -1.0, 1.0)
        position = pd.Series(position, index=prices.index).fillna(0)
        return position * log_ret

    # ── 15. Adaptive momentum ───────────────────────────────────────────── #
    @staticmethod
    def adaptive_momentum(prices: pd.Series, short: int = 10,
                          long: int = 60) -> pd.Series:
        """Long when short-term momentum exceeds long-term momentum."""
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        short_mom = prices.pct_change(short)
        long_mom = prices.pct_change(long)
        position = np.where(short_mom > long_mom, 1.0, -1.0)
        position = pd.Series(position, index=prices.index).fillna(0)
        return position * log_ret

    # ── Registry ────────────────────────────────────────────────────────── #
    @classmethod
    def all_strategies(cls) -> Dict[str, Callable[[pd.Series], pd.Series]]:
        """Return a dict mapping strategy name → callable(prices) → returns."""
        return {
            "buy_and_hold": cls.buy_and_hold,
            "momentum": cls.momentum,
            "mean_reversion": cls.mean_reversion,
            "sma_crossover": cls.sma_crossover,
            "ema_crossover": cls.ema_crossover,
            "rsi_strategy": cls.rsi_strategy,
            "bollinger_band": cls.bollinger_band,
            "random_strategy": cls.random_strategy,
            "volatility_scaling": cls.volatility_scaling,
            "trend_following": cls.trend_following,
            "macd_strategy": cls.macd_strategy,
            "donchian_breakout": cls.donchian_breakout,
            "risk_adjusted_momentum": cls.risk_adjusted_momentum,
            "contrarian": cls.contrarian,
            "adaptive_momentum": cls.adaptive_momentum,
        }
