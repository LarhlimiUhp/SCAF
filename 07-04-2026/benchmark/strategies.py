import numpy as np
import pandas as pd


class BenchmarkStrategies:
    @staticmethod
    def buy_and_hold(prices: pd.Series):
        returns = np.log(prices / prices.shift(1)).fillna(0)
        return returns

    @staticmethod
    def momentum(prices: pd.Series, window: int = 20):
        signal = prices.pct_change(window)
        position = np.where(signal > 0, 1.0, 0.0)
        returns = position * np.log(prices / prices.shift(1)).fillna(0)
        return pd.Series(returns, index=prices.index)

    @staticmethod
    def sixty_forty(equity_prices: pd.Series, bond_prices: pd.Series,
                    rebalance_freq: int = 21) -> pd.Series:
        """
        Classic 60 % equity / 40 % bond portfolio rebalanced monthly.

        Parameters
        ----------
        equity_prices : daily close prices for the equity leg (e.g. S&P 500)
        bond_prices   : daily close prices for the bond leg (e.g. AGG / TLT)
        rebalance_freq: number of trading days between rebalancing (default 21 ≈ 1 month)

        Returns
        -------
        pd.Series of daily portfolio log-returns
        """
        eq_ret = np.log(equity_prices / equity_prices.shift(1)).fillna(0)
        bd_ret = np.log(bond_prices / bond_prices.shift(1)).fillna(0)

        # Align on common index
        eq_ret, bd_ret = eq_ret.align(bd_ret, join='inner', fill_value=0)

        port_returns = pd.Series(index=eq_ret.index, dtype=float)
        w_eq, w_bd = 0.60, 0.40

        for i, dt in enumerate(eq_ret.index):
            if i > 0 and i % rebalance_freq == 0:
                # Reset weights (drift since last rebalance is small for daily returns)
                w_eq, w_bd = 0.60, 0.40
            port_returns[dt] = w_eq * eq_ret[dt] + w_bd * bd_ret[dt]

        return port_returns

    @staticmethod
    def risk_parity(asset_prices: pd.DataFrame, lookback: int = 60,
                    rebalance_freq: int = 21) -> pd.Series:
        """
        Equal-risk-contribution (risk parity) portfolio.

        Each asset receives a weight inversely proportional to its realised
        volatility over the trailing ``lookback`` window.

        Parameters
        ----------
        asset_prices  : DataFrame of daily close prices, one column per asset
        lookback      : rolling window (days) used to estimate volatility
        rebalance_freq: number of trading days between rebalancing

        Returns
        -------
        pd.Series of daily portfolio log-returns
        """
        log_returns = np.log(asset_prices / asset_prices.shift(1)).fillna(0)
        port_returns = pd.Series(index=log_returns.index, dtype=float)
        weights = None

        for i, dt in enumerate(log_returns.index):
            if i < lookback:
                port_returns[dt] = 0.0
                continue

            if i == lookback or (i > lookback and (i - lookback) % rebalance_freq == 0):
                window = log_returns.iloc[i - lookback: i]
                vols = window.std(ddof=1).replace(0, np.nan).fillna(1e-8)
                inv_vol = 1.0 / vols
                weights = inv_vol / inv_vol.sum()  # normalise to 1

            if weights is not None:
                port_returns[dt] = (log_returns.loc[dt] * weights).sum()
            else:
                port_returns[dt] = 0.0

        return port_returns
