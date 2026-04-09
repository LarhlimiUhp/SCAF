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
