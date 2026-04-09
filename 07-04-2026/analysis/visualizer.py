import matplotlib.pyplot as plt
import pandas as pd


class BacktestVisualizer:
    @staticmethod
    def plot_equity_curve(equity: pd.Series, title: str = 'Equity Curve'):
        fig, ax = plt.subplots(figsize=(10, 5))
        equity.plot(ax=ax)
        ax.set_title(title)
        ax.set_ylabel('Equity')
        ax.set_xlabel('Date')
        return fig

    @staticmethod
    def plot_drawdown(equity: pd.Series, title: str = 'Drawdown'):
        peak = equity.cummax()
        dd = equity / peak - 1
        fig, ax = plt.subplots(figsize=(10, 4))
        dd.plot(ax=ax, color='tab:red')
        ax.set_title(title)
        ax.set_ylabel('Drawdown')
        ax.set_xlabel('Date')
        return fig

    @staticmethod
    def plot_positions_by_regime(positions: pd.DataFrame, title: str = 'Positions by Regime'):
        fig, ax = plt.subplots(figsize=(10, 4))
        positions.plot(ax=ax)
        ax.set_title(title)
        ax.set_ylabel('Position Size')
        ax.set_xlabel('Date')
        return fig
