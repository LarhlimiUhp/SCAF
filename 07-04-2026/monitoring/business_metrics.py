"""
Métriques métier temps réel : P&L, Drawdown, Sharpe, Win Rate
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque


@dataclass
class TradeMetrics:
    """Métriques de trading"""
    timestamp: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # %
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_return: float  # %
    
    
@dataclass
class PortfolioMetrics:
    """Métriques du portefeuille"""
    timestamp: str
    current_equity: float
    initial_equity: float
    total_return: float  # %
    daily_return: float  # %
    cumulative_return: float  # %
    max_drawdown: float  # %
    current_drawdown: float  # %
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    recovery_factor: float
    

class BusinessMetricsCollector:
    """Collecteur de métriques métier temps réel"""
    
    def __init__(self, initial_equity: float = 100000.0, max_history: int = 10000):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.max_history = max_history
        self.equity_history: deque = deque(maxlen=max_history)
        self.daily_returns: deque = deque(maxlen=max_history)
        self.trades: List[Dict] = []
        self.start_time = datetime.utcnow()
        self.last_close = initial_equity
        self.high_water_mark = initial_equity
        
    def update_equity(self, new_equity: float, timestamp: Optional[datetime] = None):
        """Mettre à jour le montant du portefeuille"""
        timestamp = timestamp or datetime.utcnow()
        
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': new_equity,
        })
        
        # Calculer le rendement journalier
        daily_return = (new_equity - self.last_close) / self.last_close if self.last_close > 0 else 0
        self.daily_returns.append(daily_return)
        
        # Mettre à jour high water mark pour les drawdowns
        if new_equity > self.high_water_mark:
            self.high_water_mark = new_equity
        
        self.current_equity = new_equity
        self.last_close = new_equity
    
    def record_trade(self, entry_price: float, exit_price: float, quantity: int,
                     side: str = "long", commission: float = 0.0):
        """Enregistrer un trade"""
        pnl = (exit_price - entry_price) * quantity if side == "long" else (entry_price - exit_price) * quantity
        pnl_after_commission = pnl - commission
        
        self.trades.append({
            'timestamp': datetime.utcnow().isoformat(),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'side': side,
            'pnl': pnl,
            'pnl_after_commission': pnl_after_commission,
            'commission': commission,
            'winning': pnl_after_commission > 0,
        })
    
    def get_trade_metrics(self) -> TradeMetrics:
        """Obtenir les métriques de trading"""
        if not self.trades:
            return TradeMetrics(
                timestamp=datetime.utcnow().isoformat(),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                total_return=0.0,
            )
        
        winning_trades = sum(1 for t in self.trades if t['winning'])
        losing_trades = len(self.trades) - winning_trades
        
        wins = [t['pnl_after_commission'] for t in self.trades if t['winning']]
        losses = [t['pnl_after_commission'] for t in self.trades if not t['winning']]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        total_pnl = sum(t['pnl_after_commission'] for t in self.trades)
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0.0
        
        return TradeMetrics(
            timestamp=datetime.utcnow().isoformat(),
            total_trades=len(self.trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=(winning_trades / len(self.trades) * 100) if self.trades else 0.0,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_return=((self.current_equity - self.initial_equity) / self.initial_equity * 100),
        )
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Obtenir les métriques du portefeuille"""
        if not self.equity_history:
            return PortfolioMetrics(
                timestamp=datetime.utcnow().isoformat(),
                current_equity=self.current_equity,
                initial_equity=self.initial_equity,
                total_return=0.0,
                daily_return=0.0,
                cumulative_return=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                recovery_factor=0.0,
            )
        
        # Rendements
        total_return = ((self.current_equity - self.initial_equity) / self.initial_equity * 100)
        daily_return = (self.daily_returns[-1] * 100) if self.daily_returns else 0.0
        
        # Drawdown
        max_drawdown = self._calculate_max_drawdown()
        current_drawdown = ((self.current_equity - self.high_water_mark) / self.high_water_mark * 100)
        
        # Sharpe ratio (annualisé, 252 trading days par an)
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        
        # Calmar ratio = Annual return / Max drawdown
        annual_return = total_return  # Approximé
        calmar_ratio = (annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
        
        # Recovery factor = Total return / Max drawdown
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown < 0 else 0.0
        
        return PortfolioMetrics(
            timestamp=datetime.utcnow().isoformat(),
            current_equity=self.current_equity,
            initial_equity=self.initial_equity,
            total_return=total_return,
            daily_return=daily_return,
            cumulative_return=total_return,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            recovery_factor=recovery_factor,
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculer le drawdown maximum"""
        if not self.equity_history:
            return 0.0
        
        equities = np.array([e['equity'] for e in self.equity_history])
        cummax = np.maximum.accumulate(equities)
        drawdown = (equities - cummax) / cummax
        
        return float(np.min(drawdown) * 100) if len(drawdown) > 0 else 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculer Sharpe ratio annualisé (sans taux sans risque)"""
        if not self.daily_returns or len(self.daily_returns) < 2:
            return 0.0
        
        returns = np.array(list(self.daily_returns))
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualisé: sqrt(252 trading days)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return float(sharpe)
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculer Sortino ratio annualisé"""
        if not self.daily_returns or len(self.daily_returns) < 2:
            return 0.0
        
        returns = np.array(list(self.daily_returns))
        mean_return = np.mean(returns)
        
        # Volatilité des rendements négatifs uniquement
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return / downside_std) * np.sqrt(252)
        return float(sortino)
    
    def get_health_score(self) -> Dict[str, Any]:
        """Score de santé du trading (0-100)"""
        portfolio_metrics = self.get_portfolio_metrics()
        trade_metrics = self.get_trade_metrics()
        
        scores = []
        
        # Score basé sur win rate
        if trade_metrics.total_trades > 0:
            win_rate_score = trade_metrics.win_rate
            scores.append(min(100, win_rate_score * 2))  # Ratio 2:1
        
        # Score basé sur Sharpe ratio
        sharpe_score = min(100, max(0, portfolio_metrics.sharpe_ratio * 10 + 50))
        scores.append(sharpe_score)
        
        # Score basé sur drawdown (moins c'est négatif, mieux c'est)
        drawdown_score = min(100, 100 + portfolio_metrics.max_drawdown * 5)
        scores.append(max(0, drawdown_score))
        
        # Score basé sur le profit factor
        if trade_metrics.profit_factor > 0:
            pf_score = min(100, trade_metrics.profit_factor * 25)
            scores.append(pf_score)
        
        return {
            'overall_health': float(np.mean(scores)) if scores else 50.0,
            'win_rate': float(trade_metrics.win_rate),
            'sharpe_ratio': float(portfolio_metrics.sharpe_ratio),
            'max_drawdown': float(portfolio_metrics.max_drawdown),
            'total_return': float(portfolio_metrics.total_return),
            'profit_factor': float(trade_metrics.profit_factor),
        }
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Obtenir un résumé complet"""
        trade_metrics = self.get_trade_metrics()
        portfolio_metrics = self.get_portfolio_metrics()
        health = self.get_health_score()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'trades': asdict(trade_metrics),
            'portfolio': asdict(portfolio_metrics),
            'health': health,
        }
