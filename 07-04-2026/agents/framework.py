"""
SCAF-LS Agent Framework for Risk Management
Implements 300 sub-agents for drawdown control and volatility management
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all risk management agents"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.is_active = True
        self.performance_history = deque(maxlen=1000)
        self.signal_history = deque(maxlen=500)

    @abstractmethod
    def evaluate_conditions(self, market_data: Dict[str, Any]) -> float:
        """Evaluate market conditions and return agent signal (-1 to 1)"""
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """Return agent's confidence in its signal (0 to 1)"""
        pass

    def update_performance(self, pnl: float, market_conditions: Dict[str, Any]):
        """Update agent performance tracking"""
        self.performance_history.append({
            'pnl': pnl,
            'conditions': market_conditions,
            'timestamp': pd.Timestamp.now()
        })

    def get_recent_performance(self, window: int = 50) -> Dict[str, float]:
        """Get recent performance metrics"""
        if len(self.performance_history) < window:
            return {'sharpe': 0.0, 'win_rate': 0.0, 'avg_return': 0.0}

        recent = list(self.performance_history)[-window:]
        pnls = [p['pnl'] for p in recent]

        returns = np.array(pnls)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        win_rate = np.mean(returns > 0)

        return {
            'sharpe': sharpe,
            'win_rate': win_rate,
            'avg_return': np.mean(returns),
            'volatility': np.std(returns)
        }


class VolatilityControlAgent(BaseAgent):
    """Agent specialized in volatility targeting and control"""

    def __init__(self, agent_id: str, config: Dict[str, Any], target_vol: float = 0.12):
        super().__init__(agent_id, config)
        self.target_vol = target_vol
        self.vol_window = config.get('vol_window', 20)
        self.vol_history = deque(maxlen=self.vol_window * 2)
        self.circuit_breaker_threshold = config.get('circuit_breaker', 0.25)  # 25% vol spike
        self.leverage_scaler = config.get('leverage_scaler', 0.8)

    def evaluate_conditions(self, market_data: Dict[str, Any]) -> float:
        """Evaluate volatility conditions and return position adjustment signal"""
        returns = market_data.get('returns', [])
        current_vol = market_data.get('current_vol', 0.02)

        if len(returns) < self.vol_window:
            return 0.0

        # Calculate realized volatility
        realized_vol = np.std(returns[-self.vol_window:]) * np.sqrt(252)

        # Store for trend analysis
        self.vol_history.append(realized_vol)

        # Volatility ratio (current vs target)
        vol_ratio = self.target_vol / (realized_vol + 1e-6)

        # Circuit breaker for extreme volatility
        if realized_vol > self.circuit_breaker_threshold:
            return -1.0  # Force reduction to minimum position

        # Dynamic leverage based on volatility
        leverage_signal = np.clip(vol_ratio * self.leverage_scaler, 0.1, 2.0)

        # Trend analysis - reduce exposure if volatility is increasing
        if len(self.vol_history) >= 10:
            vol_trend = np.polyfit(range(len(self.vol_history)), list(self.vol_history), 1)[0]
            if vol_trend > 0.001:  # Volatility increasing
                leverage_signal *= 0.8

        return np.clip(leverage_signal - 1.0, -0.9, 0.9)  # Convert to adjustment signal

    def get_confidence(self) -> float:
        """Return confidence based on volatility stability"""
        if len(self.vol_history) < 10:
            return 0.5

        vol_std = np.std(list(self.vol_history))
        vol_mean = np.mean(list(self.vol_history))

        # Confidence increases with volatility stability
        cv = vol_std / (vol_mean + 1e-6)  # Coefficient of variation
        confidence = 1.0 / (1.0 + cv)

        return np.clip(confidence, 0.1, 0.95)


class DrawdownManagementAgent(BaseAgent):
    """Agent specialized in drawdown control and recovery"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.max_drawdown_limit = config.get('max_drawdown', 0.15)
        self.trailing_stop_pct = config.get('trailing_stop', 0.05)
        self.recovery_mode_threshold = config.get('recovery_threshold', 0.10)
        self.position_reduction_schedule = config.get('reduction_schedule', [0.8, 0.6, 0.4, 0.2])

        self.equity_history = deque(maxlen=500)
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        self.trailing_stop_level = 0.0
        self.in_recovery_mode = False

    def evaluate_conditions(self, market_data: Dict[str, Any]) -> float:
        """Evaluate drawdown conditions and return risk adjustment signal"""
        current_equity = market_data.get('current_equity', 1.0)
        self.equity_history.append(current_equity)

        # Update peak equity and drawdown
        self.peak_equity = max(self.peak_equity, current_equity)
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity

        # Update trailing stop
        if current_equity > self.trailing_stop_level:
            self.trailing_stop_level = current_equity * (1.0 - self.trailing_stop_pct)

        # Check trailing stop violation
        if current_equity <= self.trailing_stop_level and self.trailing_stop_level > 0:
            return -1.0  # Force position close

        # Recovery mode logic
        if self.current_drawdown > self.recovery_mode_threshold:
            self.in_recovery_mode = True
            # Graduated position reduction based on drawdown severity
            reduction_idx = min(int(self.current_drawdown * 10), len(self.position_reduction_schedule) - 1)
            reduction_factor = self.position_reduction_schedule[reduction_idx]
            return reduction_factor - 1.0  # Convert to adjustment signal
        else:
            self.in_recovery_mode = False
            return 0.0  # No adjustment needed

    def get_confidence(self) -> float:
        """Return confidence based on drawdown stability"""
        if len(self.equity_history) < 20:
            return 0.5

        recent_equity = list(self.equity_history)[-20:]
        drawdowns = []

        peak = recent_equity[0]
        for eq in recent_equity:
            peak = max(peak, eq)
            drawdowns.append((peak - eq) / peak)

        avg_drawdown = np.mean(drawdowns)
        drawdown_volatility = np.std(drawdowns)

        # Confidence decreases with drawdown volatility
        confidence = 1.0 / (1.0 + drawdown_volatility * 10)

        return np.clip(confidence, 0.1, 0.9)


class CrisisDetectionAgent(BaseAgent):
    """Agent specialized in crisis detection and regime-based risk management"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.regime_window = config.get('regime_window', 50)
        self.crisis_thresholds = config.get('crisis_thresholds', {
            'vol_spike': 0.20,  # 20% vol increase
            'return_drop': 0.05,  # 5% single day drop
            'correlation_break': 0.8,  # Correlation breakdown
            'vix_spike': 2.0  # 2x VIX increase
        })

        self.regime_history = deque(maxlen=self.regime_window)
        self.crisis_signals = deque(maxlen=20)
        self.stress_indicators = {
            'volatility': deque(maxlen=self.regime_window),
            'returns': deque(maxlen=self.regime_window),
            'vix': deque(maxlen=self.regime_window),
            'correlations': deque(maxlen=self.regime_window)
        }

    def evaluate_conditions(self, market_data: Dict[str, Any]) -> float:
        """Evaluate crisis conditions and return risk adjustment signal"""
        returns = market_data.get('returns', [])
        volatility = market_data.get('volatility', 0.02)
        vix = market_data.get('vix', 20.0)
        correlations = market_data.get('correlations', {})

        if len(returns) < 10:
            return 0.0

        # Update stress indicators
        self.stress_indicators['volatility'].append(volatility)
        self.stress_indicators['returns'].append(returns[-1] if returns else 0.0)
        self.stress_indicators['vix'].append(vix)
        self.stress_indicators['correlations'].append(correlations.get('avg_corr', 0.0))

        # Crisis detection logic
        crisis_score = 0.0

        # 1. Volatility spike detection
        if len(self.stress_indicators['volatility']) >= 10:
            recent_vol = np.mean(list(self.stress_indicators['volatility'])[-5:])
            baseline_vol = np.mean(list(self.stress_indicators['volatility'])[-20:-5])
            if baseline_vol > 0 and recent_vol / baseline_vol > self.crisis_thresholds['vol_spike']:
                crisis_score += 0.3

        # 2. Sharp return drops
        if returns and abs(returns[-1]) > self.crisis_thresholds['return_drop']:
            crisis_score += 0.2

        # 3. VIX spike
        if len(self.stress_indicators['vix']) >= 5:
            recent_vix = np.mean(list(self.stress_indicators['vix'])[-3:])
            baseline_vix = np.mean(list(self.stress_indicators['vix'])[-10:-3])
            if baseline_vix > 0 and recent_vix / baseline_vix > self.crisis_thresholds['vix_spike']:
                crisis_score += 0.25

        # 4. Correlation breakdown (flight to quality)
        if correlations and correlations.get('avg_corr', 1.0) < self.crisis_thresholds['correlation_break']:
            crisis_score += 0.25

        # Store crisis signal
        self.crisis_signals.append(crisis_score)

        # Determine regime and risk adjustment
        if crisis_score > 0.5:
            return -1.0  # Maximum risk reduction
        elif crisis_score > 0.3:
            return -0.7  # High risk reduction
        elif crisis_score > 0.15:
            return -0.4  # Moderate risk reduction
        else:
            return 0.0  # Normal conditions

    def get_confidence(self) -> float:
        """Return confidence based on crisis signal stability"""
        if len(self.crisis_signals) < 5:
            return 0.5

        recent_signals = list(self.crisis_signals)[-10:]
        signal_volatility = np.std(recent_signals)

        # Confidence increases with signal stability
        confidence = 1.0 / (1.0 + signal_volatility * 5)

        return np.clip(confidence, 0.2, 0.95)


class AgentOrchestrator:
    """Orchestrates 300 sub-agents for comprehensive risk management"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.agents = []
        self.agent_weights = {}
        self.performance_tracker = {}
        self.risk_thresholds = {
            'max_drawdown': 0.15,
            'volatility_target': 0.12,
            'crisis_signal_threshold': 0.3
        }

        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all 300 sub-agents"""

        # Volatility Control Agents (100)
        vol_configs = [
            {'vol_window': 20, 'circuit_breaker': 0.25, 'leverage_scaler': 0.8},
            {'vol_window': 30, 'circuit_breaker': 0.30, 'leverage_scaler': 0.7},
            {'vol_window': 40, 'circuit_breaker': 0.20, 'leverage_scaler': 0.9},
            # ... more variations
        ]

        for i in range(100):
            config = vol_configs[i % len(vol_configs)].copy()
            config.update({
                'vol_window': config['vol_window'] + (i % 10),
                'circuit_breaker': config['circuit_breaker'] + (i % 5) * 0.01
            })
            agent = VolatilityControlAgent(f'vol_ctrl_{i}', config)
            self.agents.append(agent)
            self.agent_weights[agent.agent_id] = 1.0

        # Drawdown Management Agents (100) - Higher weight for risk control
        dd_configs = [
            {'max_drawdown': 0.12, 'trailing_stop': 0.03, 'recovery_threshold': 0.08},  # Tighter limits
            {'max_drawdown': 0.10, 'trailing_stop': 0.025, 'recovery_threshold': 0.06},  # Even tighter
            {'max_drawdown': 0.15, 'trailing_stop': 0.035, 'recovery_threshold': 0.10},  # Slightly looser
            # ... more variations
        ]

        for i in range(100):
            config = dd_configs[i % len(dd_configs)].copy()
            config.update({
                'max_drawdown': config['max_drawdown'] + (i % 3) * 0.005,
                'trailing_stop': config['trailing_stop'] + (i % 2) * 0.002
            })
            agent = DrawdownManagementAgent(f'dd_mgmt_{i}', config)
            self.agents.append(agent)
            self.agent_weights[agent.agent_id] = 1.5  # Higher weight for drawdown control

        # Crisis Detection Agents (100)
        crisis_configs = [
            {'regime_window': 50, 'crisis_thresholds': {'vol_spike': 0.20, 'return_drop': 0.05, 'correlation_break': 0.8, 'vix_spike': 2.0}},
            {'regime_window': 40, 'crisis_thresholds': {'vol_spike': 0.25, 'return_drop': 0.04, 'correlation_break': 0.75, 'vix_spike': 2.2}},
            {'regime_window': 60, 'crisis_thresholds': {'vol_spike': 0.18, 'return_drop': 0.06, 'correlation_break': 0.85, 'vix_spike': 1.8}},
            # ... more variations
        ]

        for i in range(100):
            config = crisis_configs[i % len(crisis_configs)].copy()
            config.update({
                'regime_window': config['regime_window'] + (i % 10),
                'crisis_thresholds': {
                    k: v + (i % 3) * 0.02 if 'spike' in k else v - (i % 3) * 0.05
                    for k, v in config['crisis_thresholds'].items()
                }
            })
            agent = CrisisDetectionAgent(f'crisis_det_{i}', config)
            self.agents.append(agent)
            self.agent_weights[agent.agent_id] = 1.2  # Higher weight for crisis detection

    def get_aggregate_signal(self, market_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Get aggregated signal from all active agents"""

        signals = []
        confidences = []
        agent_signals = {}

        for agent in self.agents:
            if not agent.is_active:
                continue

            try:
                signal = agent.evaluate_conditions(market_data)
                confidence = agent.get_confidence()

                signals.append(signal)
                confidences.append(confidence)
                agent_signals[agent.agent_id] = {
                    'signal': signal,
                    'confidence': confidence,
                    'type': agent.__class__.__name__
                }

            except Exception as e:
                logger.warning(f"Agent {agent.agent_id} failed: {e}")
                continue

        if not signals:
            return 0.0, {'error': 'No active agents'}

        # Weighted aggregation by confidence and agent performance
        weights = np.array(confidences)
        weights = weights / (np.sum(weights) + 1e-6)

        aggregate_signal = np.average(signals, weights=weights)

        # Apply risk thresholds
        if self._check_risk_limits(market_data):
            aggregate_signal = min(aggregate_signal, -0.5)  # Force risk reduction

        metadata = {
            'aggregate_signal': aggregate_signal,
            'active_agents': len(signals),
            'avg_confidence': np.mean(confidences),
            'signal_std': np.std(signals),
            'agent_signals': agent_signals,
            'risk_limits_triggered': self._check_risk_limits(market_data)
        }

        return aggregate_signal, metadata

    def _check_risk_limits(self, market_data: Dict[str, Any]) -> bool:
        """Check if any risk limits are violated"""
        current_drawdown = market_data.get('current_drawdown', 0.0)
        current_vol = market_data.get('current_vol', 0.02)

        return (current_drawdown > self.risk_thresholds['max_drawdown'] or
                current_vol > self.risk_thresholds['volatility_target'] * 2)

    def update_agent_performance(self, pnl: float, market_data: Dict[str, Any]):
        """Update all agents with performance feedback"""
        for agent in self.agents:
            agent.update_performance(pnl, market_data)

    def get_agent_health_report(self) -> Dict[str, Any]:
        """Get comprehensive agent health report"""
        agent_types = {}
        performance_stats = {}

        for agent in self.agents:
            agent_type = agent.__class__.__name__
            if agent_type not in agent_types:
                agent_types[agent_type] = []
            agent_types[agent_type].append(agent)

            perf = agent.get_recent_performance()
            performance_stats[agent.agent_id] = perf

        return {
            'total_agents': len(self.agents),
            'active_agents': sum(1 for a in self.agents if a.is_active),
            'agent_types': {k: len(v) for k, v in agent_types.items()},
            'performance_stats': performance_stats,
            'top_performers': sorted(
                [(k, v['sharpe']) for k, v in performance_stats.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }