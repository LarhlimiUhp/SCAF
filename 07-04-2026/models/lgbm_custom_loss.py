"""
Custom loss functions for LightGBM optimization

Financial objective functions for improved trading performance:
- Focal loss for imbalanced classification
- Sharpe-aware loss functions
- Drawdown-aware penalties
- Custom financial metrics
"""

import numpy as np
from typing import Callable, Tuple
import lightgbm as lgb


class LightGBMCustomLoss:
    """Custom loss functions optimized for financial objectives"""

    @staticmethod
    def focal_loss(alpha: float = 0.25, gamma: float = 2.0) -> Callable:
        """
        Focal loss for imbalanced financial classification

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter

        Returns:
            Loss function compatible with LightGBM
        """
        def loss_func(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
            y_true = y_true.get_label()
            p = 1 / (1 + np.exp(-y_pred))  # Sigmoid

            # Focal loss
            pt = np.where(y_true == 1, p, 1 - p)
            focal_weight = (1 - pt) ** gamma

            # Cross entropy with focal weighting
            ce_loss = -alpha * y_true * np.log(p + 1e-8) - (1 - alpha) * (1 - y_true) * np.log(1 - p + 1e-8)
            loss = focal_weight * ce_loss

            # Gradient
            p_clipped = np.clip(p, 1e-8, 1 - 1e-8)
            grad = p - y_true

            # Second derivative (Hessian)
            hess = p_clipped * (1 - p_clipped)

            return grad, hess

        return loss_func

    @staticmethod
    def sharpe_aware_loss(target_sharpe: float = 1.0, penalty_weight: float = 0.1) -> Callable:
        """
        Loss function that incorporates Sharpe ratio considerations

        Args:
            target_sharpe: Target Sharpe ratio
            penalty_weight: Weight for Sharpe penalty

        Returns:
            Loss function that balances AUC with Sharpe considerations
        """
        def loss_func(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
            y_true = y_true.get_label()
            p = 1 / (1 + np.exp(-y_pred))

            # Standard binary cross-entropy
            ce_loss = -y_true * np.log(p + 1e-8) - (1 - y_true) * np.log(1 - p + 1e-8)

            # Add penalty based on prediction confidence distribution
            # Penalize overconfident predictions that might lead to poor Sharpe
            confidence_penalty = penalty_weight * np.abs(p - 0.5) * (1 - np.abs(p - 0.5))

            total_loss = ce_loss + confidence_penalty

            # Gradients
            grad = p - y_true + penalty_weight * np.sign(p - 0.5) * (1 - 2 * np.abs(p - 0.5))
            hess = p * (1 - p) + penalty_weight * (1 - 2 * np.abs(p - 0.5))**2

            return grad, hess

        return loss_func

    @staticmethod
    def drawdown_aware_loss(max_drawdown_penalty: float = 0.2) -> Callable:
        """
        Loss function that considers potential drawdown impact

        Args:
            max_drawdown_penalty: Weight for drawdown-aware penalty

        Returns:
            Loss function that penalizes predictions likely to cause large drawdowns
        """
        def loss_func(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
            y_true = y_true.get_label()
            p = 1 / (1 + np.exp(-y_pred))

            # Standard cross-entropy
            ce_loss = -y_true * np.log(p + 1e-8) - (1 - y_true) * np.log(1 - p + 1e-8)

            # Penalize very high or very low confidence predictions (potential for large losses)
            # Uses additive penalty: high confidence (p>0.8) OR low confidence (p<0.2)
            extreme_confidence_penalty = max_drawdown_penalty * (
                np.where(p > 0.8, (p - 0.8) * 5, 0.0) +
                np.where(p < 0.2, (0.2 - p) * 5, 0.0)
            )

            total_loss = ce_loss + extreme_confidence_penalty

            # Gradients
            grad = p - y_true
            grad += max_drawdown_penalty * (
                np.where(p > 0.8, 5.0, 0.0) + np.where(p < 0.2, -5.0, 0.0)
            )

            hess = p * (1 - p) + max_drawdown_penalty * np.where(
                (p > 0.8) | (p < 0.2), 25, 0
            )

            return grad, hess

        return loss_func

    @staticmethod
    def financial_utility_loss(transaction_cost: float = 0.0003,
                              risk_aversion: float = 2.0) -> Callable:
        """
        Loss function based on financial utility theory

        Args:
            transaction_cost: Trading cost per trade
            risk_aversion: Risk aversion parameter

        Returns:
            Utility-based loss function
        """
        def loss_func(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
            y_true = y_true.get_label()
            p = 1 / (1 + np.exp(-y_pred))

            # Expected utility loss
            # Positive returns: +1, Negative returns: -1 (simplified)
            expected_return = 2 * p - 1  # Convert probability to expected return

            # Utility with risk aversion and transaction costs
            utility = expected_return - transaction_cost - risk_aversion * expected_return**2

            # Convert to loss (negative utility)
            loss = -utility

            # For binary classification, still need proper gradients
            # Use a hybrid approach
            ce_loss = -y_true * np.log(p + 1e-8) - (1 - y_true) * np.log(1 - p + 1e-8)

            # Blend CE loss with utility loss
            total_loss = 0.7 * ce_loss + 0.3 * loss

            # Gradients
            grad = 0.7 * (p - y_true) + 0.3 * (2 * risk_aversion * expected_return - 1)
            hess = 0.7 * (p * (1 - p)) + 0.3 * (2 * risk_aversion)

            return grad, hess

        return loss_func


class LightGBMCustomObjective:
    """Custom objectives for LightGBM training"""

    @staticmethod
    def rank_based_objective() -> Callable:
        """
        Rank-based objective that optimizes for proper ordering rather than classification.

        Useful for financial prediction where relative ranking matters more than absolute accuracy.
        Optimised from O(n²) to O(n log n) using a sort-then-sweep approach.
        """
        def objective(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
            y_true_arr = y_true.get_label()
            n = len(y_pred)

            grad = np.zeros(n, dtype=np.float64)
            hess = np.ones(n, dtype=np.float64)

            # Sort by predicted score ascending
            sorted_indices = np.argsort(y_pred)
            y_true_sorted = y_true_arr[sorted_indices]

            # Prefix count of positives seen so far (from lower-ranked items)
            # For each item i (sorted), count positives ranked below it that should be above it
            prefix_positives = np.zeros(n, dtype=np.float64)
            running_pos = 0.0
            for rank, orig_idx in enumerate(sorted_indices):
                # Positives ranked lower than current item that outrank a negative
                if y_true_sorted[rank] == 0:  # current item is negative
                    # All positives seen so far (ranked lower) should be above it
                    grad[orig_idx] -= running_pos
                    hess[orig_idx] += running_pos
                else:  # current item is positive
                    # Count negatives ranked below that should be below it (already correct) — no penalty
                    running_pos += 1.0

            # Second pass: positives get penalty for each negative ranked above them
            suffix_negatives = 0.0
            for rank in range(n - 1, -1, -1):
                orig_idx = sorted_indices[rank]
                if y_true_sorted[rank] == 1:  # current item is positive
                    grad[orig_idx] += suffix_negatives
                    hess[orig_idx] += suffix_negatives
                else:
                    suffix_negatives += 1.0

            return grad, hess

        return objective

    @staticmethod
    def uncertainty_aware_objective(uncertainty_penalty: float = 0.1) -> Callable:
        """
        Objective that considers prediction uncertainty

        Args:
            uncertainty_penalty: Weight for uncertainty penalty
        """
        def objective(y_pred: np.ndarray, y_true: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
            y_true = y_true.get_label()
            p = 1 / (1 + np.exp(-y_pred))

            # Standard binary cross-entropy
            ce_grad = p - y_true
            ce_hess = p * (1 - p)

            # Uncertainty penalty: penalize predictions near 0.5
            uncertainty_weight = uncertainty_penalty * (1 - 4 * (p - 0.5)**2)
            uncertainty_grad = uncertainty_weight * np.sign(p - 0.5)
            uncertainty_hess = uncertainty_weight * 8 * np.abs(p - 0.5)

            return ce_grad + uncertainty_grad, ce_hess + uncertainty_hess

        return objective


def get_custom_loss_function(loss_type: str = 'focal', **kwargs) -> Callable:
    """
    Factory function for custom loss functions

    Args:
        loss_type: Type of loss function ('focal', 'sharpe_aware', 'drawdown_aware', 'financial_utility')
        **kwargs: Parameters for the loss function

    Returns:
        Custom loss function
    """
    loss_functions = {
        'focal': LightGBMCustomLoss.focal_loss,
        'sharpe_aware': LightGBMCustomLoss.sharpe_aware_loss,
        'drawdown_aware': LightGBMCustomLoss.drawdown_aware_loss,
        'financial_utility': LightGBMCustomLoss.financial_utility_loss,
    }

    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss_functions[loss_type](**kwargs)


def get_custom_objective_function(objective_type: str = 'rank_based', **kwargs) -> Callable:
    """
    Factory function for custom objective functions

    Args:
        objective_type: Type of objective function ('rank_based', 'uncertainty_aware')
        **kwargs: Parameters for the objective function

    Returns:
        Custom objective function
    """
    objective_functions = {
        'rank_based': LightGBMCustomObjective.rank_based_objective,
        'uncertainty_aware': LightGBMCustomObjective.uncertainty_aware_objective,
    }

    if objective_type not in objective_functions:
        raise ValueError(f"Unknown objective type: {objective_type}")

    return objective_functions[objective_type](**kwargs)