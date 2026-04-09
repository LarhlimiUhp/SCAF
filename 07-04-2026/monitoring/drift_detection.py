"""
Drift Detection - Data Drift, Model Drift, Concept Drift
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
from collections import deque


@dataclass
class DriftMetrics:
    """Métriques de drift"""
    timestamp: str
    data_drift_score: float
    model_drift_score: float
    concept_drift_score: float
    overall_drift: float
    is_drifting: bool
    drift_indicators: Dict[str, bool]


class DriftDetector:
    """Détecteur de drift multi-types"""
    
    def __init__(self, reference_window_size: int = 1000, test_window_size: int = 100):
        self.reference_window_size = reference_window_size
        self.test_window_size = test_window_size
        
        # Historiques
        self.reference_data: Optional[np.ndarray] = None
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_targets: Optional[np.ndarray] = None
        
        self.test_data_history: deque = deque(maxlen=test_window_size)
        self.test_predictions_history: deque = deque(maxlen=test_window_size)
        self.test_targets_history: deque = deque(maxlen=test_window_size)
        
        self.drift_history: deque = deque(maxlen=1000)
        
    def set_reference_data(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                           predictions: Optional[np.ndarray] = None):
        """Définir les données de référence"""
        self.reference_data = X
        self.reference_targets = y
        self.reference_predictions = predictions
    
    def add_test_sample(self, X_sample: np.ndarray, y_true: Optional[float] = None,
                        y_pred: Optional[float] = None):
        """Ajouter un échantillon de test"""
        self.test_data_history.append(X_sample)
        if y_true is not None:
            self.test_targets_history.append(y_true)
        if y_pred is not None:
            self.test_predictions_history.append(y_pred)
    
    def detect_data_drift(self, method: str = "ks") -> float:
        """
        Détecter la dérive des données
        
        Methods:
        - "ks": Kolmogorov-Smirnov test
        - "manhattan": Distance de Manhattan
        - "wasserstein": Distance Wasserstein (earth mover's distance)
        """
        if self.reference_data is None or len(self.test_data_history) == 0:
            return 0.0
        
        test_data = np.array(list(self.test_data_history))
        
        if method == "ks":
            # Kolmogorov-Smirnov test
            p_values = []
            for i in range(self.reference_data.shape[1]):
                try:
                    _, p_value = stats.ks_2samp(
                        self.reference_data[:, i],
                        test_data[:, i]
                    )
                    p_values.append(p_value)
                except:
                    pass
            
            # Score: moyenne des p-values inversées (plus élevé = plus de drift)
            if p_values:
                drift_score = 1 - np.mean(p_values)
            else:
                drift_score = 0.0
        
        elif method == "manhattan":
            # Distance de Manhattan normalisée
            ref_mean = np.mean(self.reference_data, axis=0)
            test_mean = np.mean(test_data, axis=0)
            
            distance = np.sum(np.abs(ref_mean - test_mean))
            max_distance = np.sum(np.abs(np.max(self.reference_data, axis=0) - 
                                        np.min(self.reference_data, axis=0)))
            
            drift_score = distance / max_distance if max_distance > 0 else 0.0
        
        elif method == "wasserstein":
            # Distance Wasserstein (approximée)
            ref_mean = np.mean(self.reference_data, axis=0)
            test_mean = np.mean(test_data, axis=0)
            ref_std = np.std(self.reference_data, axis=0)
            test_std = np.std(test_data, axis=0)
            
            # Approximation: sqrt(mean_diff^2 + std_diff^2)
            mean_diff = np.sum((ref_mean - test_mean) ** 2)
            std_diff = np.sum((ref_std - test_std) ** 2)
            
            drift_score = np.sqrt(mean_diff + std_diff)
            drift_score = min(1.0, drift_score)
        
        else:
            drift_score = 0.0
        
        return float(drift_score)
    
    def detect_model_drift(self) -> float:
        """
        Détecter la dérive du modèle (changement de performance)
        """
        if (self.reference_predictions is None or
            self.reference_targets is None or
            len(self.test_predictions_history) == 0 or
            len(self.test_targets_history) == 0):
            return 0.0
        
        # Comparer les erreurs de prédiction entre les ensembles
        ref_errors = np.abs(self.reference_predictions - self.reference_targets)
        test_errors = np.abs(np.array(list(self.test_predictions_history)) - 
                             np.array(list(self.test_targets_history)))
        
        if len(test_errors) == 0:
            return 0.0
        
        ref_mean_error = np.mean(ref_errors)
        test_mean_error = np.mean(test_errors)
        
        # Drift score: augmentation relative de l'erreur
        if ref_mean_error > 0:
            drift_score = max(0, (test_mean_error - ref_mean_error) / ref_mean_error)
            drift_score = min(1.0, drift_score)
        else:
            drift_score = 0.0
        
        return float(drift_score)
    
    def detect_concept_drift(self, window_size: int = 50) -> float:
        """
        Détecter le concept drift (changement de la relation input-output)
        """
        if (len(self.test_data_history) < window_size or 
            len(self.test_targets_history) < window_size):
            return 0.0
        
        # Diviser les données de test réantes en deux fenêtres
        mid_point = len(self.test_data_history) // 2
        
        first_half_data = np.array(list(self.test_data_history))[:mid_point]
        first_half_targets = np.array(list(self.test_targets_history))[:mid_point]
        
        second_half_data = np.array(list(self.test_data_history))[mid_point:]
        second_half_targets = np.array(list(self.test_targets_history))[mid_point:]
        
        if len(first_half_data) == 0 or len(second_half_data) == 0:
            return 0.0
        
        # Calculer les correlations moyenne
        try:
            first_corr = np.mean([
                np.corrcoef(first_half_data[:, i], first_half_targets)[0, 1]
                for i in range(first_half_data.shape[1])
            ])
            
            second_corr = np.mean([
                np.corrcoef(second_half_data[:, i], second_half_targets)[0, 1]
                for i in range(second_half_data.shape[1])
            ])
            
            # Drift score: différence de corrélation
            drift_score = abs(first_corr - second_corr)
            drift_score = min(1.0, drift_score)
        except:
            drift_score = 0.0
        
        return float(drift_score)
    
    def get_drift_metrics(self) -> DriftMetrics:
        """Obtenir toutes les métriques de drift"""
        data_drift = self.detect_data_drift(method="ks")
        model_drift = self.detect_model_drift()
        concept_drift = self.detect_concept_drift()
        
        # Score global: moyenne pondérée
        overall_drift = (0.3 * data_drift + 0.4 * model_drift + 0.3 * concept_drift)
        
        # Déterminer si drifting
        drift_indicators = {
            'data_drift': data_drift > 0.5,
            'model_drift': model_drift > 0.3,
            'concept_drift': concept_drift > 0.5,
        }
        
        is_drifting = any(drift_indicators.values())
        
        metrics = DriftMetrics(
            timestamp=datetime.utcnow().isoformat(),
            data_drift_score=data_drift,
            model_drift_score=model_drift,
            concept_drift_score=concept_drift,
            overall_drift=overall_drift,
            is_drifting=is_drifting,
            drift_indicators=drift_indicators,
        )
        
        self.drift_history.append(metrics)
        
        return metrics
    
    def get_drift_trend(self, window_size: int = 30) -> Dict[str, float]:
        """Obtenir la tendance du drift"""
        if len(self.drift_history) < window_size:
            window_size = len(self.drift_history)
        
        if window_size == 0:
            return {
                'data_drift_trend': 0.0,
                'model_drift_trend': 0.0,
                'concept_drift_trend': 0.0,
                'overall_drift_trend': 0.0,
            }
        
        recent_metrics = list(self.drift_history)[-window_size:]
        
        data_drifts = [m.data_drift_score for m in recent_metrics]
        model_drifts = [m.model_drift_score for m in recent_metrics]
        concept_drifts = [m.concept_drift_score for m in recent_metrics]
        overall_drifts = [m.overall_drift for m in recent_metrics]
        
        return {
            'data_drift_trend': float(np.polyfit(range(len(data_drifts)), data_drifts, 1)[0]),
            'model_drift_trend': float(np.polyfit(range(len(model_drifts)), model_drifts, 1)[0]),
            'concept_drift_trend': float(np.polyfit(range(len(concept_drifts)), concept_drifts, 1)[0]),
            'overall_drift_trend': float(np.polyfit(range(len(overall_drifts)), overall_drifts, 1)[0]),
        }
    
    def should_retrain(self, overall_drift_threshold: float = 0.6,
                       trend_threshold: float = 0.01) -> bool:
        """Déterminer si un réentraînement est nécessaire"""
        if not self.drift_history:
            return False
        
        latest_metrics = self.drift_history[-1]
        
        # Condition 1: Drift global dépassé
        if latest_metrics.overall_drift > overall_drift_threshold:
            return True
        
        # Condition 2: Tendance croissante de drift
        trends = self.get_drift_trend(window_size=20)
        if trends['overall_drift_trend'] > trend_threshold:
            return True
        
        return False
