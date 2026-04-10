from .base import BaseModel
from .registry import registry
from .sklearn_models import *
from .torch_models import *
from .extra_models import *
from .conformal import SplitConformalClassifier, AdaptiveConformalClassifier, ConformalEnsemble
from .qlearning_selector import QLearningSelector

__all__ = ['BaseModel', 'registry'] + [
    'LogRegL2Model', 'RFClassModel', 'LightGBMModel', 'KNNClassModel',
    'LSTMClassModel', 'TabNetModel', 'GraphNNModel',
    # extra models (8 new)
    'XGBoostModel', 'ExtraTreesModel', 'AdaBoostModel', 'HistGBTModel',
    'MLPModel', 'RidgeClassModel', 'BaggingModel', 'CatBoostModel',
    # conformal
    'SplitConformalClassifier', 'AdaptiveConformalClassifier', 'ConformalEnsemble',
    # q-learning
    'QLearningSelector',
]
