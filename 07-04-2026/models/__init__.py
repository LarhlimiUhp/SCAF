from .base import BaseModel
from .registry import registry
from .sklearn_models import *

__all__ = ['BaseModel', 'registry', 'LogRegL2Model', 'RFClassModel', 'LightGBMModel', 'KNNClassModel']

try:
    from .torch_models import *
    __all__ += ['LSTMClassModel', 'TabNetModel', 'GraphNNModel']
except ImportError:
    pass

from .extra_models import *  # registers XGBoost, ExtraTrees, AdaBoost, HistGBT, MLP, RidgeClass, BaggingLR, CatBoost
__all__ += ['XGBoostModel', 'ExtraTreesModel', 'AdaBoostModel', 'HistGBTModel',
            'MLPClassModel', 'RidgeClassModel', 'BaggingLRModel', 'CatBoostModel']

from .conformal import SplitConformalClassifier, AdaptiveConformalClassifier, ConformalEnsemble
from .qlearning_selector import QLearningModelSelector
__all__ += ['SplitConformalClassifier', 'AdaptiveConformalClassifier',
            'ConformalEnsemble', 'QLearningModelSelector']
