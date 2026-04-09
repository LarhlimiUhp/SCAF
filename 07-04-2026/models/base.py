from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict_proba_one(self, X_row):
        raise NotImplementedError

    def predict_signal(self, X_row):
        prob, unc = self.predict_proba_one(X_row)
        return prob - 0.5, unc


class ModelRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str, constructor):
        self._registry[name] = constructor

    def create(self, name: str, **kwargs):
        if name not in self._registry:
            raise KeyError(f'Model {name} is not registered')
        return self._registry[name](**kwargs)

    def names(self) -> List[str]:
        return list(self._registry.keys())

    def all(self) -> Dict[str, Any]:
        return self._registry.copy()
