import numpy as np
from sklearn.inspection import permutation_importance


class ImportanceAnalyzer:
    @staticmethod
    def permutation_importance(model, X, y, n_repeats=10, random_state=42):
        try:
            result = permutation_importance(model, X, y,
                                           n_repeats=n_repeats,
                                           random_state=random_state,
                                           n_jobs=-1)
            return {
                'importances_mean': result.importances_mean,
                'importances_std': result.importances_std,
                'feature_names': np.array(result.importances_mean).tolist(),
            }
        except Exception:
            return None

    @staticmethod
    def shap_summary(model, X):
        try:
            import shap
            explainer = shap.Explainer(model.predict, X)
            shap_values = explainer(X)
            return shap_values
        except Exception:
            return None
