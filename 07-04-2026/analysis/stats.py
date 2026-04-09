import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


class StatisticalAnalyzer:
    @staticmethod
    def adf_test(series, signif=0.05):
        result = adfuller(series.dropna())
        return {
            'adf_stat': result[0],
            'p_value': result[1],
            'used_lag': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'stationary': result[1] < signif,
        }

    @staticmethod
    def performance_ttest(strategy_returns, benchmark_returns):
        tstat, pvalue = stats.ttest_rel(strategy_returns, benchmark_returns, nan_policy='omit')
        return {
            't_stat': float(tstat),
            'p_value': float(pvalue),
            'outperform': pvalue < 0.05 and np.mean(strategy_returns) > np.mean(benchmark_returns)
        }

    @staticmethod
    def residual_independence(residuals, lags=10):
        result = acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)
        return {
            'ljung_box_stat': float(result['lb_stat'].iloc[0]),
            'p_value': float(result['lb_pvalue'].iloc[0]),
            'independent': float(result['lb_pvalue'].iloc[0]) > 0.05
        }
