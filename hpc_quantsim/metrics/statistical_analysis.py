"""
Statistical analysis tools for HPC QuantSim.

Provides advanced statistical analysis capabilities:
- Distribution analysis and fitting
- Statistical hypothesis testing
- Correlation and dependency analysis
- Time series analysis
- Regime detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
from scipy import stats, optimize
from dataclasses import dataclass
import warnings


@dataclass
class DistributionFit:
    """Results of distribution fitting."""
    distribution: str
    parameters: Dict[str, float]
    goodness_of_fit: float
    p_value: float
    aic: float
    bic: float


class DistributionAnalyzer:
    """
    Distribution analysis and fitting for financial returns.
    
    Features:
    - Multiple distribution fitting (normal, t, skew-t, etc.)
    - Goodness-of-fit testing
    - Model comparison
    - Fat tail analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Supported distributions
        self.distributions = {
            'normal': stats.norm,
            't': stats.t,
            'skewnorm': stats.skewnorm,
            'genextreme': stats.genextreme,  # GEV
            'laplace': stats.laplace,
            'logistic': stats.logistic,
            'uniform': stats.uniform,
        }
    
    def fit_distribution(self, data: Union[np.ndarray, pd.Series], 
                        distribution: str = 'normal') -> Optional[DistributionFit]:
        """Fit a specific distribution to the data."""
        if isinstance(data, pd.Series):
            data = data.dropna().values
        
        data = data[~np.isnan(data)]
        
        if len(data) < 10:
            self.logger.warning("Insufficient data for distribution fitting")
            return None
        
        if distribution not in self.distributions:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        dist = self.distributions[distribution]
        
        try:
            # Fit parameters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(data)
            
            # Calculate goodness of fit
            ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
            
            # Calculate information criteria
            log_likelihood = np.sum(dist.logpdf(data, *params))
            n_params = len(params)
            n_obs = len(data)
            
            aic = 2 * n_params - 2 * log_likelihood
            bic = np.log(n_obs) * n_params - 2 * log_likelihood
            
            # Create parameter dictionary
            param_names = dist.shapes.split(',') if dist.shapes else []
            param_names.extend(['loc', 'scale'])
            
            param_dict = dict(zip(param_names, params))
            
            return DistributionFit(
                distribution=distribution,
                parameters=param_dict,
                goodness_of_fit=ks_stat,
                p_value=p_value,
                aic=aic,
                bic=bic
            )
            
        except Exception as e:
            self.logger.error(f"Failed to fit {distribution} distribution: {e}")
            return None
    
    def fit_best_distribution(self, data: Union[np.ndarray, pd.Series]) -> List[DistributionFit]:
        """Fit multiple distributions and rank by goodness of fit."""
        results = []
        
        for dist_name in self.distributions.keys():
            fit_result = self.fit_distribution(data, dist_name)
            if fit_result is not None:
                results.append(fit_result)
        
        # Sort by AIC (lower is better)
        results.sort(key=lambda x: x.aic)
        
        return results
    
    def analyze_tails(self, data: Union[np.ndarray, pd.Series],
                     tail_threshold: float = 0.05) -> Dict[str, Any]:
        """Analyze tail behavior of the distribution."""
        if isinstance(data, pd.Series):
            data = data.dropna().values
        
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            return {}
        
        # Calculate tail statistics
        left_tail = np.percentile(data, tail_threshold * 100)
        right_tail = np.percentile(data, (1 - tail_threshold) * 100)
        
        left_tail_data = data[data <= left_tail]
        right_tail_data = data[data >= right_tail]
        
        return {
            'left_tail_threshold': left_tail,
            'right_tail_threshold': right_tail,
            'left_tail_observations': len(left_tail_data),
            'right_tail_observations': len(right_tail_data),
            'left_tail_mean': np.mean(left_tail_data) if len(left_tail_data) > 0 else 0,
            'right_tail_mean': np.mean(right_tail_data) if len(right_tail_data) > 0 else 0,
            'tail_ratio': len(right_tail_data) / len(left_tail_data) if len(left_tail_data) > 0 else np.inf,
        }


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for financial time series.
    
    Features:
    - Normality testing
    - Correlation analysis
    - Time series properties
    - Regime detection
    - Statistical hypothesis testing
    """
    
    def __init__(self):
        self.distribution_analyzer = DistributionAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def test_normality(self, data: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Test for normality using multiple methods."""
        if isinstance(data, pd.Series):
            data = data.dropna().values
        
        data = data[~np.isnan(data)]
        
        if len(data) < 8:
            return {'error': 'Insufficient data for normality testing'}
        
        results = {}
        
        # Shapiro-Wilk test (good for small samples)
        if len(data) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                results['shapiro_wilk'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            except Exception as e:
                self.logger.warning(f"Shapiro-Wilk test failed: {e}")
        
        # Kolmogorov-Smirnov test against normal
        try:
            mean, std = np.mean(data), np.std(data)
            ks_stat, ks_p = stats.kstest(data, lambda x: stats.norm.cdf(x, mean, std))
            results['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > 0.05
            }
        except Exception as e:
            self.logger.warning(f"K-S test failed: {e}")
        
        # D'Agostino's normality test
        try:
            dagostino_stat, dagostino_p = stats.normaltest(data)
            results['dagostino'] = {
                'statistic': dagostino_stat,
                'p_value': dagostino_p,
                'is_normal': dagostino_p > 0.05
            }
        except Exception as e:
            self.logger.warning(f"D'Agostino test failed: {e}")
        
        # Jarque-Bera test
        try:
            jb_stat, jb_p = stats.jarque_bera(data)
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > 0.05
            }
        except Exception as e:
            self.logger.warning(f"Jarque-Bera test failed: {e}")
        
        return results
    
    def test_stationarity(self, data: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        if isinstance(data, pd.Series):
            data = data.dropna().values
        
        data = data[~np.isnan(data)]
        
        if len(data) < 10:
            return {'error': 'Insufficient data for stationarity testing'}
        
        try:
            # Simple ADF test implementation
            # Full implementation would require statsmodels
            
            # Calculate first differences
            diff_data = np.diff(data)
            
            # Basic test - check if mean and variance are stable
            n = len(data)
            mid = n // 2
            
            first_half_mean = np.mean(data[:mid])
            second_half_mean = np.mean(data[mid:])
            first_half_var = np.var(data[:mid])
            second_half_var = np.var(data[mid:])
            
            mean_diff = abs(first_half_mean - second_half_mean)
            var_ratio = max(first_half_var, second_half_var) / max(min(first_half_var, second_half_var), 1e-10)
            
            return {
                'method': 'basic_stability_test',
                'mean_difference': mean_diff,
                'variance_ratio': var_ratio,
                'likely_stationary': mean_diff < np.std(data) * 0.5 and var_ratio < 2.0,
                'first_diff_volatility': np.std(diff_data),
                'note': 'Basic test - use statsmodels ADF for rigorous testing'
            }
            
        except Exception as e:
            return {'error': f'Stationarity test failed: {e}'}
    
    def calculate_correlations(self, data: pd.DataFrame, 
                             method: str = 'pearson') -> Dict[str, Any]:
        """Calculate correlation matrix and statistics."""
        try:
            # Calculate correlation matrix
            if method == 'pearson':
                corr_matrix = data.corr(method='pearson')
            elif method == 'spearman':
                corr_matrix = data.corr(method='spearman')
            elif method == 'kendall':
                corr_matrix = data.corr(method='kendall')
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            # Calculate correlation statistics
            corr_values = corr_matrix.values
            
            # Remove diagonal (self-correlation)
            mask = ~np.eye(corr_values.shape[0], dtype=bool)
            off_diagonal = corr_values[mask]
            
            return {
                'correlation_matrix': corr_matrix,
                'method': method,
                'statistics': {
                    'mean_correlation': np.mean(off_diagonal),
                    'max_correlation': np.max(off_diagonal),
                    'min_correlation': np.min(off_diagonal),
                    'std_correlation': np.std(off_diagonal),
                },
                'highly_correlated_pairs': self._find_high_correlations(corr_matrix, threshold=0.7)
            }
            
        except Exception as e:
            return {'error': f'Correlation analysis failed: {e}'}
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, 
                               threshold: float = 0.7) -> List[Dict]:
        """Find pairs of variables with high correlation."""
        high_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corrs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return sorted(high_corrs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def detect_outliers(self, data: Union[np.ndarray, pd.Series],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, Any]:
        """Detect outliers using various methods."""
        if isinstance(data, pd.Series):
            clean_data = data.dropna().values
            index = data.dropna().index
        else:
            clean_data = data[~np.isnan(data)]
            index = np.arange(len(clean_data))
        
        if len(clean_data) == 0:
            return {'outliers': [], 'method': method}
        
        outlier_indices = []
        
        if method == 'iqr':
            q1 = np.percentile(clean_data, 25)
            q3 = np.percentile(clean_data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outlier_mask = (clean_data < lower_bound) | (clean_data > upper_bound)
            outlier_indices = index[outlier_mask]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(clean_data))
            outlier_mask = z_scores > threshold
            outlier_indices = index[outlier_mask]
            
        elif method == 'modified_zscore':
            median = np.median(clean_data)
            mad = np.median(np.abs(clean_data - median))
            modified_z_scores = 0.6745 * (clean_data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
            outlier_indices = index[outlier_mask]
        
        return {
            'outliers': outlier_indices.tolist() if hasattr(outlier_indices, 'tolist') else list(outlier_indices),
            'outlier_count': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(clean_data) * 100,
            'method': method,
            'threshold': threshold
        }
    
    def analyze_time_series_properties(self, data: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Analyze basic time series properties."""
        if isinstance(data, pd.Series):
            data = data.dropna()
        else:
            data = pd.Series(data)
            data = data.dropna()
        
        if len(data) == 0:
            return {}
        
        # Basic statistics
        basic_stats = {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
        }
        
        # Autocorrelation
        try:
            autocorr_1 = data.autocorr(lag=1) if len(data) > 1 else 0
            autocorr_5 = data.autocorr(lag=5) if len(data) > 5 else 0
            autocorr_10 = data.autocorr(lag=10) if len(data) > 10 else 0
        except:
            autocorr_1 = autocorr_5 = autocorr_10 = 0
        
        autocorr_stats = {
            'autocorr_lag1': autocorr_1,
            'autocorr_lag5': autocorr_5,
            'autocorr_lag10': autocorr_10,
        }
        
        # Normality test
        normality = self.test_normality(data.values)
        
        # Stationarity test
        stationarity = self.test_stationarity(data.values)
        
        # Outlier detection
        outliers = self.detect_outliers(data.values)
        
        return {
            'basic_statistics': basic_stats,
            'autocorrelation': autocorr_stats,
            'normality_tests': normality,
            'stationarity_test': stationarity,
            'outlier_analysis': outliers,
        }
