"""
Risk metrics calculation for HPC QuantSim.

Provides comprehensive risk analysis:
- Value at Risk (VaR) calculations
- Conditional Value at Risk (CVaR)
- Maximum Drawdown analysis
- Risk-adjusted performance metrics
- Portfolio risk decomposition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from scipy import stats
from dataclasses import dataclass


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    var: float
    confidence_level: float
    method: str
    horizon_days: int
    observations: int
    
    def __dict__(self):
        return {
            'var': self.var,
            'confidence_level': self.confidence_level,
            'method': self.method,
            'horizon_days': self.horizon_days,
            'observations': self.observations
        }


class VaRCalculator:
    """
    Value at Risk calculator with multiple methodologies.
    
    Supports:
    - Historical simulation
    - Parametric (Normal) VaR
    - Monte Carlo simulation
    - Cornish-Fisher expansion
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def historical_var(self, returns: Union[np.ndarray, pd.Series],
                      confidence_level: float = 0.95,
                      horizon_days: int = 1) -> VaRResult:
        """Calculate VaR using historical simulation."""
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return VaRResult(0, confidence_level, 'historical', horizon_days, 0)
        
        # Scale returns to horizon
        scaled_returns = returns * np.sqrt(horizon_days)
        
        # Calculate percentile
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(scaled_returns, var_percentile)
        
        return VaRResult(var, confidence_level, 'historical', horizon_days, len(returns))
    
    def parametric_var(self, returns: Union[np.ndarray, pd.Series],
                      confidence_level: float = 0.95,
                      horizon_days: int = 1) -> VaRResult:
        """Calculate VaR using parametric (normal) method."""
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return VaRResult(0, confidence_level, 'parametric', horizon_days, 0)
        
        # Calculate mean and std
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Scale to horizon
        horizon_mean = mean_return * horizon_days
        horizon_std = std_return * np.sqrt(horizon_days)
        
        # Calculate z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        var = horizon_mean + z_score * horizon_std
        
        return VaRResult(var, confidence_level, 'parametric', horizon_days, len(returns))
    
    def monte_carlo_var(self, returns: Union[np.ndarray, pd.Series],
                       confidence_level: float = 0.95,
                       horizon_days: int = 1,
                       num_simulations: int = 10000) -> VaRResult:
        """Calculate VaR using Monte Carlo simulation."""
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return VaRResult(0, confidence_level, 'monte_carlo', horizon_days, 0)
        
        # Estimate parameters
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate random scenarios
        random_returns = np.random.normal(mean_return, std_return, 
                                        (num_simulations, horizon_days))
        
        # Calculate cumulative returns for each simulation
        cumulative_returns = np.prod(1 + random_returns, axis=1) - 1
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(cumulative_returns, var_percentile)
        
        return VaRResult(var, confidence_level, 'monte_carlo', horizon_days, len(returns))


class DrawdownAnalyzer:
    """
    Drawdown analysis and metrics calculation.
    
    Provides detailed drawdown statistics including:
    - Maximum drawdown
    - Average drawdown
    - Drawdown duration statistics
    - Recovery time analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_drawdowns(self, returns: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """Calculate comprehensive drawdown statistics."""
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdowns
        drawdowns = (cumulative - running_max) / running_max
        
        # Find drawdown periods
        drawdown_periods = self._identify_drawdown_periods(drawdowns)
        
        return {
            'max_drawdown': drawdowns.min(),
            'max_drawdown_pct': drawdowns.min() * 100,
            'avg_drawdown': drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0,
            'drawdown_periods': len(drawdown_periods),
            'max_drawdown_duration': max([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0,
            'avg_drawdown_duration': np.mean([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0,
            'max_recovery_time': max([p['recovery_time'] for p in drawdown_periods if p['recovery_time'] > 0]) if drawdown_periods else 0,
            'drawdown_series': drawdowns,
            'drawdown_details': drawdown_periods
        }
    
    def _identify_drawdown_periods(self, drawdowns: pd.Series) -> List[Dict]:
        """Identify individual drawdown periods."""
        periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdowns):
            if not in_drawdown and dd < 0:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
            elif in_drawdown and dd >= 0:
                # End of drawdown
                in_drawdown = False
                
                # Calculate period statistics
                period_drawdowns = drawdowns.iloc[start_idx:i]
                max_dd = period_drawdowns.min()
                duration = i - start_idx
                
                # Find recovery time (if any)
                recovery_time = 0
                for j in range(i, len(drawdowns)):
                    if drawdowns.iloc[j] >= 0:
                        recovery_time = j - i
                        break
                
                periods.append({
                    'start_idx': start_idx,
                    'end_idx': i - 1,
                    'duration': duration,
                    'max_drawdown': max_dd,
                    'recovery_time': recovery_time
                })
        
        # Handle case where series ends in drawdown
        if in_drawdown and start_idx is not None:
            period_drawdowns = drawdowns.iloc[start_idx:]
            max_dd = period_drawdowns.min()
            duration = len(drawdowns) - start_idx
            
            periods.append({
                'start_idx': start_idx,
                'end_idx': len(drawdowns) - 1,
                'duration': duration,
                'max_drawdown': max_dd,
                'recovery_time': -1  # Not recovered
            })
        
        return periods


class RiskMetrics:
    """
    Comprehensive risk metrics calculation.
    
    Combines various risk measures into a unified interface:
    - VaR and CVaR calculations
    - Drawdown analysis
    - Volatility metrics
    - Risk-adjusted returns
    """
    
    def __init__(self):
        self.var_calculator = VaRCalculator()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def calculate_var(self, returns: Union[np.ndarray, pd.Series],
                     confidence_level: float = 0.95,
                     method: str = 'historical',
                     horizon_days: int = 1) -> VaRResult:
        """Calculate Value at Risk using specified method."""
        if method == 'historical':
            return self.var_calculator.historical_var(returns, confidence_level, horizon_days)
        elif method == 'parametric':
            return self.var_calculator.parametric_var(returns, confidence_level, horizon_days)
        elif method == 'monte_carlo':
            return self.var_calculator.monte_carlo_var(returns, confidence_level, horizon_days)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def calculate_cvar(self, returns: Union[np.ndarray, pd.Series],
                      confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate VaR threshold
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)
        
        # Calculate CVaR as mean of returns below VaR
        tail_returns = returns[returns <= var_threshold]
        
        return np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
    
    def calculate_risk_metrics(self, returns: Union[np.ndarray, pd.Series],
                              confidence_levels: List[float] = [0.95, 0.99],
                              risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Annualized metrics (assuming daily data)
        annual_return = mean_return * 252
        annual_vol = std_return * np.sqrt(252)
        
        # Risk-adjusted returns
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        sortino_ratio = annual_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
        
        # VaR and CVaR for different confidence levels
        var_metrics = {}
        for cl in confidence_levels:
            var_hist = self.calculate_var(returns, cl, 'historical')
            var_para = self.calculate_var(returns, cl, 'parametric')
            cvar = self.calculate_cvar(returns, cl)
            
            var_metrics[f'var_{int(cl*100)}'] = {
                'historical': var_hist.var,
                'parametric': var_para.var
            }
            var_metrics[f'cvar_{int(cl*100)}'] = cvar
        
        # Drawdown analysis
        drawdown_stats = self.drawdown_analyzer.calculate_drawdowns(returns)
        
        # Additional risk metrics
        downside_deviation = returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'return_statistics': {
                'mean_daily': mean_return,
                'std_daily': std_return,
                'annualized_return': annual_return,
                'annualized_volatility': annual_vol,
                'skewness': skewness,
                'kurtosis': kurtosis,
            },
            'risk_adjusted_returns': {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': annual_return / abs(drawdown_stats.get('max_drawdown', 1)) if drawdown_stats.get('max_drawdown', 0) < 0 else 0,
            },
            'var_cvar_metrics': var_metrics,
            'drawdown_analysis': drawdown_stats,
            'risk_metrics': {
                'downside_deviation': downside_deviation,
                'upside_deviation': returns[returns > 0].std() if len(returns[returns > 0]) > 0 else 0,
                'positive_periods': len(returns[returns > 0]),
                'negative_periods': len(returns[returns < 0]),
                'win_rate': len(returns[returns > 0]) / len(returns) * 100 if len(returns) > 0 else 0,
            }
        }
