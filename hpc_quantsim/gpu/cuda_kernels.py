"""
CUDA kernels for high-performance strategy execution and computation in HPC QuantSim.

Provides GPU-accelerated implementations of:
- Parallel strategy execution across multiple scenarios
- Technical indicator calculations (moving averages, RSI, etc.)
- Statistical computations (returns, volatility, correlations)
- Matrix operations for portfolio optimization
- Risk metric calculations
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import time

# Try to import CUDA libraries
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = None

from .gpu_memory_pool import get_global_pool, GPUMemoryPool
from .gpu_utils import GPUTimer, GPUUtils


class CUDAKernels:
    """
    High-performance CUDA kernels for quantitative finance computations.
    
    Features:
    - Vectorized strategy calculations
    - Parallel scenario execution
    - Technical indicator computations
    - Risk metric calculations
    - Memory-efficient implementations
    """
    
    def __init__(self, device_id: int = 0, memory_pool: Optional[GPUMemoryPool] = None):
        """Initialize CUDA kernels."""
        if not HAS_CUDA:
            raise ImportError("CUDA/CuPy not available")
        
        self.device_id = device_id
        self.device = cp.cuda.Device(device_id)
        self.logger = logging.getLogger(__name__)
        
        # Memory pool for efficient allocation
        self.memory_pool = memory_pool or get_global_pool(device_id)
        
        # Performance tracking
        self.kernel_stats = {
            'executions': 0,
            'total_time_ms': 0,
            'average_time_ms': 0,
        }
        
        # CUDA streams for concurrent execution
        self.streams = GPUUtils.create_streams(4)
        
        with self.device:
            # Compile and cache frequently used kernels
            self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile and cache CUDA kernels."""
        self.logger.info("Compiling CUDA kernels...")
        
        # Moving average kernel
        self.moving_average_kernel = cp.ElementwiseKernel(
            'raw T prices, int32 window_size, int32 n',
            'T ma_values',
            '''
            int idx = i;
            if (idx < window_size - 1) {
                ma_values = 0.0;  // Not enough data
                return;
            }
            
            T sum = 0.0;
            for (int j = 0; j < window_size; j++) {
                sum += prices[idx - j];
            }
            ma_values = sum / window_size;
            ''',
            'moving_average'
        )
        
        # RSI kernel
        self.rsi_kernel = cp.ElementwiseKernel(
            'raw T prices, int32 window_size, int32 n',
            'T rsi_values',
            '''
            int idx = i;
            if (idx < window_size) {
                rsi_values = 50.0;  // Neutral RSI
                return;
            }
            
            T gains = 0.0, losses = 0.0;
            for (int j = 1; j <= window_size; j++) {
                T change = prices[idx - j + 1] - prices[idx - j];
                if (change > 0) gains += change;
                else losses -= change;  // Make positive
            }
            
            if (losses == 0.0) {
                rsi_values = 100.0;
            } else {
                T rs = gains / losses;
                rsi_values = 100.0 - (100.0 / (1.0 + rs));
            }
            ''',
            'rsi_calculation'
        )
        
        # Bollinger Bands kernel
        self.bollinger_kernel = cp.ElementwiseKernel(
            'raw T prices, int32 window_size, T std_dev, int32 n',
            'T upper_band, T middle_band, T lower_band',
            '''
            int idx = i;
            if (idx < window_size - 1) {
                upper_band = prices[idx];
                middle_band = prices[idx];
                lower_band = prices[idx];
                return;
            }
            
            // Calculate moving average
            T sum = 0.0;
            for (int j = 0; j < window_size; j++) {
                sum += prices[idx - j];
            }
            T ma = sum / window_size;
            middle_band = ma;
            
            // Calculate standard deviation
            T sum_sq = 0.0;
            for (int j = 0; j < window_size; j++) {
                T diff = prices[idx - j] - ma;
                sum_sq += diff * diff;
            }
            T std = sqrt(sum_sq / window_size);
            
            upper_band = ma + std_dev * std;
            lower_band = ma - std_dev * std;
            ''',
            'bollinger_bands'
        )
        
        self.logger.info("CUDA kernels compiled successfully")
    
    def calculate_moving_averages(self, prices: Union[np.ndarray, cp.ndarray], 
                                 window_sizes: List[int]) -> Dict[int, cp.ndarray]:
        """
        Calculate multiple moving averages in parallel.
        
        Args:
            prices: Price array
            window_sizes: List of window sizes
            
        Returns:
            Dictionary mapping window sizes to moving average arrays
        """
        if not isinstance(prices, cp.ndarray):
            prices = cp.asarray(prices)
        
        results = {}
        
        with self.device:
            for window_size in window_sizes:
                with GPUTimer() as timer:
                    ma_values = cp.zeros_like(prices)
                    self.moving_average_kernel(prices, window_size, len(prices), ma_values)
                    results[window_size] = ma_values
                
                self._update_stats(timer.stop())
        
        return results
    
    def calculate_rsi(self, prices: Union[np.ndarray, cp.ndarray], 
                     window_size: int = 14) -> cp.ndarray:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price array
            window_size: RSI calculation window
            
        Returns:
            RSI values array
        """
        if not isinstance(prices, cp.ndarray):
            prices = cp.asarray(prices)
        
        with self.device:
            with GPUTimer() as timer:
                rsi_values = cp.zeros_like(prices)
                self.rsi_kernel(prices, window_size, len(prices), rsi_values)
                
            self._update_stats(timer.stop())
            return rsi_values
    
    def calculate_bollinger_bands(self, prices: Union[np.ndarray, cp.ndarray],
                                 window_size: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price array
            window_size: Moving average window
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if not isinstance(prices, cp.ndarray):
            prices = cp.asarray(prices)
        
        with self.device:
            with GPUTimer() as timer:
                upper_band = cp.zeros_like(prices)
                middle_band = cp.zeros_like(prices)
                lower_band = cp.zeros_like(prices)
                
                self.bollinger_kernel(prices, window_size, std_dev, len(prices),
                                    upper_band, middle_band, lower_band)
                
            self._update_stats(timer.stop())
            return upper_band, middle_band, lower_band
    
    def calculate_returns(self, prices: Union[np.ndarray, cp.ndarray], 
                         log_returns: bool = True) -> cp.ndarray:
        """
        Calculate price returns (log or simple).
        
        Args:
            prices: Price array
            log_returns: Use log returns if True, simple returns if False
            
        Returns:
            Returns array
        """
        if not isinstance(prices, cp.ndarray):
            prices = cp.asarray(prices)
        
        with self.device:
            with GPUTimer() as timer:
                if log_returns:
                    returns = cp.diff(cp.log(prices))
                else:
                    returns = cp.diff(prices) / prices[:-1]
                
            self._update_stats(timer.stop())
            return returns
    
    def calculate_volatility(self, returns: Union[np.ndarray, cp.ndarray],
                           window_size: int = 252, 
                           annualized: bool = True) -> cp.ndarray:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Returns array
            window_size: Rolling window size
            annualized: Annualize volatility if True
            
        Returns:
            Volatility array
        """
        if not isinstance(returns, cp.ndarray):
            returns = cp.asarray(returns)
        
        with self.device:
            with GPUTimer() as timer:
                # Pad returns for rolling calculation
                padded_returns = cp.pad(returns, (window_size-1, 0), mode='edge')
                
                # Create sliding window view
                volatilities = cp.zeros(len(returns))
                
                for i in range(len(returns)):
                    window_returns = padded_returns[i:i+window_size]
                    vol = cp.std(window_returns)
                    if annualized:
                        vol *= cp.sqrt(252)  # Annualize assuming 252 trading days
                    volatilities[i] = vol
                
            self._update_stats(timer.stop())
            return volatilities
    
    def calculate_correlations(self, returns_matrix: Union[np.ndarray, cp.ndarray]) -> cp.ndarray:
        """
        Calculate correlation matrix for multiple return series.
        
        Args:
            returns_matrix: Matrix where each column is a return series
            
        Returns:
            Correlation matrix
        """
        if not isinstance(returns_matrix, cp.ndarray):
            returns_matrix = cp.asarray(returns_matrix)
        
        with self.device:
            with GPUTimer() as timer:
                # Use CuPy's optimized correlation calculation
                correlation_matrix = cp.corrcoef(returns_matrix.T)
                
            self._update_stats(timer.stop())
            return correlation_matrix
    
    def calculate_portfolio_metrics(self, returns: Union[np.ndarray, cp.ndarray],
                                   weights: Union[np.ndarray, cp.ndarray]) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            returns: Return matrix (time x assets)
            weights: Portfolio weights
            
        Returns:
            Dictionary of portfolio metrics
        """
        if not isinstance(returns, cp.ndarray):
            returns = cp.asarray(returns)
        if not isinstance(weights, cp.ndarray):
            weights = cp.asarray(weights)
        
        with self.device:
            with GPUTimer() as timer:
                # Portfolio returns
                portfolio_returns = cp.dot(returns, weights)
                
                # Calculate metrics
                mean_return = cp.mean(portfolio_returns)
                volatility = cp.std(portfolio_returns)
                sharpe_ratio = mean_return / volatility if volatility > 0 else 0
                
                # Drawdown calculation
                cumulative_returns = cp.cumprod(1 + portfolio_returns)
                running_max = cp.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = cp.min(drawdowns)
                
                # Annualize metrics (assuming daily data)
                annualized_return = mean_return * 252
                annualized_volatility = volatility * cp.sqrt(252)
                annualized_sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
                
            self._update_stats(timer.stop())
            
            return {
                'daily_return': float(mean_return),
                'daily_volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'annualized_return': float(annualized_return),
                'annualized_volatility': float(annualized_volatility),
                'annualized_sharpe': float(annualized_sharpe),
            }
    
    def parallel_strategy_execution(self, price_data: Union[np.ndarray, cp.ndarray],
                                   strategy_params: List[Dict],
                                   strategy_type: str = 'moving_average') -> List[Dict]:
        """
        Execute multiple strategy configurations in parallel.
        
        Args:
            price_data: Price data matrix (time x symbols)
            strategy_params: List of strategy parameter dictionaries
            strategy_type: Type of strategy to execute
            
        Returns:
            List of strategy results
        """
        if not isinstance(price_data, cp.ndarray):
            price_data = cp.asarray(price_data)
        
        results = []
        
        with self.device:
            with GPUTimer() as timer:
                
                if strategy_type == 'moving_average':
                    results = self._parallel_moving_average_strategy(price_data, strategy_params)
                elif strategy_type == 'mean_reversion':
                    results = self._parallel_mean_reversion_strategy(price_data, strategy_params)
                else:
                    raise ValueError(f"Unsupported strategy type: {strategy_type}")
                
            self._update_stats(timer.stop())
        
        return results
    
    def _parallel_moving_average_strategy(self, prices: cp.ndarray, 
                                        params_list: List[Dict]) -> List[Dict]:
        """Execute moving average strategies in parallel."""
        results = []
        
        # Execute each parameter set
        for params in params_list:
            short_window = params.get('short_window', 20)
            long_window = params.get('long_window', 50)
            
            # Calculate moving averages
            mas = self.calculate_moving_averages(prices[:, 0], [short_window, long_window])
            short_ma = mas[short_window]
            long_ma = mas[long_window]
            
            # Generate signals
            signals = cp.zeros_like(short_ma)
            signals[short_ma > long_ma] = 1   # Buy signal
            signals[short_ma < long_ma] = -1  # Sell signal
            
            # Calculate strategy returns
            returns = self.calculate_returns(prices[:, 0])
            strategy_returns = signals[1:] * returns  # Lag signals by 1
            
            # Performance metrics
            total_return = cp.sum(strategy_returns)
            volatility = cp.std(strategy_returns)
            sharpe = total_return / volatility if volatility > 0 else 0
            
            results.append({
                'params': params,
                'total_return': float(total_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe),
                'num_trades': int(cp.sum(cp.abs(cp.diff(signals)) > 0)),
            })
        
        return results
    
    def _parallel_mean_reversion_strategy(self, prices: cp.ndarray,
                                        params_list: List[Dict]) -> List[Dict]:
        """Execute mean reversion strategies in parallel."""
        results = []
        
        for params in params_list:
            window = params.get('window', 20)
            threshold = params.get('threshold', 2.0)
            
            # Calculate z-scores
            ma = self.calculate_moving_averages(prices[:, 0], [window])[window]
            
            # Manual std calculation for rolling window
            rolling_std = cp.zeros_like(prices[:, 0])
            for i in range(window-1, len(prices)):
                window_data = prices[i-window+1:i+1, 0]
                rolling_std[i] = cp.std(window_data)
            
            # Avoid division by zero
            rolling_std = cp.maximum(rolling_std, 1e-8)
            
            z_scores = (prices[:, 0] - ma) / rolling_std
            
            # Generate signals
            signals = cp.zeros_like(z_scores)
            signals[z_scores > threshold] = -1   # Sell when too high
            signals[z_scores < -threshold] = 1   # Buy when too low
            
            # Calculate returns
            returns = self.calculate_returns(prices[:, 0])
            strategy_returns = signals[1:] * returns
            
            # Performance metrics
            total_return = cp.sum(strategy_returns)
            volatility = cp.std(strategy_returns)
            sharpe = total_return / volatility if volatility > 0 else 0
            
            results.append({
                'params': params,
                'total_return': float(total_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe),
                'num_trades': int(cp.sum(cp.abs(cp.diff(signals)) > 0)),
            })
        
        return results
    
    def calculate_var(self, returns: Union[np.ndarray, cp.ndarray],
                     confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR value
        """
        if not isinstance(returns, cp.ndarray):
            returns = cp.asarray(returns)
        
        with self.device:
            with GPUTimer() as timer:
                # Sort returns
                sorted_returns = cp.sort(returns)
                
                # Find percentile
                index = int((1 - confidence_level) * len(sorted_returns))
                var = sorted_returns[index]
                
            self._update_stats(timer.stop())
            return float(var)
    
    def calculate_cvar(self, returns: Union[np.ndarray, cp.ndarray],
                      confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        """
        if not isinstance(returns, cp.ndarray):
            returns = cp.asarray(returns)
        
        with self.device:
            with GPUTimer() as timer:
                # Calculate VaR first
                var = self.calculate_var(returns, confidence_level)
                
                # Calculate CVaR as mean of returns below VaR
                tail_returns = returns[returns <= var]
                cvar = cp.mean(tail_returns) if len(tail_returns) > 0 else var
                
            self._update_stats(timer.stop())
            return float(cvar)
    
    def monte_carlo_simulation(self, initial_price: float, 
                              drift: float, volatility: float,
                              num_steps: int, num_simulations: int,
                              dt: float = 1.0/252) -> cp.ndarray:
        """
        Run Monte Carlo price simulations using geometric Brownian motion.
        
        Args:
            initial_price: Starting price
            drift: Expected return (mu)
            volatility: Volatility (sigma)
            num_steps: Number of time steps
            num_simulations: Number of simulation paths
            dt: Time step size
            
        Returns:
            Price paths array (num_steps x num_simulations)
        """
        with self.device:
            with GPUTimer() as timer:
                # Generate random numbers
                random_shocks = cp.random.normal(0, 1, (num_steps, num_simulations))
                
                # Calculate price increments
                increments = (drift - 0.5 * volatility**2) * dt + \
                           volatility * cp.sqrt(dt) * random_shocks
                
                # Calculate cumulative price paths
                log_prices = cp.cumsum(increments, axis=0)
                log_prices += cp.log(initial_price)
                
                prices = cp.exp(log_prices)
                
            self._update_stats(timer.stop())
            return prices
    
    def _update_stats(self, execution_time_ms: float):
        """Update kernel execution statistics."""
        self.kernel_stats['executions'] += 1
        self.kernel_stats['total_time_ms'] += execution_time_ms
        self.kernel_stats['average_time_ms'] = (
            self.kernel_stats['total_time_ms'] / self.kernel_stats['executions']
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get kernel performance statistics."""
        return {
            'device_id': self.device_id,
            'kernel_stats': self.kernel_stats.copy(),
            'memory_pool_stats': self.memory_pool.get_statistics() if self.memory_pool else None,
            'streams_count': len(self.streams),
        }
    
    def clear_stats(self):
        """Clear performance statistics."""
        self.kernel_stats = {
            'executions': 0,
            'total_time_ms': 0,
            'average_time_ms': 0,
        }
    
    def __del__(self):
        """Cleanup CUDA resources."""
        try:
            if hasattr(self, 'streams'):
                for stream in self.streams:
                    del stream
        except Exception:
            pass
