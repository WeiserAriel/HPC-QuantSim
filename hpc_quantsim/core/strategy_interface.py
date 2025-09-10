"""
Strategy interface and base classes for HPC QuantSim.

Defines the contract that all trading strategies must implement,
along with supporting data structures for strategy execution.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
import numpy as np
from datetime import datetime


class StrategyState(Enum):
    """Strategy execution states."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StrategyResult:
    """Result from strategy tick processing."""
    orders: List['Order'] = None
    signals: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.orders is None:
            self.orders = []
        if self.signals is None:
            self.signals = {}
        if self.metadata is None:
            self.metadata = {}


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement the core methods for initialization,
    tick processing, and cleanup. Strategies can be stateful and maintain
    internal state across ticks.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize strategy with optional name."""
        self.name = name or self.__class__.__name__
        self.state = StrategyState.INITIALIZED
        self.parameters = {}
        self.position = 0.0
        self.cash = 0.0
        self.pnl = 0.0
        
        # Performance tracking
        self.tick_count = 0
        self.trade_count = 0
        self.total_execution_time_ns = 0
        
        # Strategy-specific state
        self.custom_state = {}
    
    @abstractmethod
    def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize strategy with parameters.
        
        Args:
            parameters: Dictionary of strategy-specific parameters
        """
        self.parameters = parameters.copy()
        self.state = StrategyState.RUNNING
    
    @abstractmethod
    def on_tick(self, tick_data, order_book, position: float, cash: float) -> StrategyResult:
        """
        Process a new market tick and generate trading signals.
        
        Args:
            tick_data: Current market tick data
            order_book: Current order book state
            position: Current position size
            cash: Available cash
            
        Returns:
            StrategyResult containing orders and signals
        """
        pass
    
    def on_trade_fill(self, trade_fill) -> None:
        """
        Handle trade fill notification.
        
        Args:
            trade_fill: Trade fill information
        """
        self.trade_count += 1
    
    def on_market_open(self) -> None:
        """Called when market opens."""
        pass
    
    def on_market_close(self) -> None:
        """Called when market closes."""
        pass
    
    def cleanup(self) -> None:
        """Clean up strategy resources."""
        self.state = StrategyState.STOPPED
    
    def get_state(self) -> Dict[str, Any]:
        """Get current strategy state for serialization."""
        return {
            'name': self.name,
            'state': self.state.value,
            'parameters': self.parameters,
            'position': self.position,
            'cash': self.cash,
            'pnl': self.pnl,
            'tick_count': self.tick_count,
            'trade_count': self.trade_count,
            'custom_state': self.custom_state
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load strategy state from serialized data."""
        self.name = state.get('name', self.name)
        self.state = StrategyState(state.get('state', StrategyState.INITIALIZED.value))
        self.parameters = state.get('parameters', {})
        self.position = state.get('position', 0.0)
        self.cash = state.get('cash', 0.0)
        self.pnl = state.get('pnl', 0.0)
        self.tick_count = state.get('tick_count', 0)
        self.trade_count = state.get('trade_count', 0)
        self.custom_state = state.get('custom_state', {})


class StatelessStrategy(Strategy):
    """
    Base class for stateless strategies.
    
    Stateless strategies don't maintain internal state between ticks,
    making them easier to parallelize and test.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
    
    def get_state(self) -> Dict[str, Any]:
        """Stateless strategies have minimal state."""
        return {
            'name': self.name,
            'parameters': self.parameters
        }


class MovingAverageStrategy(Strategy):
    """
    Example moving average crossover strategy.
    
    This is a simple momentum strategy that generates buy/sell signals
    based on short-term and long-term moving average crossovers.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.short_window = 20
        self.long_window = 50
        self.price_history = []
        self.short_ma = 0.0
        self.long_ma = 0.0
        self.last_signal = 0  # -1: sell, 0: hold, 1: buy
    
    def initialize(self, parameters: Dict[str, Any]) -> None:
        """Initialize with MA parameters."""
        super().initialize(parameters)
        
        self.short_window = parameters.get('short_window', 20)
        self.long_window = parameters.get('long_window', 50)
        self.price_history = []
        
        # Validate parameters
        if self.short_window >= self.long_window:
            raise ValueError("Short window must be smaller than long window")
    
    def on_tick(self, tick_data, order_book, position: float, cash: float) -> StrategyResult:
        """Generate signals based on moving average crossover."""
        import time
        start_time = time.time_ns()
        
        try:
            # Get current price
            current_price = order_book.get_mid_price()
            
            # Update price history
            self.price_history.append(current_price)
            
            # Keep only necessary history
            max_history = max(self.long_window, 100)
            if len(self.price_history) > max_history:
                self.price_history = self.price_history[-max_history:]
            
            # Calculate moving averages
            if len(self.price_history) >= self.short_window:
                self.short_ma = np.mean(self.price_history[-self.short_window:])
            
            if len(self.price_history) >= self.long_window:
                self.long_ma = np.mean(self.price_history[-self.long_window:])
            
            # Generate signals only if we have enough data
            result = StrategyResult()
            
            if len(self.price_history) >= self.long_window:
                # Determine signal
                if self.short_ma > self.long_ma and self.last_signal != 1:
                    # Golden cross - buy signal
                    if position <= 0:  # Only buy if not long
                        from .order_book import Order, OrderType, OrderSide
                        
                        order_size = min(cash * 0.1 / current_price, 100)  # 10% of cash or 100 shares
                        if order_size > 0:
                            buy_order = Order(
                                symbol=tick_data.symbol,
                                order_type=OrderType.MARKET,
                                side=OrderSide.BUY,
                                quantity=order_size,
                                price=current_price
                            )
                            result.orders.append(buy_order)
                            self.last_signal = 1
                
                elif self.short_ma < self.long_ma and self.last_signal != -1:
                    # Death cross - sell signal
                    if position > 0:  # Only sell if long
                        from .order_book import Order, OrderType, OrderSide
                        
                        sell_order = Order(
                            symbol=tick_data.symbol,
                            order_type=OrderType.MARKET,
                            side=OrderSide.SELL,
                            quantity=position,
                            price=current_price
                        )
                        result.orders.append(sell_order)
                        self.last_signal = -1
            
            # Add signals to result
            result.signals = {
                'short_ma': self.short_ma,
                'long_ma': self.long_ma,
                'signal': self.last_signal,
                'price': current_price
            }
            
            result.metadata = {
                'strategy_name': self.name,
                'data_points': len(self.price_history),
                'ready': len(self.price_history) >= self.long_window
            }
            
            self.tick_count += 1
            return result
            
        except Exception as e:
            self.state = StrategyState.ERROR
            result = StrategyResult()
            result.metadata = {'error': str(e)}
            return result
        
        finally:
            self.total_execution_time_ns += time.time_ns() - start_time


class MeanReversionStrategy(Strategy):
    """
    Example mean reversion strategy.
    
    Trades based on price deviation from a moving average,
    betting that prices will revert to the mean.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.lookback_window = 50
        self.entry_threshold = 2.0  # Standard deviations
        self.exit_threshold = 0.5
        self.price_history = []
        self.current_mean = 0.0
        self.current_std = 0.0
        self.entry_price = 0.0
        
    def initialize(self, parameters: Dict[str, Any]) -> None:
        """Initialize mean reversion parameters."""
        super().initialize(parameters)
        
        self.lookback_window = parameters.get('lookback_window', 50)
        self.entry_threshold = parameters.get('entry_threshold', 2.0)
        self.exit_threshold = parameters.get('exit_threshold', 0.5)
        
    def on_tick(self, tick_data, order_book, position: float, cash: float) -> StrategyResult:
        """Generate mean reversion signals."""
        import time
        start_time = time.time_ns()
        
        try:
            current_price = order_book.get_mid_price()
            self.price_history.append(current_price)
            
            # Keep rolling window
            if len(self.price_history) > self.lookback_window:
                self.price_history = self.price_history[-self.lookback_window:]
            
            result = StrategyResult()
            
            if len(self.price_history) >= self.lookback_window:
                # Calculate statistics
                prices = np.array(self.price_history)
                self.current_mean = np.mean(prices)
                self.current_std = np.std(prices)
                
                if self.current_std > 0:
                    # Calculate z-score
                    z_score = (current_price - self.current_mean) / self.current_std
                    
                    from .order_book import Order, OrderType, OrderSide
                    
                    # Entry logic
                    if abs(position) < 0.1:  # No position
                        if z_score > self.entry_threshold:
                            # Price too high, short
                            order_size = min(cash * 0.05 / current_price, 50)
                            if order_size > 0:
                                short_order = Order(
                                    symbol=tick_data.symbol,
                                    order_type=OrderType.MARKET,
                                    side=OrderSide.SELL,
                                    quantity=order_size,
                                    price=current_price
                                )
                                result.orders.append(short_order)
                                self.entry_price = current_price
                        
                        elif z_score < -self.entry_threshold:
                            # Price too low, long
                            order_size = min(cash * 0.05 / current_price, 50)
                            if order_size > 0:
                                long_order = Order(
                                    symbol=tick_data.symbol,
                                    order_type=OrderType.MARKET,
                                    side=OrderSide.BUY,
                                    quantity=order_size,
                                    price=current_price
                                )
                                result.orders.append(long_order)
                                self.entry_price = current_price
                    
                    # Exit logic
                    elif abs(z_score) < self.exit_threshold:
                        # Close position when price reverts to mean
                        if position > 0:
                            close_order = Order(
                                symbol=tick_data.symbol,
                                order_type=OrderType.MARKET,
                                side=OrderSide.SELL,
                                quantity=position,
                                price=current_price
                            )
                            result.orders.append(close_order)
                        elif position < 0:
                            close_order = Order(
                                symbol=tick_data.symbol,
                                order_type=OrderType.MARKET,
                                side=OrderSide.BUY,
                                quantity=abs(position),
                                price=current_price
                            )
                            result.orders.append(close_order)
                    
                    # Store signals
                    result.signals = {
                        'z_score': z_score,
                        'mean': self.current_mean,
                        'std': self.current_std,
                        'price': current_price,
                        'entry_price': self.entry_price
                    }
            
            result.metadata = {
                'strategy_name': self.name,
                'data_points': len(self.price_history)
            }
            
            self.tick_count += 1
            return result
            
        except Exception as e:
            self.state = StrategyState.ERROR
            result = StrategyResult()
            result.metadata = {'error': str(e)}
            return result
        
        finally:
            self.total_execution_time_ns += time.time_ns() - start_time
