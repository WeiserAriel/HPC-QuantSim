"""
Tick data structures and market data containers for HPC QuantSim.

Defines the core data structures for representing market ticks,
quotes, trades, and order book snapshots.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
import numpy as np
from datetime import datetime, timezone
import pandas as pd


class TickType(Enum):
    """Types of market ticks."""
    TRADE = "trade"
    QUOTE = "quote" 
    DEPTH = "depth"
    ORDER = "order"
    SNAPSHOT = "snapshot"


class MarketCondition(Enum):
    """Market condition indicators."""
    NORMAL = "normal"
    FAST_MARKET = "fast_market"
    SLOW_MARKET = "slow_market"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    HALTED = "halted"


@dataclass
class TickData:
    """
    Basic tick data structure.
    
    Represents a single market event (trade, quote update, etc.)
    with timestamp and symbol information.
    """
    symbol: str
    timestamp: datetime
    tick_type: TickType
    price: float = 0.0
    volume: float = 0.0
    sequence_number: Optional[int] = None
    exchange: Optional[str] = None
    condition: MarketCondition = MarketCondition.NORMAL
    
    def __post_init__(self):
        # Ensure timestamp is timezone-aware
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class TradeData(TickData):
    """Trade tick data with execution details."""
    trade_id: Optional[str] = None
    buyer_initiated: Optional[bool] = None  # True if buyer initiated
    tick_type: TickType = TickType.TRADE
    
    def __post_init__(self):
        super().__post_init__()
        if self.trade_id is None:
            # Generate simple trade ID
            self.trade_id = f"{self.symbol}_{int(self.timestamp.timestamp() * 1000)}"


@dataclass  
class QuoteData(TickData):
    """Quote tick data with bid/ask information."""
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    tick_type: TickType = TickType.QUOTE
    
    def __post_init__(self):
        super().__post_init__()
        # Set price to mid-price if not provided
        if self.price == 0.0 and self.bid_price > 0 and self.ask_price > 0:
            self.price = (self.bid_price + self.ask_price) / 2


@dataclass
class DepthData(TickData):
    """Market depth data with multiple price levels."""
    bid_prices: Optional[List[float]] = None
    ask_prices: Optional[List[float]] = None  
    bid_sizes: Optional[List[float]] = None
    ask_sizes: Optional[List[float]] = None
    levels: int = 10
    tick_type: TickType = TickType.DEPTH
    
    def __post_init__(self):
        super().__post_init__()
        if self.bid_prices is None:
            self.bid_prices = []
        if self.ask_prices is None:
            self.ask_prices = []
        if self.bid_sizes is None:
            self.bid_sizes = []
        if self.ask_sizes is None:
            self.ask_sizes = []
        
        # Set price to best bid/ask midpoint
        if (self.price == 0.0 and self.bid_prices and self.ask_prices and 
            len(self.bid_prices) > 0 and len(self.ask_prices) > 0):
            self.price = (self.bid_prices[0] + self.ask_prices[0]) / 2


@dataclass
class OrderData(TickData):
    """Order book event data."""
    tick_type: TickType = TickType.ORDER
    order_id: str = ""
    side: str = ""  # "buy" or "sell"
    action: str = ""  # "add", "modify", "cancel"
    order_type: str = "limit"
    
    def __post_init__(self):
        super().__post_init__()


class MarketData:
    """
    Container for market data with efficient storage and access.
    
    Provides methods for storing, querying, and iterating over
    large volumes of tick data with optimized memory usage.
    """
    
    def __init__(self, symbol: str, start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None):
        """Initialize market data container."""
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        
        # Data storage
        self.ticks: List[TickData] = []
        self.trades: List[TradeData] = []
        self.quotes: List[QuoteData] = []
        self.depth: List[DepthData] = []
        self.orders: List[OrderData] = []
        
        # Indices for fast lookup
        self._time_index: Dict[datetime, List[int]] = {}
        self._sequence_index: Dict[int, TickData] = {}
        
        # Statistics
        self.total_ticks = 0
        self.total_volume = 0.0
        self.price_range = (float('inf'), float('-inf'))
        
        # Caching for performance
        self._sorted_timestamps = None
        self._price_array = None
        self._volume_array = None
    
    def add_tick(self, tick: TickData) -> None:
        """Add a tick to the market data."""
        # Validate time range
        if self.start_time and tick.timestamp < self.start_time:
            return
        if self.end_time and tick.timestamp > self.end_time:
            return
        
        # Add to appropriate collection
        self.ticks.append(tick)
        
        if isinstance(tick, TradeData):
            self.trades.append(tick)
        elif isinstance(tick, QuoteData):
            self.quotes.append(tick)
        elif isinstance(tick, DepthData):
            self.depth.append(tick)
        elif isinstance(tick, OrderData):
            self.orders.append(tick)
        
        # Update indices
        if tick.timestamp not in self._time_index:
            self._time_index[tick.timestamp] = []
        self._time_index[tick.timestamp].append(len(self.ticks) - 1)
        
        if tick.sequence_number is not None:
            self._sequence_index[tick.sequence_number] = tick
        
        # Update statistics
        self.total_ticks += 1
        self.total_volume += tick.volume
        
        if tick.price > 0:
            self.price_range = (
                min(self.price_range[0], tick.price),
                max(self.price_range[1], tick.price)
            )
        
        # Invalidate caches
        self._sorted_timestamps = None
        self._price_array = None
        self._volume_array = None
    
    def get_ticks_in_range(self, start_time: datetime, 
                          end_time: datetime) -> List[TickData]:
        """Get all ticks within time range."""
        result = []
        for tick in self.ticks:
            if start_time <= tick.timestamp <= end_time:
                result.append(tick)
        return result
    
    def get_ticks_by_type(self, tick_type: TickType) -> List[TickData]:
        """Get all ticks of specific type."""
        return [tick for tick in self.ticks if tick.tick_type == tick_type]
    
    def get_trades_in_range(self, start_time: datetime,
                           end_time: datetime) -> List[TradeData]:
        """Get trades within time range."""
        return [trade for trade in self.trades 
                if start_time <= trade.timestamp <= end_time]
    
    def get_quotes_in_range(self, start_time: datetime,
                           end_time: datetime) -> List[QuoteData]:
        """Get quotes within time range."""
        return [quote for quote in self.quotes
                if start_time <= quote.timestamp <= end_time]
    
    def get_ohlcv_bars(self, bar_size_ms: int = 1000) -> pd.DataFrame:
        """
        Generate OHLCV bars from tick data.
        
        Args:
            bar_size_ms: Bar size in milliseconds
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.trades:
            return pd.DataFrame()
        
        # Convert trades to DataFrame
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'timestamp': trade.timestamp,
                'price': trade.price,
                'volume': trade.volume
            })
        
        df = pd.DataFrame(trade_data)
        df = df.set_index('timestamp')
        
        # Resample to create bars
        bars = df.resample(f'{bar_size_ms}ms').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        }).dropna()
        
        # Flatten column names
        bars.columns = ['open', 'high', 'low', 'close', 'volume']
        
        return bars
    
    def get_vwap(self, start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None) -> float:
        """Calculate Volume Weighted Average Price (VWAP)."""
        trades = self.trades
        
        if start_time or end_time:
            start_time = start_time or datetime.min.replace(tzinfo=timezone.utc)
            end_time = end_time or datetime.max.replace(tzinfo=timezone.utc)
            trades = [t for t in trades if start_time <= t.timestamp <= end_time]
        
        if not trades:
            return 0.0
        
        total_value = sum(t.price * t.volume for t in trades)
        total_volume = sum(t.volume for t in trades)
        
        return total_value / total_volume if total_volume > 0 else 0.0
    
    def get_price_array(self) -> np.ndarray:
        """Get numpy array of all prices for efficient computation."""
        if self._price_array is None:
            prices = [tick.price for tick in self.ticks if tick.price > 0]
            self._price_array = np.array(prices)
        return self._price_array
    
    def get_volume_array(self) -> np.ndarray:
        """Get numpy array of all volumes."""
        if self._volume_array is None:
            volumes = [tick.volume for tick in self.ticks]
            self._volume_array = np.array(volumes)
        return self._volume_array
    
    def get_returns(self) -> np.ndarray:
        """Calculate price returns."""
        prices = self.get_price_array()
        if len(prices) < 2:
            return np.array([])
        
        returns = np.diff(np.log(prices))
        return returns
    
    def get_volatility(self, window_ms: int = 60000) -> float:
        """
        Calculate realized volatility.
        
        Args:
            window_ms: Window size in milliseconds
            
        Returns:
            Annualized volatility
        """
        returns = self.get_returns()
        if len(returns) == 0:
            return 0.0
        
        # Calculate volatility (assuming 252 trading days, 6.5 hours per day)
        volatility = np.std(returns) * np.sqrt(252 * 6.5 * 3600 * 1000 / window_ms)
        return volatility
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive market data statistics."""
        if not self.ticks:
            return {}
        
        prices = self.get_price_array()
        volumes = self.get_volume_array()
        returns = self.get_returns()
        
        return {
            'symbol': self.symbol,
            'total_ticks': self.total_ticks,
            'total_trades': len(self.trades),
            'total_quotes': len(self.quotes),
            'total_volume': self.total_volume,
            'price_statistics': {
                'min': float(np.min(prices)) if len(prices) > 0 else 0,
                'max': float(np.max(prices)) if len(prices) > 0 else 0,
                'mean': float(np.mean(prices)) if len(prices) > 0 else 0,
                'std': float(np.std(prices)) if len(prices) > 0 else 0,
            },
            'volume_statistics': {
                'min': float(np.min(volumes)) if len(volumes) > 0 else 0,
                'max': float(np.max(volumes)) if len(volumes) > 0 else 0,
                'mean': float(np.mean(volumes)) if len(volumes) > 0 else 0,
                'total': self.total_volume,
            },
            'return_statistics': {
                'mean': float(np.mean(returns)) if len(returns) > 0 else 0,
                'std': float(np.std(returns)) if len(returns) > 0 else 0,
                'skewness': float(self._calculate_skewness(returns)) if len(returns) > 0 else 0,
                'kurtosis': float(self._calculate_kurtosis(returns)) if len(returns) > 0 else 0,
            },
            'time_range': {
                'start': self.ticks[0].timestamp.isoformat() if self.ticks else None,
                'end': self.ticks[-1].timestamp.isoformat() if self.ticks else None,
                'duration_ms': int((self.ticks[-1].timestamp - self.ticks[0].timestamp).total_seconds() * 1000) if len(self.ticks) > 1 else 0,
            },
            'vwap': self.get_vwap(),
            'volatility': self.get_volatility(),
        }
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((returns - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) < 4:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((returns - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurtosis
    
    def sort_by_time(self) -> None:
        """Sort all ticks by timestamp."""
        self.ticks.sort(key=lambda x: x.timestamp)
        self.trades.sort(key=lambda x: x.timestamp)
        self.quotes.sort(key=lambda x: x.timestamp)
        self.depth.sort(key=lambda x: x.timestamp)
        self.orders.sort(key=lambda x: x.timestamp)
        
        # Rebuild time index
        self._time_index.clear()
        for i, tick in enumerate(self.ticks):
            if tick.timestamp not in self._time_index:
                self._time_index[tick.timestamp] = []
            self._time_index[tick.timestamp].append(i)
    
    def __len__(self) -> int:
        """Return number of ticks."""
        return len(self.ticks)
    
    def __iter__(self):
        """Iterate over ticks."""
        return iter(self.ticks)
