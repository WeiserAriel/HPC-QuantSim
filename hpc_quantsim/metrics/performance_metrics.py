"""
Performance metrics collection and calculation for HPC QuantSim.

Provides comprehensive performance analysis including:
- PnL tracking and calculation
- Sharpe, Sortino, and other risk-adjusted returns
- Trade-level analysis
- Position and exposure metrics
- Transaction cost analysis
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import math


@dataclass
class TradeMetrics:
    """Individual trade performance metrics."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    commission: float = 0.0
    market_impact: float = 0.0
    
    # Calculated fields
    notional: float = field(init=False)
    
    def __post_init__(self):
        self.notional = abs(self.quantity * self.price)


@dataclass
class PositionMetrics:
    """Position-level metrics."""
    symbol: str
    timestamp: datetime
    quantity: float
    average_price: float
    market_price: float
    
    # Calculated fields
    market_value: float = field(init=False)
    unrealized_pnl: float = field(init=False)
    
    def __post_init__(self):
        self.market_value = self.quantity * self.market_price
        self.unrealized_pnl = self.quantity * (self.market_price - self.average_price)


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator.
    
    Tracks and calculates key performance indicators including:
    - Profit and Loss (PnL)
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Trade statistics
    - Position and exposure metrics
    """
    
    def __init__(self, initial_capital: float = 1000000.0, benchmark_rate: float = 0.02):
        """Initialize performance tracker."""
        self.initial_capital = initial_capital
        self.benchmark_rate = benchmark_rate  # Risk-free rate
        
        # Core metrics
        self.current_capital = initial_capital
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Time series data
        self.pnl_history: List[Tuple[datetime, float]] = []
        self.returns_history: List[float] = []
        self.drawdown_history: List[float] = []
        
        # Trade tracking
        self.trades: List[TradeMetrics] = []
        self.positions: Dict[str, PositionMetrics] = {}
        
        # Performance statistics
        self.max_drawdown = 0.0
        self.max_drawdown_duration = timedelta()
        self.peak_capital = initial_capital
        self.last_peak_time: Optional[datetime] = None
        
        # Win/loss statistics
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        
        # Rolling window for calculations
        self.window_size = 252  # 1 year of daily data
        self.rolling_returns = deque(maxlen=self.window_size)
        self.rolling_pnl = deque(maxlen=self.window_size)
        
        # Transaction costs
        self.total_commission = 0.0
        self.total_market_impact = 0.0
        
        # Cache for expensive calculations
        self._cached_metrics: Dict[str, Any] = {}
        self._last_calculation_time: Optional[datetime] = None
    
    def record_trade(self, timestamp: datetime, symbol: str, side: str, 
                    quantity: float, price: float, trade_id: Optional[str] = None,
                    commission: float = 0.0, market_impact: float = 0.0) -> None:
        """Record a trade execution."""
        if trade_id is None:
            trade_id = f"{symbol}_{int(timestamp.timestamp())}"
        
        trade = TradeMetrics(
            trade_id=trade_id,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            market_impact=market_impact
        )
        
        self.trades.append(trade)
        self.total_trades += 1
        self.total_commission += commission
        self.total_market_impact += market_impact
        
        # Update position
        self._update_position(trade)
        
        # Update PnL
        trade_pnl = self._calculate_trade_pnl(trade)
        self.realized_pnl += trade_pnl
        
        # Update win/loss statistics
        if trade_pnl > 0:
            self.winning_trades += 1
        elif trade_pnl < 0:
            self.losing_trades += 1
        
        # Clear cache
        self._cached_metrics.clear()
    
    def update_pnl(self, total_pnl: float, timestamp: datetime) -> None:
        """Update total PnL and derived metrics."""
        self.total_pnl = total_pnl
        self.current_capital = self.initial_capital + total_pnl
        
        # Record PnL history
        self.pnl_history.append((timestamp, total_pnl))
        
        # Calculate return
        if len(self.pnl_history) > 1:
            prev_pnl = self.pnl_history[-2][1]
            prev_capital = self.initial_capital + prev_pnl
            
            if prev_capital > 0:
                period_return = (total_pnl - prev_pnl) / prev_capital
                self.returns_history.append(period_return)
                self.rolling_returns.append(period_return)
        
        # Update drawdown
        self._update_drawdown(timestamp)
        
        # Update rolling PnL
        self.rolling_pnl.append(total_pnl)
        
        # Clear cache
        self._cached_metrics.clear()
    
    def _update_position(self, trade: TradeMetrics) -> None:
        """Update position from trade."""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = PositionMetrics(
                symbol=symbol,
                timestamp=trade.timestamp,
                quantity=trade.quantity if trade.side == 'buy' else -trade.quantity,
                average_price=trade.price,
                market_price=trade.price
            )
        else:
            # Update existing position
            pos = self.positions[symbol]
            old_quantity = pos.quantity
            old_avg_price = pos.average_price
            
            # Calculate new quantity
            if trade.side == 'buy':
                new_quantity = old_quantity + trade.quantity
            else:
                new_quantity = old_quantity - trade.quantity
            
            # Calculate new average price (if increasing position)
            if (old_quantity >= 0 and trade.side == 'buy') or (old_quantity <= 0 and trade.side == 'sell'):
                # Adding to position
                total_cost = old_quantity * old_avg_price + trade.quantity * trade.price
                new_avg_price = total_cost / new_quantity if new_quantity != 0 else 0
            else:
                # Reducing position - keep old average price
                new_avg_price = old_avg_price
            
            # Update position
            pos.quantity = new_quantity
            pos.average_price = new_avg_price
            pos.timestamp = trade.timestamp
            pos.market_price = trade.price
            
            # Remove position if flat
            if abs(new_quantity) < 1e-6:
                del self.positions[symbol]
    
    def _calculate_trade_pnl(self, trade: TradeMetrics) -> float:
        """Calculate PnL impact of a trade."""
        # For now, simple calculation - would be more complex with position tracking
        # This is placeholder - actual PnL calculation depends on strategy
        return -trade.commission - trade.market_impact
    
    def _update_drawdown(self, timestamp: datetime) -> None:
        """Update drawdown statistics."""
        current_capital = self.current_capital
        
        # Update peak
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
            self.last_peak_time = timestamp
        
        # Calculate current drawdown
        if self.peak_capital > 0:
            current_drawdown = (current_capital - self.peak_capital) / self.peak_capital
            self.drawdown_history.append(current_drawdown)
            
            # Update max drawdown
            if current_drawdown < self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Update drawdown duration
            if self.last_peak_time and current_drawdown < 0:
                current_duration = timestamp - self.last_peak_time
                if current_duration > self.max_drawdown_duration:
                    self.max_drawdown_duration = current_duration
    
    def get_sharpe_ratio(self, periods_per_year: int = 252) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history[-self.window_size:])
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.benchmark_rate / periods_per_year)
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
        return float(sharpe)
    
    def get_sortino_ratio(self, periods_per_year: int = 252) -> float:
        """Calculate annualized Sortino ratio."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history[-self.window_size:])
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.benchmark_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(periods_per_year)
        return float(sortino)
    
    def get_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if self.max_drawdown >= 0 or len(self.pnl_history) < 2:
            return 0.0
        
        # Calculate annualized return
        if len(self.pnl_history) < 252:
            return 0.0
        
        total_days = len(self.pnl_history)
        total_return = (self.current_capital / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / total_days) - 1
        
        calmar = annualized_return / abs(self.max_drawdown)
        return float(calmar)
    
    def get_information_ratio(self) -> float:
        """Calculate information ratio (alpha / tracking error)."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history[-self.window_size:])
        benchmark_returns = np.full_like(returns, self.benchmark_rate / 252)
        
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        information_ratio = np.mean(excess_returns) / tracking_error
        return float(information_ratio)
    
    def get_win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def get_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not self.trades:
            return 0.0
        
        gross_profit = sum(abs(t.notional) for t in self.trades if t.side == 'sell')  # Simplified
        gross_loss = sum(abs(t.notional) for t in self.trades if t.side == 'buy')    # Simplified
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_average_trade_pnl(self) -> float:
        """Calculate average PnL per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.realized_pnl / self.total_trades
    
    def get_trade_frequency(self) -> float:
        """Calculate trades per day."""
        if len(self.trades) < 2:
            return 0.0
        
        first_trade_time = self.trades[0].timestamp
        last_trade_time = self.trades[-1].timestamp
        
        duration_days = (last_trade_time - first_trade_time).total_seconds() / 86400
        
        if duration_days == 0:
            return 0.0
        
        return len(self.trades) / duration_days
    
    def get_volatility(self, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history[-self.window_size:])
        volatility = np.std(returns) * np.sqrt(periods_per_year)
        return float(volatility)
    
    def get_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(self.returns_history) < 10:
            return 0.0
        
        returns = np.array(self.returns_history[-self.window_size:])
        var = np.percentile(returns, (1 - confidence) * 100)
        return float(var * self.current_capital)  # Convert to dollar amount
    
    def get_cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(self.returns_history) < 10:
            return 0.0
        
        returns = np.array(self.returns_history[-self.window_size:])
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
        
        cvar = np.mean(tail_losses)
        return float(cvar * self.current_capital)  # Convert to dollar amount
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if self._cached_metrics and self._last_calculation_time:
            # Return cached results if recent
            cache_age = datetime.now() - self._last_calculation_time
            if cache_age.total_seconds() < 60:  # Cache for 1 minute
                return self._cached_metrics
        
        summary = {
            # Core metrics
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'current_capital': self.current_capital,
            'total_return_pct': ((self.current_capital / self.initial_capital) - 1) * 100,
            
            # Risk-adjusted returns
            'sharpe_ratio': self.get_sharpe_ratio(),
            'sortino_ratio': self.get_sortino_ratio(),
            'calmar_ratio': self.get_calmar_ratio(),
            'information_ratio': self.get_information_ratio(),
            
            # Risk metrics
            'max_drawdown_pct': self.max_drawdown * 100,
            'max_drawdown_duration_days': self.max_drawdown_duration.total_seconds() / 86400,
            'volatility_pct': self.get_volatility() * 100,
            'var_95_pct': self.get_var() / self.current_capital * 100,
            'cvar_95_pct': self.get_cvar() / self.current_capital * 100,
            
            # Trade statistics
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': self.get_win_rate(),
            'profit_factor': self.get_profit_factor(),
            'avg_trade_pnl': self.get_average_trade_pnl(),
            'trade_frequency_per_day': self.get_trade_frequency(),
            
            # Cost analysis
            'total_commission': self.total_commission,
            'total_market_impact': self.total_market_impact,
            'total_transaction_costs': self.total_commission + self.total_market_impact,
            'cost_ratio_pct': ((self.total_commission + self.total_market_impact) / self.initial_capital) * 100,
            
            # Position metrics
            'active_positions': len(self.positions),
            'position_symbols': list(self.positions.keys()) if self.positions else [],
            
            # Time series stats
            'data_points': len(self.pnl_history),
            'returns_count': len(self.returns_history),
            'start_time': self.pnl_history[0][0].isoformat() if self.pnl_history else None,
            'end_time': self.pnl_history[-1][0].isoformat() if self.pnl_history else None,
        }
        
        # Cache results
        self._cached_metrics = summary
        self._last_calculation_time = datetime.now()
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert PnL history to DataFrame."""
        if not self.pnl_history:
            return pd.DataFrame()
        
        data = {
            'timestamp': [t[0] for t in self.pnl_history],
            'pnl': [t[1] for t in self.pnl_history],
            'capital': [self.initial_capital + t[1] for t in self.pnl_history],
            'drawdown': self.drawdown_history + [0] * (len(self.pnl_history) - len(self.drawdown_history))
        }
        
        df = pd.DataFrame(data)
        
        # Add returns if available
        if self.returns_history:
            returns = [0] + self.returns_history  # Pad with 0 for first period
            if len(returns) < len(df):
                returns.extend([0] * (len(df) - len(returns)))
            df['returns'] = returns[:len(df)]
        
        return df
    
    def finalize(self) -> None:
        """Finalize metrics calculation (called at end of simulation)."""
        # Calculate any final statistics
        if self.positions:
            # Calculate final unrealized PnL
            self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # Clear cache to force fresh calculation
        self._cached_metrics.clear()
    
    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio property for backward compatibility."""
        return self.get_sharpe_ratio()
    
    @property 
    def max_drawdown_pct(self) -> float:
        """Max drawdown percentage property."""
        return self.max_drawdown * 100
