"""
High-performance order book implementation for HPC QuantSim.

Provides a realistic limit order book with full depth, bid-ask spreads,
and market microstructure simulation for accurate strategy backtesting.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import heapq
import bisect
from collections import defaultdict, deque
import numpy as np
from datetime import datetime


class OrderType(Enum):
    """Order types supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Individual order representation."""
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    order_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    
    def __post_init__(self):
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity
        if self.order_id is None:
            # Generate simple order ID
            import uuid
            self.order_id = str(uuid.uuid4())[:8]
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Trade:
    """Trade execution record."""
    symbol: str
    price: float
    quantity: float
    timestamp: datetime
    buy_order_id: str
    sell_order_id: str
    trade_id: Optional[str] = None
    
    def __post_init__(self):
        if self.trade_id is None:
            import uuid
            self.trade_id = str(uuid.uuid4())[:8]


@dataclass
class BookLevel:
    """Order book price level."""
    price: float
    quantity: float
    order_count: int = 1
    
    def __lt__(self, other):
        return self.price < other.price


class OrderBook:
    """
    High-performance limit order book implementation.
    
    Features:
    - Full depth tracking with price-time priority
    - Efficient order matching using heaps
    - Market microstructure simulation
    - Real-time market data integration
    - Performance optimized for HFT scenarios
    """
    
    def __init__(self, symbol: str, max_depth: int = 1000):
        """Initialize order book for a symbol."""
        self.symbol = symbol
        self.max_depth = max_depth
        
        # Bid side (buy orders) - max heap (highest price first)
        self.bids: List[BookLevel] = []
        self.bid_orders: Dict[str, Order] = {}
        
        # Ask side (sell orders) - min heap (lowest price first)  
        self.asks: List[BookLevel] = []
        self.ask_orders: Dict[str, Order] = {}
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        
        # Market data
        self.last_trade: Optional[Trade] = None
        self.last_price: float = 0.0
        self.volume: float = 0.0
        self.trade_count: int = 0
        
        # Performance tracking
        self.update_count: int = 0
        self.match_count: int = 0
        
        # Market microstructure
        self.spread_bps: float = 0.0
        self.mid_price: float = 0.0
        self.weighted_mid_price: float = 0.0
        
        # Historical data for simulation
        self.price_history: deque = deque(maxlen=1000)
        self.volume_history: deque = deque(maxlen=1000)
    
    def add_order(self, order: Order) -> bool:
        """
        Add order to the book.
        
        Args:
            order: Order to add
            
        Returns:
            bool: True if order was added successfully
        """
        try:
            if order.order_type == OrderType.MARKET:
                return self._execute_market_order(order)
            elif order.order_type == OrderType.LIMIT:
                return self._add_limit_order(order)
            else:
                # Handle stop orders, etc.
                return self._add_conditional_order(order)
        
        except Exception as e:
            order.status = OrderStatus.REJECTED
            return False
    
    def _add_limit_order(self, order: Order) -> bool:
        """Add limit order to appropriate side of book."""
        self.orders[order.order_id] = order
        
        if order.side == OrderSide.BUY:
            # Try to match against asks first
            if self.asks and order.price >= self.asks[0].price:
                return self._match_order(order)
            else:
                # Add to bid side
                self._insert_bid(order)
                
        else:  # SELL
            # Try to match against bids first
            if self.bids and order.price <= self.bids[0].price:
                return self._match_order(order)
            else:
                # Add to ask side
                self._insert_ask(order)
        
        return True
    
    def _execute_market_order(self, order: Order) -> bool:
        """Execute market order immediately."""
        if order.side == OrderSide.BUY:
            return self._match_against_asks(order)
        else:
            return self._match_against_bids(order)
    
    def _match_order(self, order: Order) -> bool:
        """Match order against opposite side of book."""
        if order.side == OrderSide.BUY:
            return self._match_against_asks(order)
        else:
            return self._match_against_bids(order)
    
    def _match_against_asks(self, order: Order) -> bool:
        """Match buy order against ask side."""
        while self.asks and order.remaining_quantity > 0:
            best_ask = self.asks[0]
            
            # Check price compatibility
            if order.order_type == OrderType.LIMIT and order.price < best_ask.price:
                break
            
            # Execute trade
            trade_quantity = min(order.remaining_quantity, best_ask.quantity)
            trade_price = best_ask.price
            
            # Create trade record
            trade = Trade(
                symbol=self.symbol,
                price=trade_price,
                quantity=trade_quantity,
                timestamp=datetime.now(),
                buy_order_id=order.order_id,
                sell_order_id="book_ask"  # Placeholder for ask order ID
            )
            
            # Update order
            order.filled_quantity += trade_quantity
            order.remaining_quantity -= trade_quantity
            
            # Update book level
            best_ask.quantity -= trade_quantity
            if best_ask.quantity <= 0:
                heapq.heappop(self.asks)
            
            # Update market data
            self._update_market_data(trade)
            
            self.match_count += 1
        
        # Update order status
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL_FILL
        
        return order.filled_quantity > 0
    
    def _match_against_bids(self, order: Order) -> bool:
        """Match sell order against bid side."""
        # Convert max heap to handle properly
        while self.bids and order.remaining_quantity > 0:
            best_bid = self.bids[0]
            
            # Check price compatibility  
            if order.order_type == OrderType.LIMIT and order.price > best_bid.price:
                break
            
            # Execute trade
            trade_quantity = min(order.remaining_quantity, best_bid.quantity)
            trade_price = best_bid.price
            
            # Create trade record
            trade = Trade(
                symbol=self.symbol,
                price=trade_price,
                quantity=trade_quantity,
                timestamp=datetime.now(),
                buy_order_id="book_bid",  # Placeholder
                sell_order_id=order.order_id
            )
            
            # Update order
            order.filled_quantity += trade_quantity
            order.remaining_quantity -= trade_quantity
            
            # Update book level
            best_bid.quantity -= trade_quantity
            if best_bid.quantity <= 0:
                # Remove from max heap (negative prices for max heap behavior)
                heapq.heappop(self.bids)
            
            # Update market data
            self._update_market_data(trade)
            
            self.match_count += 1
        
        # Update order status
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL_FILL
        
        return order.filled_quantity > 0
    
    def _insert_bid(self, order: Order) -> None:
        """Insert buy order into bid side."""
        # Use negative price for max heap behavior
        level = BookLevel(price=-order.price, quantity=order.remaining_quantity)
        heapq.heappush(self.bids, level)
        self.bid_orders[order.order_id] = order
    
    def _insert_ask(self, order: Order) -> None:
        """Insert sell order into ask side."""
        level = BookLevel(price=order.price, quantity=order.remaining_quantity)
        heapq.heappush(self.asks, level)
        self.ask_orders[order.order_id] = order
    
    def _add_conditional_order(self, order: Order) -> bool:
        """Handle stop and other conditional orders."""
        # Placeholder for advanced order types
        self.orders[order.order_id] = order
        return True
    
    def _update_market_data(self, trade: Trade) -> None:
        """Update market statistics after trade."""
        self.last_trade = trade
        self.last_price = trade.price
        self.volume += trade.quantity
        self.trade_count += 1
        
        # Update price history
        self.price_history.append(trade.price)
        self.volume_history.append(trade.quantity)
        
        # Update market microstructure metrics
        self._update_microstructure_metrics()
    
    def _update_microstructure_metrics(self) -> None:
        """Update spread, mid-price, and other microstructure metrics."""
        if self.bids and self.asks:
            best_bid = -self.bids[0].price  # Convert back from negative
            best_ask = self.asks[0].price
            
            self.spread_bps = ((best_ask - best_bid) / ((best_ask + best_bid) / 2)) * 10000
            self.mid_price = (best_ask + best_bid) / 2
            
            # Weighted mid price based on quantities
            bid_qty = self.bids[0].quantity
            ask_qty = self.asks[0].quantity
            total_qty = bid_qty + ask_qty
            
            if total_qty > 0:
                self.weighted_mid_price = ((best_bid * ask_qty) + (best_ask * bid_qty)) / total_qty
    
    def update(self, tick_data) -> None:
        """
        Update order book with new market tick data.
        
        Args:
            tick_data: Market tick data (price, volume, etc.)
        """
        try:
            # Simulate order book updates from tick data
            if hasattr(tick_data, 'bid_price') and hasattr(tick_data, 'ask_price'):
                # Update with real bid/ask data
                self._update_from_market_data(tick_data)
            else:
                # Simulate microstructure from trade data
                self._simulate_microstructure(tick_data)
            
            self.update_count += 1
            
        except Exception as e:
            print(f"Error updating order book: {e}")
    
    def _update_from_market_data(self, tick_data) -> None:
        """Update book from real market data."""
        # Clear existing levels
        self.bids.clear()
        self.asks.clear()
        
        # Add bid levels
        if hasattr(tick_data, 'bid_prices'):
            for i, (price, size) in enumerate(zip(tick_data.bid_prices, tick_data.bid_sizes)):
                if i < self.max_depth:
                    level = BookLevel(price=-price, quantity=size)  # Negative for max heap
                    heapq.heappush(self.bids, level)
        
        # Add ask levels
        if hasattr(tick_data, 'ask_prices'):
            for i, (price, size) in enumerate(zip(tick_data.ask_prices, tick_data.ask_sizes)):
                if i < self.max_depth:
                    level = BookLevel(price=price, quantity=size)
                    heapq.heappush(self.asks, level)
        
        # Update microstructure metrics
        self._update_microstructure_metrics()
    
    def _simulate_microstructure(self, tick_data) -> None:
        """Simulate order book microstructure from basic tick data."""
        if not hasattr(tick_data, 'price'):
            return
        
        price = tick_data.price
        volume = getattr(tick_data, 'volume', 100)
        
        # Simulate spread (typically 1-5 basis points for liquid stocks)
        spread_bps = np.random.uniform(1, 5)
        spread = price * (spread_bps / 10000)
        
        # Create synthetic bid/ask levels
        mid_price = price
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        # Clear and rebuild book levels
        self.bids.clear()
        self.asks.clear()
        
        # Add multiple depth levels with decreasing size
        base_size = volume * np.random.uniform(0.5, 2.0)
        
        for i in range(min(10, self.max_depth)):
            # Bid levels (decreasing prices)
            level_bid_price = bid_price - (i * spread * 0.1)
            level_size = base_size * np.exp(-i * 0.3)  # Exponential decay
            
            bid_level = BookLevel(price=-level_bid_price, quantity=level_size)
            heapq.heappush(self.bids, bid_level)
            
            # Ask levels (increasing prices)
            level_ask_price = ask_price + (i * spread * 0.1)
            
            ask_level = BookLevel(price=level_ask_price, quantity=level_size)
            heapq.heappush(self.asks, ask_level)
        
        # Update metrics
        self.last_price = price
        self._update_microstructure_metrics()
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if self.bids:
            return -self.bids[0].price  # Convert back from negative
        return None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if self.asks:
            return self.asks[0].price
        return None
    
    def get_spread(self) -> float:
        """Get current bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return 0.0
    
    def get_mid_price(self) -> float:
        """Get current mid price."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        elif self.last_price > 0:
            return self.last_price
        else:
            return 100.0  # Default price for testing
    
    def get_market_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get market depth (top N levels).
        
        Args:
            levels: Number of levels to return
            
        Returns:
            Dictionary with 'bids' and 'asks' lists of (price, quantity) tuples
        """
        bids = []
        asks = []
        
        # Get bid levels
        bid_heap = self.bids.copy()
        for _ in range(min(levels, len(bid_heap))):
            if bid_heap:
                level = heapq.heappop(bid_heap)
                bids.append((-level.price, level.quantity))  # Convert back from negative
        
        # Get ask levels
        ask_heap = self.asks.copy()
        for _ in range(min(levels, len(ask_heap))):
            if ask_heap:
                level = heapq.heappop(ask_heap)
                asks.append((level.price, level.quantity))
        
        return {'bids': bids, 'asks': asks}
    
    def get_volume_at_price(self, price: float, side: OrderSide) -> float:
        """Get total volume available at specific price level."""
        if side == OrderSide.BUY:
            for level in self.bids:
                if abs(-level.price - price) < 0.01:  # Price tolerance
                    return level.quantity
        else:
            for level in self.asks:
                if abs(level.price - price) < 0.01:
                    return level.quantity
        return 0.0
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.status = OrderStatus.CANCELLED
        
        # Remove from book (simplified - would need more complex logic)
        if order_id in self.bid_orders:
            del self.bid_orders[order_id]
        elif order_id in self.ask_orders:
            del self.ask_orders[order_id]
        
        del self.orders[order_id]
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order book statistics."""
        return {
            'symbol': self.symbol,
            'last_price': self.last_price,
            'mid_price': self.mid_price,
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'spread': self.get_spread(),
            'spread_bps': self.spread_bps,
            'volume': self.volume,
            'trade_count': self.trade_count,
            'update_count': self.update_count,
            'match_count': self.match_count,
            'total_orders': len(self.orders),
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks)
        }
