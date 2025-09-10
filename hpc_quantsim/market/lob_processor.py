"""
Limit Order Book (LOB) processor for HPC QuantSim.

Reconstructs and maintains order book state from market data feeds:
- L1 (BBO) reconstruction from trades and quotes
- L2 (depth) reconstruction from order book snapshots  
- L3 (full depth) from order-by-order feeds
- Market microstructure analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from .tick_data import TickData, TradeData, QuoteData, DepthData, OrderData, TickType
from ..core.order_book import OrderBook, Order, OrderType, OrderSide


@dataclass
class OrderBookSnapshot:
    """Order book snapshot at a point in time."""
    timestamp: datetime
    symbol: str
    bid_levels: List[Tuple[float, float]]  # (price, size) pairs
    ask_levels: List[Tuple[float, float]]  # (price, size) pairs
    last_trade_price: Optional[float] = None
    last_trade_volume: Optional[float] = None
    sequence_number: Optional[int] = None
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self.bid_levels[0][0] if self.bid_levels else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self.ask_levels[0][0] if self.ask_levels else None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Get spread in basis points."""
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price) * 10000
        return None


class LOBProcessor:
    """
    Limit Order Book processor for market microstructure reconstruction.
    
    Features:
    - Real-time order book maintenance
    - Historical reconstruction from various data types
    - Market impact analysis
    - Liquidity metrics calculation
    - Cross-validation of different data sources
    """
    
    def __init__(self, symbol: str, max_levels: int = 10):
        """Initialize LOB processor."""
        self.symbol = symbol
        self.max_levels = max_levels
        
        # Current order book state
        self.current_snapshot: Optional[OrderBookSnapshot] = None
        self.order_book = OrderBook(symbol, max_depth=max_levels * 2)
        
        # Historical snapshots
        self.snapshots: List[OrderBookSnapshot] = []
        self.max_snapshots = 10000  # Keep last N snapshots
        
        # Order tracking for L3 reconstruction
        self.active_orders: Dict[str, Order] = {}
        self.order_sequence = 0
        
        # Market microstructure metrics
        self.trade_count = 0
        self.total_volume = 0.0
        self.volume_weighted_price = 0.0
        
        # Price level analysis
        self.price_levels = defaultdict(float)  # price -> cumulative volume
        self.level_updates = deque(maxlen=1000)  # Recent level changes
        
        # Statistics
        self.stats = {
            'updates_processed': 0,
            'snapshots_created': 0,
            'reconstruction_errors': 0,
            'last_update_time': None
        }
    
    def process_tick(self, tick: TickData) -> Optional[OrderBookSnapshot]:
        """
        Process a market tick and update order book state.
        
        Args:
            tick: Market tick data
            
        Returns:
            OrderBookSnapshot if book state changed significantly
        """
        self.stats['updates_processed'] += 1
        self.stats['last_update_time'] = tick.timestamp
        
        snapshot = None
        
        try:
            if tick.tick_type == TickType.TRADE:
                snapshot = self._process_trade(tick)
            elif tick.tick_type == TickType.QUOTE:
                snapshot = self._process_quote(tick)
            elif tick.tick_type == TickType.DEPTH:
                snapshot = self._process_depth(tick)
            elif tick.tick_type == TickType.ORDER:
                snapshot = self._process_order(tick)
            
            # Update order book with tick data
            self.order_book.update(tick)
            
            return snapshot
            
        except Exception as e:
            self.stats['reconstruction_errors'] += 1
            print(f"Error processing tick: {e}")
            return None
    
    def _process_trade(self, tick: TradeData) -> Optional[OrderBookSnapshot]:
        """Process trade tick and infer order book impact."""
        self.trade_count += 1
        self.total_volume += tick.volume
        
        # Update VWAP
        self.volume_weighted_price = (
            (self.volume_weighted_price * (self.total_volume - tick.volume) + 
             tick.price * tick.volume) / self.total_volume
        )
        
        # Infer which side of the book was hit
        current_mid = self.order_book.get_mid_price()
        
        if current_mid > 0:
            # Determine if buyer or seller initiated
            if tick.price > current_mid:
                # Likely buyer initiated (hit ask)
                side_hit = 'ask'
            elif tick.price < current_mid:
                # Likely seller initiated (hit bid)
                side_hit = 'bid'
            else:
                # At mid price - use additional signals
                side_hit = 'ask' if getattr(tick, 'buyer_initiated', True) else 'bid'
        else:
            side_hit = 'unknown'
        
        # Create snapshot if significant trade
        if tick.volume > self.total_volume / self.trade_count * 2:  # Above average volume
            return self._create_snapshot(tick.timestamp, last_trade_price=tick.price, 
                                       last_trade_volume=tick.volume)
        
        return None
    
    def _process_quote(self, tick: QuoteData) -> Optional[OrderBookSnapshot]:
        """Process quote tick and update BBO."""
        # Update bid/ask levels
        bid_levels = [(tick.bid_price, tick.bid_size)] if tick.bid_price > 0 else []
        ask_levels = [(tick.ask_price, tick.ask_size)] if tick.ask_price > 0 else []
        
        # Check if this represents a significant change
        should_snapshot = False
        
        if self.current_snapshot:
            # Compare with previous snapshot
            prev_bid = self.current_snapshot.best_bid
            prev_ask = self.current_snapshot.best_ask
            
            # Significant price change (>1 tick)
            min_tick = 0.01  # Assume 1 cent minimum tick
            if (prev_bid and abs(tick.bid_price - prev_bid) > min_tick) or \
               (prev_ask and abs(tick.ask_price - prev_ask) > min_tick):
                should_snapshot = True
                
            # Significant size change (>50%)
            if self.current_snapshot.bid_levels and len(self.current_snapshot.bid_levels) > 0:
                prev_bid_size = self.current_snapshot.bid_levels[0][1]
                if prev_bid_size > 0 and abs(tick.bid_size - prev_bid_size) / prev_bid_size > 0.5:
                    should_snapshot = True
        else:
            should_snapshot = True  # First quote
        
        if should_snapshot:
            snapshot = OrderBookSnapshot(
                timestamp=tick.timestamp,
                symbol=self.symbol,
                bid_levels=bid_levels,
                ask_levels=ask_levels
            )
            
            self.current_snapshot = snapshot
            return snapshot
        
        return None
    
    def _process_depth(self, tick: DepthData) -> Optional[OrderBookSnapshot]:
        """Process full depth tick and reconstruct order book."""
        # Build bid and ask levels from depth data
        bid_levels = []
        ask_levels = []
        
        # Process bid levels
        for i in range(min(len(tick.bid_prices), len(tick.bid_sizes), self.max_levels)):
            if tick.bid_prices[i] > 0 and tick.bid_sizes[i] > 0:
                bid_levels.append((tick.bid_prices[i], tick.bid_sizes[i]))
        
        # Process ask levels  
        for i in range(min(len(tick.ask_prices), len(tick.ask_sizes), self.max_levels)):
            if tick.ask_prices[i] > 0 and tick.ask_sizes[i] > 0:
                ask_levels.append((tick.ask_prices[i], tick.ask_sizes[i]))
        
        # Sort levels (bids descending, asks ascending)
        bid_levels.sort(key=lambda x: x[0], reverse=True)
        ask_levels.sort(key=lambda x: x[0])
        
        # Create snapshot
        snapshot = OrderBookSnapshot(
            timestamp=tick.timestamp,
            symbol=self.symbol,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            sequence_number=tick.sequence_number
        )
        
        self.current_snapshot = snapshot
        self.stats['snapshots_created'] += 1
        
        return snapshot
    
    def _process_order(self, tick: OrderData) -> Optional[OrderBookSnapshot]:
        """Process individual order event for L3 reconstruction."""
        order_id = tick.order_id
        
        if tick.action == 'add':
            # New order
            order = Order(
                symbol=self.symbol,
                order_type=OrderType.LIMIT,
                side=OrderSide.BUY if tick.side == 'buy' else OrderSide.SELL,
                quantity=tick.volume,
                price=tick.price,
                order_id=order_id,
                timestamp=tick.timestamp
            )
            
            self.active_orders[order_id] = order
            self.order_book.add_order(order)
            
        elif tick.action == 'modify':
            # Modify existing order
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.quantity = tick.volume
                order.price = tick.price
        
        elif tick.action == 'cancel':
            # Cancel order
            if order_id in self.active_orders:
                self.order_book.cancel_order(order_id)
                del self.active_orders[order_id]
        
        # Create snapshot periodically or on significant changes
        self.order_sequence += 1
        if self.order_sequence % 10 == 0:  # Every 10 orders
            return self._create_snapshot_from_order_book(tick.timestamp)
        
        return None
    
    def _create_snapshot(self, timestamp: datetime, 
                        last_trade_price: Optional[float] = None,
                        last_trade_volume: Optional[float] = None) -> OrderBookSnapshot:
        """Create snapshot from current order book state."""
        depth = self.order_book.get_market_depth(self.max_levels)
        
        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            symbol=self.symbol,
            bid_levels=depth['bids'],
            ask_levels=depth['asks'],
            last_trade_price=last_trade_price,
            last_trade_volume=last_trade_volume
        )
        
        self.current_snapshot = snapshot
        self._store_snapshot(snapshot)
        
        return snapshot
    
    def _create_snapshot_from_order_book(self, timestamp: datetime) -> OrderBookSnapshot:
        """Create snapshot directly from order book."""
        return self._create_snapshot(timestamp)
    
    def _store_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Store snapshot in history."""
        self.snapshots.append(snapshot)
        
        # Maintain maximum history size
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        
        self.stats['snapshots_created'] += 1
    
    def get_snapshot_at_time(self, target_time: datetime) -> Optional[OrderBookSnapshot]:
        """Get order book snapshot at specific time."""
        if not self.snapshots:
            return None
        
        # Binary search for closest snapshot
        left, right = 0, len(self.snapshots) - 1
        best_snapshot = None
        
        while left <= right:
            mid = (left + right) // 2
            snapshot = self.snapshots[mid]
            
            if snapshot.timestamp <= target_time:
                best_snapshot = snapshot
                left = mid + 1
            else:
                right = mid - 1
        
        return best_snapshot
    
    def get_snapshots_in_range(self, start_time: datetime, 
                              end_time: datetime) -> List[OrderBookSnapshot]:
        """Get all snapshots in time range."""
        result = []
        for snapshot in self.snapshots:
            if start_time <= snapshot.timestamp <= end_time:
                result.append(snapshot)
        return result
    
    def calculate_liquidity_metrics(self, levels: int = 5) -> Dict[str, float]:
        """Calculate liquidity metrics from current order book state."""
        if not self.current_snapshot:
            return {}
        
        snapshot = self.current_snapshot
        
        # Basic metrics
        metrics = {
            'spread': snapshot.spread or 0,
            'spread_bps': snapshot.spread_bps or 0,
            'mid_price': snapshot.mid_price or 0,
        }
        
        # Depth metrics
        if snapshot.bid_levels and snapshot.ask_levels:
            # Total depth at each level
            bid_depth = sum(size for _, size in snapshot.bid_levels[:levels])
            ask_depth = sum(size for _, size in snapshot.ask_levels[:levels])
            
            metrics.update({
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': bid_depth + ask_depth,
                'depth_imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
            })
            
            # Weighted average prices
            if bid_depth > 0:
                bid_vwap = sum(price * size for price, size in snapshot.bid_levels[:levels]) / bid_depth
                metrics['bid_vwap'] = bid_vwap
            
            if ask_depth > 0:
                ask_vwap = sum(price * size for price, size in snapshot.ask_levels[:levels]) / ask_depth
                metrics['ask_vwap'] = ask_vwap
        
        # Market impact estimation
        if snapshot.mid_price:
            # Estimate impact for various trade sizes
            for size in [100, 1000, 10000]:
                bid_impact = self._calculate_market_impact(size, 'buy')
                ask_impact = self._calculate_market_impact(size, 'sell')
                
                metrics[f'buy_impact_{size}'] = bid_impact
                metrics[f'sell_impact_{size}'] = ask_impact
        
        return metrics
    
    def _calculate_market_impact(self, order_size: float, side: str) -> float:
        """Calculate estimated market impact for given order size."""
        if not self.current_snapshot:
            return 0
        
        snapshot = self.current_snapshot
        levels = snapshot.ask_levels if side == 'buy' else snapshot.bid_levels
        
        if not levels:
            return 0
        
        remaining_size = order_size
        total_cost = 0
        
        for price, size in levels:
            if remaining_size <= 0:
                break
            
            fill_size = min(remaining_size, size)
            total_cost += fill_size * price
            remaining_size -= fill_size
        
        if remaining_size > 0:
            # Didn't find enough liquidity
            return float('inf')
        
        # Calculate impact as price difference from mid
        avg_price = total_cost / order_size
        mid_price = snapshot.mid_price
        
        if mid_price and mid_price > 0:
            impact = abs(avg_price - mid_price) / mid_price
            return impact
        
        return 0
    
    def get_price_level_analysis(self) -> Dict[str, Any]:
        """Analyze price level distribution and dynamics."""
        if not self.snapshots:
            return {}
        
        # Collect all price levels from recent snapshots
        recent_snapshots = self.snapshots[-100:]  # Last 100 snapshots
        
        bid_prices = []
        ask_prices = []
        spreads = []
        depths = []
        
        for snapshot in recent_snapshots:
            if snapshot.bid_levels:
                bid_prices.extend([price for price, _ in snapshot.bid_levels])
            if snapshot.ask_levels:
                ask_prices.extend([price for price, _ in snapshot.ask_levels])
            
            if snapshot.spread:
                spreads.append(snapshot.spread)
            
            bid_depth = sum(size for _, size in snapshot.bid_levels)
            ask_depth = sum(size for _, size in snapshot.ask_levels)
            depths.append(bid_depth + ask_depth)
        
        analysis = {}
        
        if spreads:
            analysis['spread_stats'] = {
                'mean': np.mean(spreads),
                'std': np.std(spreads),
                'min': np.min(spreads),
                'max': np.max(spreads),
                'median': np.median(spreads)
            }
        
        if depths:
            analysis['depth_stats'] = {
                'mean': np.mean(depths),
                'std': np.std(depths),
                'min': np.min(depths),
                'max': np.max(depths)
            }
        
        if bid_prices and ask_prices:
            analysis['price_stats'] = {
                'bid_range': (np.min(bid_prices), np.max(bid_prices)),
                'ask_range': (np.min(ask_prices), np.max(ask_prices)),
                'price_levels_count': len(set(bid_prices + ask_prices))
            }
        
        return analysis
    
    def export_snapshots_to_dataframe(self) -> pd.DataFrame:
        """Export order book snapshots to pandas DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()
        
        data = []
        for snapshot in self.snapshots:
            row = {
                'timestamp': snapshot.timestamp,
                'symbol': snapshot.symbol,
                'best_bid': snapshot.best_bid,
                'best_ask': snapshot.best_ask,
                'mid_price': snapshot.mid_price,
                'spread': snapshot.spread,
                'spread_bps': snapshot.spread_bps,
                'last_trade_price': snapshot.last_trade_price,
                'last_trade_volume': snapshot.last_trade_volume,
                'sequence_number': snapshot.sequence_number
            }
            
            # Add depth information
            for i in range(min(5, len(snapshot.bid_levels))):
                row[f'bid_price_{i}'] = snapshot.bid_levels[i][0]
                row[f'bid_size_{i}'] = snapshot.bid_levels[i][1]
            
            for i in range(min(5, len(snapshot.ask_levels))):
                row[f'ask_price_{i}'] = snapshot.ask_levels[i][0]
                row[f'ask_size_{i}'] = snapshot.ask_levels[i][1]
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive LOB processing statistics."""
        base_stats = self.stats.copy()
        
        base_stats.update({
            'symbol': self.symbol,
            'total_snapshots': len(self.snapshots),
            'active_orders': len(self.active_orders),
            'trade_count': self.trade_count,
            'total_volume': self.total_volume,
            'vwap': self.volume_weighted_price,
            'current_snapshot': {
                'timestamp': self.current_snapshot.timestamp.isoformat() if self.current_snapshot else None,
                'best_bid': self.current_snapshot.best_bid if self.current_snapshot else None,
                'best_ask': self.current_snapshot.best_ask if self.current_snapshot else None,
                'spread': self.current_snapshot.spread if self.current_snapshot else None,
            } if self.current_snapshot else None
        })
        
        # Add liquidity metrics if available
        if self.current_snapshot:
            base_stats['liquidity_metrics'] = self.calculate_liquidity_metrics()
        
        return base_stats
    
    def reset(self) -> None:
        """Reset processor state."""
        self.current_snapshot = None
        self.snapshots.clear()
        self.active_orders.clear()
        self.order_sequence = 0
        self.trade_count = 0
        self.total_volume = 0.0
        self.volume_weighted_price = 0.0
        self.price_levels.clear()
        self.level_updates.clear()
        
        self.stats = {
            'updates_processed': 0,
            'snapshots_created': 0,
            'reconstruction_errors': 0,
            'last_update_time': None
        }
