"""
Market replay engine for HPC QuantSim.

Provides high-performance tick-by-tick market data replay with:
- Multiple data source support (Parquet, HDF5, CSV)
- Real-time and accelerated replay modes
- Market microstructure simulation
- Anomaly injection for stress testing
"""

from enum import Enum
from typing import Iterator, List, Dict, Optional, Union, Any, Callable
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from .tick_data import TickData, TradeData, QuoteData, DepthData, MarketData, TickType, MarketCondition
from .data_loader import DataLoader, DataFormat


class ReplayMode(Enum):
    """Market replay modes."""
    HISTORICAL = "historical"    # Replay at historical speed
    ACCELERATED = "accelerated"  # Replay faster than real-time
    BATCH = "batch"             # Process all data at once
    REAL_TIME = "real_time"     # Replay at real-time speed


class AnomalyType(Enum):
    """Types of market anomalies to inject."""
    FLASH_CRASH = "flash_crash"
    VOLUME_SPIKE = "volume_spike"
    PRICE_GAP = "price_gap"
    SPOOFING = "spoofing"
    LATENCY_SPIKE = "latency_spike"


class MarketReplay:
    """
    High-performance market replay engine.
    
    Features:
    - Multi-symbol data replay
    - Configurable replay speeds
    - Market microstructure simulation
    - Anomaly injection capabilities
    - Memory-efficient streaming
    """
    
    def __init__(self, replay_mode: ReplayMode = ReplayMode.HISTORICAL,
                 acceleration_factor: float = 1.0):
        """Initialize market replay engine."""
        self.replay_mode = replay_mode
        self.acceleration_factor = acceleration_factor
        
        # Data storage
        self.market_data: Dict[str, MarketData] = {}
        self.data_loader = DataLoader()
        
        # Replay state
        self.is_replaying = False
        self.current_time: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Performance tracking
        self.ticks_processed = 0
        self.replay_start_time = 0.0
        self.total_replay_time = 0.0
        
        # Anomaly injection
        self.anomalies_enabled = False
        self.anomaly_probability = 0.001  # 0.1% chance per tick
        self.anomaly_config: Dict[AnomalyType, Dict[str, Any]] = {}
        
        # Microstructure simulation
        self.simulate_microstructure = True
        self.microstructure_noise = 0.0001  # 1 basis point
        
        # Callbacks
        self.tick_callbacks: List[Callable[[TickData], None]] = []
        self.anomaly_callbacks: List[Callable[[AnomalyType, Dict], None]] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, data_path: Union[str, Path], 
                  symbols: Optional[List[str]] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  data_format: Optional[DataFormat] = None) -> None:
        """
        Load market data for replay.
        
        Args:
            data_path: Path to market data file
            symbols: List of symbols to load (None for all)
            start_time: Start time for data filtering
            end_time: End time for data filtering  
            data_format: Data format override
        """
        self.logger.info(f"Loading market data from {data_path}")
        
        # Load data using data loader
        raw_data = self.data_loader.load(
            data_path, 
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            data_format=data_format
        )
        
        # Convert to MarketData objects
        for symbol, df in raw_data.items():
            market_data = MarketData(symbol, start_time, end_time)
            
            # Convert DataFrame rows to TickData objects
            for _, row in df.iterrows():
                tick = self._row_to_tick(row, symbol)
                if tick:
                    market_data.add_tick(tick)
            
            # Sort by timestamp for efficient replay
            market_data.sort_by_time()
            
            self.market_data[symbol] = market_data
            self.logger.info(f"Loaded {len(market_data)} ticks for {symbol}")
        
        # Set replay time range
        if self.market_data:
            all_ticks = []
            for market_data in self.market_data.values():
                all_ticks.extend(market_data.ticks)
            
            if all_ticks:
                all_ticks.sort(key=lambda x: x.timestamp)
                self.start_time = all_ticks[0].timestamp
                self.end_time = all_ticks[-1].timestamp
                self.logger.info(f"Replay range: {self.start_time} to {self.end_time}")
    
    def _row_to_tick(self, row: pd.Series, symbol: str) -> Optional[TickData]:
        """Convert DataFrame row to TickData object."""
        try:
            # Determine tick type based on available columns
            if 'trade_price' in row or 'last_price' in row:
                return TradeData(
                    symbol=symbol,
                    timestamp=pd.to_datetime(row['timestamp']),
                    price=float(row.get('trade_price', row.get('last_price', row.get('price', 0)))),
                    volume=float(row.get('volume', row.get('size', 0))),
                    trade_id=str(row.get('trade_id', '')),
                    buyer_initiated=row.get('buyer_initiated')
                )
            
            elif 'bid_price' in row and 'ask_price' in row:
                return QuoteData(
                    symbol=symbol,
                    timestamp=pd.to_datetime(row['timestamp']),
                    price=(float(row['bid_price']) + float(row['ask_price'])) / 2,
                    bid_price=float(row['bid_price']),
                    ask_price=float(row['ask_price']),
                    bid_size=float(row.get('bid_size', 0)),
                    ask_size=float(row.get('ask_size', 0))
                )
            
            elif 'price' in row:
                return TickData(
                    symbol=symbol,
                    timestamp=pd.to_datetime(row['timestamp']),
                    tick_type=TickType.TRADE,
                    price=float(row['price']),
                    volume=float(row.get('volume', 0))
                )
            
            else:
                self.logger.warning(f"Unable to parse row: {row}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error parsing row: {e}")
            return None
    
    def add_tick_callback(self, callback: Callable[[TickData], None]) -> None:
        """Add callback function called for each tick."""
        self.tick_callbacks.append(callback)
    
    def add_anomaly_callback(self, callback: Callable[[AnomalyType, Dict], None]) -> None:
        """Add callback function called when anomaly is injected."""
        self.anomaly_callbacks.append(callback)
    
    def enable_anomalies(self, probability: float = 0.001,
                        anomaly_types: Optional[List[AnomalyType]] = None) -> None:
        """Enable anomaly injection."""
        self.anomalies_enabled = True
        self.anomaly_probability = probability
        
        # Configure default anomalies
        if anomaly_types is None:
            anomaly_types = [AnomalyType.FLASH_CRASH, AnomalyType.VOLUME_SPIKE]
        
        for anomaly_type in anomaly_types:
            self.anomaly_config[anomaly_type] = self._get_default_anomaly_config(anomaly_type)
    
    def _get_default_anomaly_config(self, anomaly_type: AnomalyType) -> Dict[str, Any]:
        """Get default configuration for anomaly type."""
        configs = {
            AnomalyType.FLASH_CRASH: {
                'price_drop_pct': 0.05,  # 5% drop
                'duration_ms': 1000,     # 1 second
                'recovery_ms': 5000      # 5 second recovery
            },
            AnomalyType.VOLUME_SPIKE: {
                'volume_multiplier': 10.0,  # 10x normal volume
                'duration_ms': 2000         # 2 seconds
            },
            AnomalyType.PRICE_GAP: {
                'gap_pct': 0.02,      # 2% gap
                'direction': 'random'  # 'up', 'down', or 'random'
            },
            AnomalyType.SPOOFING: {
                'fake_depth_multiplier': 5.0,  # 5x fake depth
                'duration_ms': 3000            # 3 seconds
            },
            AnomalyType.LATENCY_SPIKE: {
                'delay_ms': 100,      # 100ms additional delay
                'duration_ms': 1000   # 1 second
            }
        }
        return configs.get(anomaly_type, {})
    
    def iterate_ticks(self, symbols: Optional[List[str]] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> Iterator[TickData]:
        """
        Iterate through ticks in chronological order.
        
        Args:
            symbols: Symbols to include (None for all)
            start_time: Start time filter
            end_time: End time filter
            
        Yields:
            TickData: Next tick in sequence
        """
        if not self.market_data:
            self.logger.warning("No market data loaded")
            return
        
        # Filter symbols
        if symbols is None:
            symbols = list(self.market_data.keys())
        
        # Collect all ticks from selected symbols
        all_ticks = []
        for symbol in symbols:
            if symbol in self.market_data:
                market_data = self.market_data[symbol]
                
                # Apply time filters
                ticks = market_data.ticks
                if start_time:
                    ticks = [t for t in ticks if t.timestamp >= start_time]
                if end_time:
                    ticks = [t for t in ticks if t.timestamp <= end_time]
                
                all_ticks.extend(ticks)
        
        # Sort by timestamp
        all_ticks.sort(key=lambda x: x.timestamp)
        
        if not all_ticks:
            self.logger.warning("No ticks found in specified time range")
            return
        
        # Start replay
        self.is_replaying = True
        self.replay_start_time = time.time()
        self.ticks_processed = 0
        
        first_tick_time = all_ticks[0].timestamp
        
        try:
            for tick in all_ticks:
                if not self.is_replaying:
                    break
                
                # Handle replay timing
                if self.replay_mode == ReplayMode.REAL_TIME:
                    self._handle_real_time_delay(tick, first_tick_time)
                elif self.replay_mode == ReplayMode.HISTORICAL:
                    self._handle_historical_delay(tick, first_tick_time)
                
                # Apply microstructure simulation
                if self.simulate_microstructure:
                    tick = self._apply_microstructure_effects(tick)
                
                # Inject anomalies if enabled
                if self.anomalies_enabled:
                    tick, anomaly_info = self._maybe_inject_anomaly(tick)
                    if anomaly_info:
                        for callback in self.anomaly_callbacks:
                            callback(anomaly_info['type'], anomaly_info)
                
                # Update current time
                self.current_time = tick.timestamp
                
                # Call tick callbacks
                for callback in self.tick_callbacks:
                    try:
                        callback(tick)
                    except Exception as e:
                        self.logger.error(f"Tick callback error: {e}")
                
                self.ticks_processed += 1
                yield tick
        
        finally:
            self.is_replaying = False
            self.total_replay_time = time.time() - self.replay_start_time
            self.logger.info(f"Replay completed: {self.ticks_processed} ticks in {self.total_replay_time:.2f}s")
    
    def _handle_real_time_delay(self, tick: TickData, first_tick_time: datetime) -> None:
        """Handle timing for real-time replay."""
        if hasattr(self, '_last_tick_time'):
            # Calculate time difference since last tick
            time_diff = (tick.timestamp - self._last_tick_time).total_seconds()
            
            # Apply acceleration factor
            sleep_time = time_diff / self.acceleration_factor
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self._last_tick_time = tick.timestamp
    
    def _handle_historical_delay(self, tick: TickData, first_tick_time: datetime) -> None:
        """Handle timing for historical replay."""
        if self.acceleration_factor <= 0:
            return
        
        # Calculate elapsed time in data
        data_elapsed = (tick.timestamp - first_tick_time).total_seconds()
        
        # Calculate expected real time elapsed
        expected_real_elapsed = data_elapsed / self.acceleration_factor
        
        # Calculate actual real time elapsed  
        actual_real_elapsed = time.time() - self.replay_start_time
        
        # Sleep if we're ahead of schedule
        sleep_time = expected_real_elapsed - actual_real_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    def _apply_microstructure_effects(self, tick: TickData) -> TickData:
        """Apply market microstructure simulation effects."""
        if self.microstructure_noise <= 0:
            return tick
        
        # Add price noise
        if tick.price > 0:
            noise = np.random.normal(0, tick.price * self.microstructure_noise)
            tick.price = max(0.01, tick.price + noise)  # Ensure positive price
        
        # Add volume noise for trades
        if isinstance(tick, TradeData) and tick.volume > 0:
            volume_noise = np.random.normal(1.0, 0.1)  # ±10% volume variation
            tick.volume = max(1, tick.volume * volume_noise)
        
        return tick
    
    def _maybe_inject_anomaly(self, tick: TickData) -> tuple[TickData, Optional[Dict]]:
        """Maybe inject an anomaly into the tick."""
        if np.random.random() > self.anomaly_probability:
            return tick, None
        
        # Choose random anomaly type
        anomaly_types = list(self.anomaly_config.keys())
        if not anomaly_types:
            return tick, None
        
        anomaly_type = np.random.choice(anomaly_types)
        config = self.anomaly_config[anomaly_type]
        
        # Apply anomaly based on type
        if anomaly_type == AnomalyType.FLASH_CRASH:
            return self._inject_flash_crash(tick, config)
        elif anomaly_type == AnomalyType.VOLUME_SPIKE:
            return self._inject_volume_spike(tick, config)
        elif anomaly_type == AnomalyType.PRICE_GAP:
            return self._inject_price_gap(tick, config)
        
        return tick, None
    
    def _inject_flash_crash(self, tick: TickData, config: Dict) -> tuple[TickData, Dict]:
        """Inject flash crash anomaly."""
        if tick.price > 0:
            drop_pct = config.get('price_drop_pct', 0.05)
            tick.price = tick.price * (1 - drop_pct)
            tick.condition = MarketCondition.VOLATILE
        
        anomaly_info = {
            'type': AnomalyType.FLASH_CRASH,
            'timestamp': tick.timestamp,
            'symbol': tick.symbol,
            'original_price': tick.price / (1 - config.get('price_drop_pct', 0.05)),
            'modified_price': tick.price,
            'config': config
        }
        
        self.logger.info(f"Injected flash crash for {tick.symbol} at {tick.timestamp}")
        return tick, anomaly_info
    
    def _inject_volume_spike(self, tick: TickData, config: Dict) -> tuple[TickData, Dict]:
        """Inject volume spike anomaly."""
        if isinstance(tick, TradeData):
            multiplier = config.get('volume_multiplier', 10.0)
            original_volume = tick.volume
            tick.volume = tick.volume * multiplier
            tick.condition = MarketCondition.VOLATILE
            
            anomaly_info = {
                'type': AnomalyType.VOLUME_SPIKE,
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'original_volume': original_volume,
                'modified_volume': tick.volume,
                'config': config
            }
            
            self.logger.info(f"Injected volume spike for {tick.symbol} at {tick.timestamp}")
            return tick, anomaly_info
        
        return tick, None
    
    def _inject_price_gap(self, tick: TickData, config: Dict) -> tuple[TickData, Dict]:
        """Inject price gap anomaly."""
        if tick.price > 0:
            gap_pct = config.get('gap_pct', 0.02)
            direction = config.get('direction', 'random')
            
            if direction == 'random':
                direction = np.random.choice(['up', 'down'])
            
            original_price = tick.price
            if direction == 'up':
                tick.price = tick.price * (1 + gap_pct)
            else:
                tick.price = tick.price * (1 - gap_pct)
            
            tick.condition = MarketCondition.FAST_MARKET
            
            anomaly_info = {
                'type': AnomalyType.PRICE_GAP,
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'original_price': original_price,
                'modified_price': tick.price,
                'direction': direction,
                'config': config
            }
            
            self.logger.info(f"Injected price gap ({direction}) for {tick.symbol} at {tick.timestamp}")
            return tick, anomaly_info
        
        return tick, None
    
    def stop_replay(self) -> None:
        """Stop the current replay."""
        self.is_replaying = False
        self.logger.info("Replay stopped")
    
    def get_replay_statistics(self) -> Dict[str, Any]:
        """Get replay performance statistics."""
        if self.total_replay_time > 0:
            ticks_per_second = self.ticks_processed / self.total_replay_time
        else:
            ticks_per_second = 0
        
        return {
            'ticks_processed': self.ticks_processed,
            'total_replay_time': self.total_replay_time,
            'ticks_per_second': ticks_per_second,
            'replay_mode': self.replay_mode.value,
            'acceleration_factor': self.acceleration_factor,
            'anomalies_enabled': self.anomalies_enabled,
            'microstructure_enabled': self.simulate_microstructure,
            'symbols_loaded': list(self.market_data.keys()),
            'time_range': {
                'start': self.start_time.isoformat() if self.start_time else None,
                'end': self.end_time.isoformat() if self.end_time else None,
                'current': self.current_time.isoformat() if self.current_time else None
            }
        }
    
    def create_synthetic_data(self, symbol: str, duration_hours: int = 1,
                             start_price: float = 100.0, 
                             volatility: float = 0.02) -> None:
        """
        Create synthetic market data for testing.
        
        Args:
            symbol: Symbol to create data for
            duration_hours: Duration in hours
            start_price: Starting price
            volatility: Price volatility (daily)
        """
        self.logger.info(f"Creating synthetic data for {symbol}")
        
        # Create market data container
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=duration_hours)
        
        market_data = MarketData(symbol, start_time, end_time)
        
        # Generate synthetic ticks (one per second)
        current_time = start_time
        current_price = start_price
        tick_interval = timedelta(seconds=1)
        
        # Random walk parameters
        dt = 1.0 / (252 * 6.5 * 3600)  # Time step (1 second as fraction of trading year)
        vol_per_step = volatility * np.sqrt(dt)
        
        while current_time <= end_time:
            # Generate price movement (geometric Brownian motion)
            random_return = np.random.normal(0, vol_per_step)
            current_price = current_price * np.exp(random_return)
            
            # Generate volume (log-normal distribution)
            base_volume = 1000
            volume = np.random.lognormal(np.log(base_volume), 0.5)
            
            # Create trade tick
            trade = TradeData(
                symbol=symbol,
                timestamp=current_time,
                price=current_price,
                volume=volume,
                buyer_initiated=np.random.random() > 0.5
            )
            
            market_data.add_tick(trade)
            
            # Sometimes generate quotes
            if np.random.random() < 0.1:  # 10% chance
                spread_bps = np.random.uniform(1, 5)  # 1-5 basis points
                spread = current_price * (spread_bps / 10000)
                
                quote = QuoteData(
                    symbol=symbol,
                    timestamp=current_time,
                    bid_price=current_price - spread/2,
                    ask_price=current_price + spread/2,
                    bid_size=np.random.uniform(100, 1000),
                    ask_size=np.random.uniform(100, 1000)
                )
                
                market_data.add_tick(quote)
            
            current_time += tick_interval
        
        # Store synthetic data
        market_data.sort_by_time()
        self.market_data[symbol] = market_data
        
        self.start_time = start_time
        self.end_time = end_time
        
        self.logger.info(f"Created {len(market_data)} synthetic ticks for {symbol}")
    
    def get_symbols(self) -> List[str]:
        """Get list of loaded symbols."""
        return list(self.market_data.keys())
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data for specific symbol."""
        return self.market_data.get(symbol)
    
    def clear_data(self) -> None:
        """Clear all loaded market data."""
        self.market_data.clear()
        self.start_time = None
        self.end_time = None
        self.current_time = None
        self.ticks_processed = 0
        self.logger.info("Cleared all market data")
