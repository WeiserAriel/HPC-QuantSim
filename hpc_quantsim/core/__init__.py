"""
Core simulation engine for HPC QuantSim.

This module contains the main simulation orchestration components:
- SimulationEngine: Main controller for running simulations
- OrderBook: Limit order book implementation
- Strategy: Base strategy interface and management
"""

from .simulation_engine import SimulationEngine
from .order_book import OrderBook, Order, OrderType, OrderSide
from .strategy_interface import Strategy, StrategyResult, StrategyState

__all__ = [
    'SimulationEngine',
    'OrderBook', 'Order', 'OrderType', 'OrderSide',
    'Strategy', 'StrategyResult', 'StrategyState',
]
