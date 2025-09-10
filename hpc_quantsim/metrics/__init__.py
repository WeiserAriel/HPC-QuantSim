"""
Metrics collection and aggregation for HPC QuantSim.

This module provides:
- Real-time performance metrics collection
- Risk metrics calculation
- Distributed metric aggregation using HPC-X collectives
- Statistical analysis and reporting
"""

from .performance_metrics import PerformanceMetrics, TradeMetrics, PositionMetrics
from .risk_metrics import RiskMetrics, VaRCalculator, DrawdownAnalyzer
from .metric_aggregator import MetricAggregator, AggregationMethod
from .statistical_analysis import StatisticalAnalyzer, DistributionAnalyzer

__all__ = [
    'PerformanceMetrics', 'TradeMetrics', 'PositionMetrics',
    'RiskMetrics', 'VaRCalculator', 'DrawdownAnalyzer', 
    'MetricAggregator', 'AggregationMethod',
    'StatisticalAnalyzer', 'DistributionAnalyzer'
]
