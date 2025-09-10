"""
HPC QuantSim - High-Frequency Market Simulator with Distributed GPU Compute

A high-performance market simulation platform for stress-testing and benchmarking
algorithmic trading strategies using real historical order book data across 
distributed GPU clusters.
"""

__version__ = "0.1.0"
__author__ = "HPC QuantSim Team"
__license__ = "Apache 2.0"

# Core imports
from .core import SimulationEngine, OrderBook, Strategy
from .market import MarketReplay, TickData, LOBProcessor
from .metrics import MetricAggregator, PerformanceMetrics, RiskMetrics
from .strategies import BaseStrategy, StrategyManager

# Optional imports based on available dependencies
try:
    from .gpu import CUDAKernels, GPUMemoryPool
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

try:
    from .hpc import MPICollectives, UCXTransport, ClusterManager
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

try:
    from .visualization import Dashboard, MetricsVisualizer
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# System capabilities
CAPABILITIES = {
    'cuda': HAS_CUDA,
    'mpi': HAS_MPI,
    'visualization': HAS_VISUALIZATION,
}

def get_capabilities():
    """Get available system capabilities."""
    return CAPABILITIES.copy()

def print_system_info():
    """Print system capabilities and configuration."""
    print(f"HPC QuantSim v{__version__}")
    print("=" * 40)
    print(f"CUDA Support: {'✓' if HAS_CUDA else '✗'}")
    print(f"MPI Support:  {'✓' if HAS_MPI else '✗'}")
    print(f"Visualization: {'✓' if HAS_VISUALIZATION else '✗'}")
    print("=" * 40)

# Convenience functions
def create_simulation(config_path=None, **kwargs):
    """Create a new simulation instance with configuration."""
    from .core import SimulationEngine
    from .config import load_config
    
    if config_path:
        config = load_config(config_path)
        kwargs.update(config)
    
    return SimulationEngine(**kwargs)

def load_market_data(data_path, format='parquet'):
    """Load market data from file."""
    from .market import MarketReplay
    return MarketReplay.load_data(data_path, format=format)

__all__ = [
    # Version info
    '__version__', '__author__', '__license__',
    
    # Core components
    'SimulationEngine', 'OrderBook', 'Strategy',
    'MarketReplay', 'TickData', 'LOBProcessor',
    'MetricAggregator', 'PerformanceMetrics', 'RiskMetrics',
    'BaseStrategy', 'StrategyManager',
    
    # System info
    'get_capabilities', 'print_system_info',
    
    # Convenience functions
    'create_simulation', 'load_market_data',
    
    # Capability flags
    'HAS_CUDA', 'HAS_MPI', 'HAS_VISUALIZATION', 'CAPABILITIES',
]
