"""
Strategy management and plugin system for HPC QuantSim.

This module provides:
- Strategy plugin loading and management
- Strategy factory and registry
- Built-in strategy implementations
- Strategy performance tracking
"""

from ..core.strategy_interface import Strategy, MovingAverageStrategy, MeanReversionStrategy, StatelessStrategy

# Create aliases for backward compatibility
BaseStrategy = Strategy


class StrategyManager:
    """
    Strategy plugin manager for HPC QuantSim.
    
    Features:
    - Dynamic strategy loading from plugins
    - Strategy registry and factory
    - Strategy configuration management
    - Performance tracking across strategies
    """
    
    def __init__(self):
        """Initialize strategy manager."""
        self.strategies = {}
        self.strategy_configs = {}
        
        # Register built-in strategies
        self._register_builtin_strategies()
    
    def _register_builtin_strategies(self):
        """Register built-in strategy implementations."""
        self.register_strategy_class('moving_average', MovingAverageStrategy)
        self.register_strategy_class('mean_reversion', MeanReversionStrategy)
    
    def register_strategy_class(self, name: str, strategy_class: type):
        """Register a strategy class."""
        if not issubclass(strategy_class, Strategy):
            raise ValueError(f"Strategy class must inherit from Strategy")
        
        self.strategies[name] = strategy_class
    
    def create_strategy(self, name: str, strategy_name: str = None, **params):
        """Create strategy instance."""
        if name not in self.strategies:
            raise ValueError(f"Unknown strategy: {name}")
        
        strategy_class = self.strategies[name]
        instance = strategy_class(strategy_name)
        
        if params:
            instance.initialize(params)
        
        return instance
    
    def get_available_strategies(self):
        """Get list of available strategies."""
        return list(self.strategies.keys())
    
    def load_strategy_plugin(self, plugin_path: str, strategy_name: str):
        """Load strategy from plugin file."""
        import importlib.util
        from pathlib import Path
        
        plugin_path = Path(plugin_path)
        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin file not found: {plugin_path}")
        
        # Load module
        spec = importlib.util.spec_from_file_location(strategy_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find strategy class
        strategy_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, Strategy) and 
                attr != Strategy):
                strategy_class = attr
                break
        
        if not strategy_class:
            raise ValueError(f"No Strategy class found in {plugin_path}")
        
        # Register strategy
        self.register_strategy_class(strategy_name, strategy_class)
        return strategy_class


# Global strategy manager instance
_global_manager = None

def get_strategy_manager():
    """Get global strategy manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = StrategyManager()
    return _global_manager


__all__ = [
    'Strategy', 'BaseStrategy', 'StatelessStrategy',
    'MovingAverageStrategy', 'MeanReversionStrategy', 
    'StrategyManager', 'get_strategy_manager'
]
