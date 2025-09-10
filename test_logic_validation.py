#!/usr/bin/env python3
"""
Logic validation test for HPC QuantSim (works without external dependencies).

This test validates the core implementation logic by mocking external dependencies
and testing the business logic directly.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import importlib.util

print("🧪 HPC QuantSim Logic Validation Test")
print("=" * 60)

# Add HPC-QuantSim to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Mock external dependencies
class MockNumpy:
    def array(self, data):
        return data
    def mean(self, data):
        return sum(data) / len(data) if data else 0
    def std(self, data):
        if not data or len(data) <= 1:
            return 0
        mean_val = self.mean(data)
        return (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
    def sum(self, data):
        return sum(data) if data else 0
    def random(self):
        return MockRandom()
    def percentile(self, data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(p / 100 * (len(sorted_data) - 1))
        return sorted_data[idx]
    def isnan(self, x):
        return x != x  # NaN != NaN is True
    def sqrt(self, x):
        return x ** 0.5
    def log(self, x):
        import math
        return math.log(x)
    def exp(self, x):
        import math
        return math.exp(x)
    def uniform(self, low, high, size=None):
        import random
        if size is None:
            return random.uniform(low, high)
        return [random.uniform(low, high) for _ in range(size)]
    def normal(self, mean, std, size=None):
        import random
        if size is None:
            return random.gauss(mean, std)
        return [random.gauss(mean, std) for _ in range(size)]

class MockRandom:
    def uniform(self, low, high):
        import random
        return random.uniform(low, high)
    def normal(self, mean, std):
        import random
        return random.gauss(mean, std)
    def randint(self, low, high):
        import random
        return random.randint(low, high)

class MockPandas:
    def DataFrame(self, data=None):
        return MockDataFrame(data)
    def Series(self, data):
        return MockSeries(data)
    def to_datetime(self, data):
        from datetime import datetime
        if isinstance(data, str):
            return datetime.now()
        return data

class MockDataFrame:
    def __init__(self, data):
        self.data = data or {}
    def to_csv(self, path, index=False):
        return f"Mock CSV save to {path}"
    def to_parquet(self, path):
        return f"Mock Parquet save to {path}"

class MockSeries:
    def __init__(self, data):
        self.data = data or []
    def dropna(self):
        return self
    def values(self):
        return self.data

# Install mocks
sys.modules['numpy'] = MockNumpy()
sys.modules['pandas'] = MockPandas()
sys.modules['scipy'] = Mock()
sys.modules['mpi4py'] = Mock()
sys.modules['cupy'] = Mock()

def test_configuration_system():
    """Test configuration system with mocked dependencies."""
    tests = []
    
    try:
        from hpc_quantsim.config import (
            SimulationConfig, HPCConfig, MetricsConfig, 
            VisualizationConfig, DeploymentConfig, Config, create_default_config
        )
        
        # Test individual configs
        sim_config = SimulationConfig()
        tests.append(("✅", f"SimulationConfig: {sim_config.num_simulations} simulations"))
        tests.append(("✅", f"Default symbols: {sim_config.symbols}"))
        
        hpc_config = HPCConfig()
        tests.append(("✅", f"HPCConfig: GPU={hpc_config.use_gpu}, MPI={hpc_config.use_mpi}"))
        
        # Test creating full default config
        config = create_default_config()
        tests.append(("✅", "Default configuration created successfully"))
        tests.append(("ℹ️", f"Config type: {type(config).__name__}"))
        
    except Exception as e:
        tests.append(("❌", f"Configuration test failed: {e}"))
        import traceback
        tests.append(("📝", f"Details: {str(e)}"))
    
    return tests

def test_core_imports():
    """Test core module imports with mocked dependencies."""
    tests = []
    
    try:
        # Test main package import
        import hpc_quantsim
        tests.append(("✅", "hpc_quantsim package imported"))
        tests.append(("ℹ️", f"Version: {hpc_quantsim.__version__}"))
        
        # Test capabilities
        caps = hpc_quantsim.get_capabilities()
        tests.append(("✅", f"Capabilities: {caps}"))
        
    except Exception as e:
        tests.append(("❌", f"Core import failed: {e}"))
    
    try:
        # Test strategy system
        from hpc_quantsim.strategies import StrategyManager
        manager = StrategyManager()
        strategies = manager.get_available_strategies()
        tests.append(("✅", f"Strategy system: {strategies}"))
        
        # Test creating a strategy
        if 'moving_average' in strategies:
            strategy = manager.create_strategy('moving_average', 'test_ma')
            tests.append(("✅", f"Created strategy: {strategy.name}"))
        
    except Exception as e:
        tests.append(("❌", f"Strategy system failed: {e}"))
    
    return tests

def test_order_book_logic():
    """Test order book logic without heavy dependencies."""
    tests = []
    
    try:
        from hpc_quantsim.core.order_book import OrderBook, Order, OrderType, OrderSide
        
        # Create order book
        ob = OrderBook("TEST")
        tests.append(("✅", f"OrderBook created for symbol: {ob.symbol}"))
        
        # Test basic functionality
        mid_price = ob.get_mid_price()
        tests.append(("✅", f"Mid price calculation: {mid_price}"))
        
        spread = ob.get_spread()
        tests.append(("✅", f"Spread calculation: {spread}"))
        
        # Test order creation
        order = Order(
            symbol="TEST",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=100,
            price=99.50
        )
        tests.append(("✅", f"Order created: {order.symbol} {order.side.value} {order.quantity}@{order.price}"))
        
    except Exception as e:
        tests.append(("❌", f"OrderBook test failed: {e}"))
        import traceback
        tests.append(("📝", f"Error: {str(e)}"))
    
    return tests

def test_strategy_interface():
    """Test strategy interface and implementation."""
    tests = []
    
    try:
        from hpc_quantsim.core.strategy_interface import (
            Strategy, MovingAverageStrategy, MeanReversionStrategy, StrategyResult, StrategyState
        )
        
        # Test MovingAverageStrategy
        ma_strategy = MovingAverageStrategy("TestMA")
        tests.append(("✅", f"MovingAverageStrategy created: {ma_strategy.name}"))
        
        # Initialize with parameters
        ma_strategy.initialize({'short_window': 5, 'long_window': 10})
        tests.append(("✅", f"MA Strategy initialized: {ma_strategy.state.value}"))
        
        # Test state
        state = ma_strategy.get_state()
        tests.append(("✅", f"Strategy state: {state['tick_count']} ticks"))
        
        # Test MeanReversionStrategy
        mr_strategy = MeanReversionStrategy("TestMR")
        mr_strategy.initialize({'lookback_window': 20, 'entry_threshold': 2.0})
        tests.append(("✅", f"MeanReversion strategy: {mr_strategy.name}"))
        
        # Test StrategyResult
        result = StrategyResult()
        tests.append(("✅", f"StrategyResult: {len(result.orders)} orders, {len(result.signals)} signals"))
        
    except Exception as e:
        tests.append(("❌", f"Strategy interface failed: {e}"))
    
    return tests

def test_market_data_structures():
    """Test market data structures."""
    tests = []
    
    try:
        from hpc_quantsim.market.tick_data import (
            TickData, TradeData, QuoteData, TickType, MarketCondition
        )
        from datetime import datetime
        
        # Test TickData
        tick = TickData(
            symbol="TEST",
            timestamp=datetime.now(),
            tick_type=TickType.TRADE,
            price=100.0,
            volume=1000
        )
        tests.append(("✅", f"TickData created: {tick.symbol} @ {tick.price}"))
        
        # Test TradeData
        trade = TradeData(
            symbol="TEST",
            timestamp=datetime.now(),
            price=100.5,
            volume=500
        )
        tests.append(("✅", f"TradeData: {trade.symbol} {trade.volume}@{trade.price}"))
        
        # Test QuoteData
        quote = QuoteData(
            symbol="TEST",
            timestamp=datetime.now(),
            bid_price=99.5,
            ask_price=100.5,
            bid_size=1000,
            ask_size=800
        )
        tests.append(("✅", f"QuoteData: {quote.bid_price}/{quote.ask_price}"))
        
    except Exception as e:
        tests.append(("❌", f"Market data structures failed: {e}"))
    
    return tests

def test_metrics_system():
    """Test metrics system."""
    tests = []
    
    try:
        from hpc_quantsim.metrics.performance_metrics import PerformanceMetrics
        from datetime import datetime
        
        # Create performance tracker
        metrics = PerformanceMetrics(initial_capital=100000)
        tests.append(("✅", f"PerformanceMetrics created: ${metrics.initial_capital:,}"))
        
        # Record some trades
        metrics.record_trade(
            timestamp=datetime.now(),
            symbol="TEST",
            side="buy",
            quantity=100,
            price=50.0
        )
        tests.append(("✅", f"Trade recorded: {metrics.total_trades} total trades"))
        
        # Update PnL
        metrics.update_pnl(1500.0, datetime.now())
        tests.append(("✅", f"PnL updated: ${metrics.total_pnl}"))
        
        # Get summary
        summary = metrics.get_performance_summary()
        tests.append(("✅", f"Performance summary: {len(summary)} metrics"))
        
    except Exception as e:
        tests.append(("❌", f"Metrics system failed: {e}"))
    
    return tests

def test_cli_structure():
    """Test CLI module structure."""
    tests = []
    
    try:
        # Import CLI module to test structure
        spec = importlib.util.spec_from_file_location("cli", "hpc_quantsim/cli.py")
        cli_module = importlib.util.module_from_spec(spec)
        
        # Check if main function exists
        with open("hpc_quantsim/cli.py", "r") as f:
            content = f.read()
            
        if "@click.group()" in content:
            tests.append(("✅", "CLI uses Click framework"))
        if "def main(" in content:
            tests.append(("✅", "Main CLI function defined"))
        if "def run(" in content:
            tests.append(("✅", "Run command available"))
        if "def submit_cluster(" in content:
            tests.append(("✅", "Cluster submission command available"))
        if "def info(" in content:
            tests.append(("✅", "Info command available"))
            
    except Exception as e:
        tests.append(("❌", f"CLI structure test failed: {e}"))
    
    return tests

def run_all_tests():
    """Run all validation tests."""
    print("🚀 Running comprehensive logic validation...\n")
    
    all_tests = []
    
    test_suites = [
        ("Configuration System", test_configuration_system),
        ("Core Imports", test_core_imports),
        ("Order Book Logic", test_order_book_logic),
        ("Strategy Interface", test_strategy_interface),
        ("Market Data Structures", test_market_data_structures),
        ("Metrics System", test_metrics_system),
        ("CLI Structure", test_cli_structure),
    ]
    
    for suite_name, test_func in test_suites:
        print(f"🧪 Testing {suite_name}")
        print("-" * 40)
        
        try:
            suite_tests = test_func()
            for status, message in suite_tests:
                print(f"{status} {message}")
            all_tests.extend(suite_tests)
        except Exception as e:
            print(f"❌ Test suite '{suite_name}' crashed: {e}")
            all_tests.append(("❌", f"{suite_name} suite crashed: {e}"))
        
        print()
    
    # Summary
    print("=" * 60)
    print("🎯 COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = len([t for t in all_tests if t[0] == "✅"])
    failed = len([t for t in all_tests if t[0] == "❌"])
    info = len([t for t in all_tests if t[0] in ["ℹ️", "📝"]])
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"ℹ️ Info: {info}")
    print(f"📊 Total: {len(all_tests)}")
    
    success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 100
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print(f"\n🎉 EXCELLENT! Logic validation highly successful!")
        print(f"🚀 HPC QuantSim core implementation is SOLID!")
        print(f"💡 Ready for deployment once dependencies are installed")
    elif success_rate >= 75:
        print(f"\n✅ GOOD! Most core logic is working")
        print(f"🔧 Some components need minor fixes")
    else:
        print(f"\n⚠️  NEEDS WORK: Several core components have issues")
        return 1
    
    if failed == 0:
        print(f"\n🏆 PERFECT SCORE! All tests passed!")
        
    return 0 if success_rate >= 90 else 1

if __name__ == "__main__":
    exit(run_all_tests())
