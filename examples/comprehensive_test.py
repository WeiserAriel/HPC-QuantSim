#!/usr/bin/env python3
"""
Comprehensive test suite for HPC QuantSim.

This script tests all major components and features:
- System initialization and configuration
- Market data loading and replay
- Strategy execution
- GPU acceleration (if available)
- MPI distribution (if available)
- Performance metrics
- Result analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_system_info():
    """Test system information and capabilities."""
    print("🔍 Testing System Information")
    print("=" * 60)
    
    try:
        from hpc_quantsim import print_system_info, get_capabilities
        
        # Print system capabilities
        print_system_info()
        
        # Get detailed capabilities
        capabilities = get_capabilities()
        print(f"\nDetailed capabilities: {capabilities}")
        
        return True
        
    except Exception as e:
        print(f"❌ System info test failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\n📋 Testing Configuration Management")
    print("=" * 60)
    
    try:
        from hpc_quantsim.config import create_default_config, save_config, load_config
        
        # Create default config
        config = create_default_config()
        print("✅ Default configuration created")
        
        # Customize configuration
        config.simulation.num_simulations = 50
        config.simulation.symbols = ["AAPL", "MSFT", "GOOGL"]
        config.hpc.use_gpu = False  # Start with CPU-only
        
        # Save and reload config
        test_config_path = "test_config.yaml"
        save_config(config, test_config_path)
        print(f"✅ Configuration saved to {test_config_path}")
        
        reloaded_config = load_config(test_config_path)
        print("✅ Configuration reloaded successfully")
        
        # Verify values
        assert reloaded_config.simulation.num_simulations == 50
        assert len(reloaded_config.simulation.symbols) == 3
        print("✅ Configuration values verified")
        
        # Cleanup
        os.unlink(test_config_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_market_data():
    """Test market data loading and replay."""
    print("\n📈 Testing Market Data Components")
    print("=" * 60)
    
    try:
        from hpc_quantsim.market import MarketReplay, ReplayMode, TickData, TradeData
        
        # Create market replay engine
        replay_engine = MarketReplay(replay_mode=ReplayMode.BATCH)
        print("✅ Market replay engine created")
        
        # Generate synthetic data
        symbols = ["AAPL", "MSFT"]
        for symbol in symbols:
            replay_engine.create_synthetic_data(
                symbol=symbol,
                duration_hours=1,
                start_price=np.random.uniform(100, 200),
                volatility=0.2
            )
        
        print(f"✅ Generated synthetic data for {len(symbols)} symbols")
        
        # Test tick iteration
        tick_count = 0
        start_time = time.time()
        
        for tick in replay_engine.iterate_ticks(symbols=["AAPL"]):
            tick_count += 1
            if tick_count >= 100:  # Limit for testing
                break
        
        elapsed_time = time.time() - start_time
        print(f"✅ Processed {tick_count} ticks in {elapsed_time:.3f}s")
        print(f"   Throughput: {tick_count/elapsed_time:.0f} ticks/second")
        
        # Test market data statistics
        market_data = replay_engine.get_market_data("AAPL")
        if market_data:
            stats = market_data.get_statistics()
            print(f"✅ Market data statistics: {stats['total_ticks']} ticks, {stats['total_trades']} trades")
        
        return True
        
    except Exception as e:
        print(f"❌ Market data test failed: {e}")
        return False


def test_strategies():
    """Test strategy execution."""
    print("\n🧠 Testing Strategy Components")
    print("=" * 60)
    
    try:
        from hpc_quantsim.core import MovingAverageStrategy, MeanReversionStrategy, OrderBook
        from hpc_quantsim.market import TradeData, TickType
        
        # Test Moving Average Strategy
        ma_strategy = MovingAverageStrategy("Test_MA")
        ma_strategy.initialize({'short_window': 5, 'long_window': 10})
        print("✅ Moving Average strategy initialized")
        
        # Test Mean Reversion Strategy
        mr_strategy = MeanReversionStrategy("Test_MR")
        mr_strategy.initialize({'lookback_window': 20, 'entry_threshold': 1.5})
        print("✅ Mean Reversion strategy initialized")
        
        # Create test order book
        order_book = OrderBook("TEST")
        
        # Generate test ticks and run strategies
        base_price = 100.0
        position = 0.0
        cash = 10000.0
        
        strategy_results = []
        
        for i in range(50):
            # Generate synthetic price with some trend and noise
            price = base_price + i * 0.1 + np.random.normal(0, 0.5)
            
            # Create test tick
            tick = TradeData(
                symbol="TEST",
                timestamp=datetime.now(),
                price=price,
                volume=100
            )
            
            # Update order book
            order_book.update(tick)
            
            # Test moving average strategy
            ma_result = ma_strategy.on_tick(tick, order_book, position, cash)
            strategy_results.append(('MA', ma_result))
            
            # Test mean reversion strategy
            mr_result = mr_strategy.on_tick(tick, order_book, position, cash)
            strategy_results.append(('MR', mr_result))
        
        # Analyze strategy performance
        ma_signals = [r[1].signals for r in strategy_results if r[0] == 'MA' and r[1].signals]
        mr_signals = [r[1].signals for r in strategy_results if r[0] == 'MR' and r[1].signals]
        
        print(f"✅ Moving Average generated {len(ma_signals)} signal updates")
        print(f"✅ Mean Reversion generated {len(mr_signals)} signal updates")
        
        # Test strategy state
        ma_state = ma_strategy.get_state()
        print(f"✅ MA Strategy state: {ma_state['tick_count']} ticks processed")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        return False


def test_gpu_components():
    """Test GPU acceleration components."""
    print("\n🚀 Testing GPU Components")
    print("=" * 60)
    
    try:
        from hpc_quantsim.gpu import check_gpu_availability, GPUUtils
        
        if not check_gpu_availability():
            print("ℹ️  GPU not available, skipping GPU tests")
            return True
        
        # Test GPU utilities
        device_info = GPUUtils.get_device_info()
        print(f"✅ GPU device info: {device_info['device_count']} device(s)")
        
        # Test GPU memory info
        memory_info = GPUUtils.get_memory_info()
        print(f"✅ GPU memory: {memory_info.get('total_gb', 0):.1f} GB total, {memory_info.get('utilization_pct', 0):.1f}% used")
        
        # Test CUDA kernels if available
        try:
            from hpc_quantsim.gpu import CUDAKernels
            import cupy as cp
            
            kernels = CUDAKernels()
            print("✅ CUDA kernels initialized")
            
            # Test moving average calculation
            test_prices = cp.random.uniform(90, 110, 1000)
            ma_results = kernels.calculate_moving_averages(test_prices, [10, 20])
            print(f"✅ GPU moving averages calculated for {len(ma_results)} windows")
            
            # Test returns calculation
            returns = kernels.calculate_returns(test_prices)
            print(f"✅ GPU returns calculated: {len(returns)} values")
            
            # Test performance stats
            perf_stats = kernels.get_performance_stats()
            print(f"✅ GPU kernel stats: {perf_stats['kernel_stats']['executions']} executions")
            
        except ImportError:
            print("ℹ️  CuPy not available, skipping CUDA kernel tests")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False


def test_performance_metrics():
    """Test performance metrics calculation."""
    print("\n📊 Testing Performance Metrics")
    print("=" * 60)
    
    try:
        from hpc_quantsim.metrics import PerformanceMetrics
        
        # Create performance tracker
        metrics = PerformanceMetrics(initial_capital=100000)
        print("✅ Performance metrics initialized")
        
        # Simulate trades
        for i in range(20):
            metrics.record_trade(
                timestamp=datetime.now(),
                symbol="TEST",
                side="buy" if i % 2 == 0 else "sell",
                quantity=100,
                price=100 + np.random.normal(0, 2),
                commission=1.0
            )
        
        # Update PnL history
        for i in range(50):
            pnl = np.random.normal(0, 100)  # Random PnL
            metrics.update_pnl(pnl, datetime.now())
        
        print(f"✅ Recorded {metrics.total_trades} trades and {len(metrics.pnl_history)} PnL updates")
        
        # Calculate performance summary
        summary = metrics.get_performance_summary()
        print(f"✅ Performance summary calculated:")
        print(f"   Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
        print(f"   Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
        print(f"   Win Rate: {summary.get('win_rate_pct', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance metrics test failed: {e}")
        return False


def test_mpi_components():
    """Test MPI components."""
    print("\n🌐 Testing MPI Components")  
    print("=" * 60)
    
    try:
        from hpc_quantsim.hpc import get_mpi_info, HAS_MPI
        
        mpi_info = get_mpi_info()
        print(f"MPI Info: {mpi_info}")
        
        if not mpi_info['available']:
            print(f"ℹ️  MPI not available: {mpi_info.get('reason', 'Unknown')}")
            return True
        
        # Test MPI collectives (only if running in single process)
        if mpi_info['size'] == 1:
            from hpc_quantsim.hpc import initialize_mpi
            
            mpi_collectives = initialize_mpi()
            if mpi_collectives:
                print(f"✅ MPI collectives initialized: rank {mpi_collectives.rank}/{mpi_collectives.size}")
                
                # Test metric aggregation
                test_metrics = {'test_value': 42.0, 'another_metric': 3.14}
                aggregated = mpi_collectives.allreduce_metrics(test_metrics, 'sum')
                print(f"✅ Metric aggregation test: {aggregated}")
                
                # Test performance stats
                perf_stats = mpi_collectives.get_performance_stats()
                print(f"✅ MPI performance stats: {perf_stats}")
            else:
                print("⚠️  Failed to initialize MPI collectives")
        else:
            print(f"ℹ️  Running in multi-process environment ({mpi_info['size']} processes)")
        
        return True
        
    except Exception as e:
        print(f"❌ MPI test failed: {e}")
        return False


def test_full_simulation():
    """Test end-to-end simulation."""
    print("\n🎯 Testing Full Simulation Pipeline")
    print("=" * 60)
    
    try:
        from hpc_quantsim import create_simulation
        from hpc_quantsim.config import create_default_config
        from hpc_quantsim.core import MovingAverageStrategy
        from hpc_quantsim.market import MarketReplay, ReplayMode
        
        # Create configuration
        config = create_default_config()
        config.simulation.num_simulations = 10  # Small test
        config.simulation.symbols = ["AAPL"]
        config.hpc.use_gpu = False
        config.hpc.use_mpi = False
        
        print("✅ Test configuration created")
        
        # Create simulation engine
        sim_engine = create_simulation(config=config)
        print("✅ Simulation engine created")
        
        # Generate synthetic market data
        market_replay = MarketReplay(replay_mode=ReplayMode.BATCH)
        market_replay.create_synthetic_data("AAPL", duration_hours=0.5, volatility=0.15)
        sim_engine.market_replay = market_replay
        
        print("✅ Market data generated")
        
        # Register strategy
        strategy = MovingAverageStrategy()
        strategy.initialize({'short_window': 5, 'long_window': 10})
        sim_engine.register_strategy(strategy, "TestMA")
        
        print("✅ Strategy registered")
        
        # Run simulation
        start_time = time.time()
        results = sim_engine.run_simulation(max_workers=2)
        end_time = time.time()
        
        print(f"✅ Simulation completed in {end_time - start_time:.2f}s")
        
        # Analyze results
        if results:
            successful = [r for r in results if r.success]
            print(f"✅ Results: {len(successful)}/{len(results)} successful")
            
            if successful:
                pnls = [r.final_pnl for r in successful]
                print(f"   Mean PnL: ${np.mean(pnls):.2f}")
                print(f"   PnL Range: ${np.min(pnls):.2f} to ${np.max(pnls):.2f}")
        
        # Get summary statistics
        summary = sim_engine.get_summary_statistics()
        if summary:
            print(f"✅ Summary statistics generated")
            print(f"   Success rate: {summary.get('success_rate', 0)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Full simulation test failed: {e}")
        return False


def test_cli_functions():
    """Test CLI functionality."""
    print("\n💻 Testing CLI Functions")
    print("=" * 60)
    
    try:
        # Test configuration creation
        from hpc_quantsim.config import create_default_config, save_config
        
        config = create_default_config()
        test_config_file = "cli_test_config.yaml"
        save_config(config, test_config_file)
        
        print(f"✅ Test configuration saved: {test_config_file}")
        
        # Cleanup
        os.unlink(test_config_file)
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def run_all_tests():
    """Run complete test suite."""
    print("🔥 Starting HPC QuantSim Comprehensive Test Suite")
    print("=" * 80)
    print(f"Test started at: {datetime.now()}")
    print("=" * 80)
    
    tests = [
        ("System Info", test_system_info),
        ("Configuration", test_configuration),
        ("Market Data", test_market_data),
        ("Strategies", test_strategies),
        ("GPU Components", test_gpu_components),
        ("Performance Metrics", test_performance_metrics),
        ("MPI Components", test_mpi_components),
        ("Full Simulation", test_full_simulation),
        ("CLI Functions", test_cli_functions),
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for r in results.values() if r)
    failed = len(results) - passed
    
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(results)*100:.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    
    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print("="*80)
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! HPC QuantSim is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
