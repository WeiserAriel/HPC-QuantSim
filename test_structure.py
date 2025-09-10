#!/usr/bin/env python3
"""
Structure validation test for HPC QuantSim (no external dependencies).

This test validates the implementation structure without requiring
numpy, pandas, or other external libraries.
"""

import sys
import os
import importlib
from pathlib import Path

print("🧪 HPC QuantSim Structure Validation Test")
print("=" * 50)

# Add HPC-QuantSim to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_module_structure():
    """Test that all expected modules and classes can be imported."""
    tests = []
    
    # Core structure test
    try:
        import hpc_quantsim
        tests.append(("✅", "hpc_quantsim package imports successfully"))
        
        # Check version and basic attributes
        if hasattr(hpc_quantsim, '__version__'):
            tests.append(("✅", f"Version: {hpc_quantsim.__version__}"))
        
        if hasattr(hpc_quantsim, 'get_capabilities'):
            tests.append(("✅", "get_capabilities function available"))
            
    except Exception as e:
        tests.append(("❌", f"hpc_quantsim import failed: {e}"))
        return tests
    
    # Test config module
    try:
        from hpc_quantsim.config import Config, SimulationConfig, HPCConfig
        tests.append(("✅", "Configuration classes import successfully"))
        
        # Test creating default config
        from hpc_quantsim.config import create_default_config
        config = create_default_config()
        tests.append(("✅", "Default configuration created"))
        tests.append(("ℹ️", f"Default simulations: {config.simulation.num_simulations}"))
        
    except Exception as e:
        tests.append(("❌", f"Config import failed: {e}"))
    
    # Test core module structure
    try:
        from hpc_quantsim.core import OrderBook, Strategy
        tests.append(("✅", "Core classes (OrderBook, Strategy) available"))
    except Exception as e:
        tests.append(("❌", f"Core imports failed: {e}"))
    
    # Test strategy system
    try:
        from hpc_quantsim.strategies import StrategyManager, BaseStrategy
        tests.append(("✅", "Strategy system available"))
        
        # Test strategy manager
        manager = StrategyManager()
        available = manager.get_available_strategies()
        tests.append(("✅", f"Built-in strategies: {available}"))
        
    except Exception as e:
        tests.append(("❌", f"Strategy system failed: {e}"))
    
    # Test market data structure
    try:
        from hpc_quantsim.market import TickType, MarketCondition
        tests.append(("✅", "Market data structures available"))
    except Exception as e:
        tests.append(("❌", f"Market data imports failed: {e}"))
    
    # Test metrics structure
    try:
        from hpc_quantsim.metrics import AggregationMethod
        tests.append(("✅", "Metrics system structure available"))
    except Exception as e:
        tests.append(("❌", f"Metrics import failed: {e}"))
    
    # Test HPC capabilities detection
    try:
        capabilities = hpc_quantsim.get_capabilities()
        tests.append(("ℹ️", f"CUDA Support: {'Yes' if capabilities.get('cuda', False) else 'No'}"))
        tests.append(("ℹ️", f"MPI Support: {'Yes' if capabilities.get('mpi', False) else 'No'}"))
        tests.append(("ℹ️", f"Visualization: {'Yes' if capabilities.get('visualization', False) else 'No'}"))
    except Exception as e:
        tests.append(("⚠️", f"Capability detection failed: {e}"))
    
    return tests

def test_file_structure():
    """Test that expected files and directories exist."""
    tests = []
    base_path = Path(__file__).parent
    
    expected_structure = {
        'hpc_quantsim': 'dir',
        'hpc_quantsim/__init__.py': 'file',
        'hpc_quantsim/config.py': 'file',
        'hpc_quantsim/cli.py': 'file',
        'hpc_quantsim/core': 'dir',
        'hpc_quantsim/core/__init__.py': 'file',
        'hpc_quantsim/core/simulation_engine.py': 'file',
        'hpc_quantsim/core/strategy_interface.py': 'file',
        'hpc_quantsim/core/order_book.py': 'file',
        'hpc_quantsim/market': 'dir',
        'hpc_quantsim/market/__init__.py': 'file',
        'hpc_quantsim/market/market_replay.py': 'file',
        'hpc_quantsim/market/tick_data.py': 'file',
        'hpc_quantsim/market/data_loader.py': 'file',
        'hpc_quantsim/market/lob_processor.py': 'file',
        'hpc_quantsim/metrics': 'dir',
        'hpc_quantsim/metrics/__init__.py': 'file',
        'hpc_quantsim/metrics/performance_metrics.py': 'file',
        'hpc_quantsim/metrics/metric_aggregator.py': 'file',
        'hpc_quantsim/metrics/risk_metrics.py': 'file',
        'hpc_quantsim/metrics/statistical_analysis.py': 'file',
        'hpc_quantsim/strategies': 'dir',
        'hpc_quantsim/strategies/__init__.py': 'file',
        'hpc_quantsim/gpu': 'dir',
        'hpc_quantsim/gpu/__init__.py': 'file',
        'hpc_quantsim/gpu/gpu_utils.py': 'file',
        'hpc_quantsim/gpu/cuda_kernels.py': 'file',
        'hpc_quantsim/gpu/gpu_memory_pool.py': 'file',
        'hpc_quantsim/hpc': 'dir',
        'hpc_quantsim/hpc/__init__.py': 'file',
        'hpc_quantsim/hpc/mpi_collectives.py': 'file',
        'hpc_quantsim/hpc/cluster_manager.py': 'file',
        'setup.py': 'file',
        'requirements.txt': 'file',
        'README.md': 'file',
        'PROJECT_STATUS.md': 'file',
        'examples': 'dir',
        'examples/simple_simulation.py': 'file',
        'examples/comprehensive_test.py': 'file',
    }
    
    for path_str, expected_type in expected_structure.items():
        path = base_path / path_str
        
        if not path.exists():
            tests.append(("❌", f"Missing: {path_str}"))
        elif expected_type == 'dir' and not path.is_dir():
            tests.append(("❌", f"Expected directory: {path_str}"))
        elif expected_type == 'file' and not path.is_file():
            tests.append(("❌", f"Expected file: {path_str}"))
        else:
            tests.append(("✅", f"Found: {path_str}"))
    
    return tests

def test_configuration_system():
    """Test configuration system without external dependencies."""
    tests = []
    
    try:
        from hpc_quantsim.config import (
            SimulationConfig, HPCConfig, MetricsConfig, 
            VisualizationConfig, DeploymentConfig, Config
        )
        
        # Test creating each config type
        sim_config = SimulationConfig()
        tests.append(("✅", f"SimulationConfig: {sim_config.num_simulations} simulations"))
        
        hpc_config = HPCConfig()
        tests.append(("✅", f"HPCConfig: GPU={hpc_config.use_gpu}, MPI={hpc_config.use_mpi}"))
        
        metrics_config = MetricsConfig()
        tests.append(("✅", f"MetricsConfig: PnL={metrics_config.collect_pnl}"))
        
        viz_config = VisualizationConfig()
        tests.append(("✅", f"VisualizationConfig: Port={viz_config.dashboard_port}"))
        
        deploy_config = DeploymentConfig()
        tests.append(("✅", f"DeploymentConfig: Scheduler={deploy_config.scheduler}"))
        
        # Test full config
        full_config = Config(
            simulation=sim_config,
            hpc=hpc_config,
            metrics=metrics_config,
            visualization=viz_config,
            deployment=deploy_config
        )
        tests.append(("✅", "Full Config object created"))
        
    except Exception as e:
        tests.append(("❌", f"Configuration system test failed: {e}"))
        import traceback
        tests.append(("📝", f"Error details: {traceback.format_exc()[:200]}..."))
    
    return tests

def main():
    """Run all structure validation tests."""
    all_tests = []
    
    print("1. Testing Module Structure")
    print("-" * 30)
    module_tests = test_module_structure()
    for status, message in module_tests:
        print(f"{status} {message}")
    all_tests.extend(module_tests)
    
    print(f"\n2. Testing File Structure")
    print("-" * 30)
    file_tests = test_file_structure()
    for status, message in file_tests[:10]:  # Show first 10 to avoid spam
        print(f"{status} {message}")
    if len(file_tests) > 10:
        print(f"... and {len(file_tests) - 10} more files")
    all_tests.extend(file_tests)
    
    print(f"\n3. Testing Configuration System")  
    print("-" * 30)
    config_tests = test_configuration_system()
    for status, message in config_tests:
        print(f"{status} {message}")
    all_tests.extend(config_tests)
    
    # Summary
    print(f"\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = len([t for t in all_tests if t[0] == "✅"])
    failed = len([t for t in all_tests if t[0] == "❌"])
    warnings = len([t for t in all_tests if t[0] == "⚠️"])
    info = len([t for t in all_tests if t[0] in ["ℹ️", "📝"]])
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️ Warnings: {warnings}")
    print(f"ℹ️ Info: {info}")
    print(f"📊 Total: {len(all_tests)}")
    
    success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    if failed == 0:
        print(f"\n🎉 ALL CORE TESTS PASSED!")
        print(f"🚀 HPC QuantSim structure is VALID and ready!")
        if warnings > 0:
            print(f"⚠️ Note: {warnings} warnings (likely missing external dependencies)")
    else:
        print(f"\n❌ {failed} tests failed - check implementation")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
