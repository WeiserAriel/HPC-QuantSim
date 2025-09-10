#!/usr/bin/env python3
"""
Simple HPC QuantSim simulation example.

This example demonstrates:
- Loading synthetic market data
- Creating and running strategies
- Collecting performance metrics
- Basic system usage
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """Run simple simulation example."""
    print("=" * 60)
    print("HPC QuantSim - Simple Simulation Example")
    print("=" * 60)
    
    try:
        # Import HPC QuantSim components
        from hpc_quantsim import create_simulation, print_system_info
        from hpc_quantsim.config import create_default_config
        from hpc_quantsim.core import MovingAverageStrategy, MeanReversionStrategy
        from hpc_quantsim.market import MarketReplay, ReplayMode
        
        # Print system capabilities
        print_system_info()
        print()
        
        # Create configuration
        print("Creating simulation configuration...")
        config = create_default_config()
        
        # Override some settings for this example
        config.simulation.num_simulations = 100
        config.simulation.max_parallel = 10
        config.simulation.symbols = ["AAPL", "MSFT", "GOOGL"]
        
        print(f"Configuration: {config.simulation.num_simulations} simulations, {len(config.simulation.symbols)} symbols")
        print()
        
        # Create simulation engine
        print("Initializing simulation engine...")
        sim_engine = create_simulation(**config.simulation.__dict__)
        print()
        
        # Create synthetic market data
        print("Generating synthetic market data...")
        market_replay = MarketReplay(replay_mode=ReplayMode.BATCH)
        
        for symbol in config.simulation.symbols:
            print(f"  Creating data for {symbol}...")
            market_replay.create_synthetic_data(
                symbol=symbol,
                duration_hours=4,  # 4 hours of trading
                start_price=np.random.uniform(50, 200),
                volatility=np.random.uniform(0.15, 0.30)  # 15-30% volatility
            )
        
        # Load data into simulation
        sim_engine.market_replay = market_replay
        print(f"Generated market data for {len(market_replay.get_symbols())} symbols")
        print()
        
        # Create and register strategies
        print("Setting up trading strategies...")
        
        # Moving Average Strategy
        ma_strategy = MovingAverageStrategy("MA_20_50")
        ma_strategy.initialize({
            'short_window': 20,
            'long_window': 50
        })
        sim_engine.register_strategy(ma_strategy, "MovingAverage")
        
        # Mean Reversion Strategy
        mr_strategy = MeanReversionStrategy("MeanRevert_2std")
        mr_strategy.initialize({
            'lookback_window': 50,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5
        })
        sim_engine.register_strategy(mr_strategy, "MeanReversion")
        
        print(f"Registered {len(sim_engine.strategies)} strategies")
        print()
        
        # Run simulation
        print("Starting simulation...")
        print("This may take a few minutes...")
        
        start_time = datetime.now()
        results = sim_engine.run_simulation(max_workers=4)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        print(f"Simulation completed in {duration:.2f} seconds")
        print()
        
        # Analyze results
        print("=" * 40)
        print("SIMULATION RESULTS")
        print("=" * 40)
        
        if results:
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            print(f"Total simulations: {len(results)}")
            print(f"Successful: {len(successful_results)}")
            print(f"Failed: {len(failed_results)}")
            print(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
            print()
            
            if successful_results:
                # Performance statistics
                pnls = [r.final_pnl for r in successful_results]
                sharpe_ratios = [r.sharpe_ratio for r in successful_results if not np.isnan(r.sharpe_ratio)]
                drawdowns = [r.max_drawdown for r in successful_results]
                
                print("Performance Statistics:")
                print(f"  PnL - Mean: ${np.mean(pnls):.2f}, Std: ${np.std(pnls):.2f}")
                print(f"  PnL - Min: ${np.min(pnls):.2f}, Max: ${np.max(pnls):.2f}")
                
                if sharpe_ratios:
                    print(f"  Sharpe Ratio - Mean: {np.mean(sharpe_ratios):.3f}, Std: {np.std(sharpe_ratios):.3f}")
                
                print(f"  Max Drawdown - Mean: {np.mean(drawdowns)*100:.2f}%, Worst: {np.min(drawdowns)*100:.2f}%")
                print()
                
                # Strategy breakdown
                print("Strategy Performance:")
                for strategy_name in sim_engine.strategies.keys():
                    strategy_results = [r for r in successful_results if r.strategy_name == strategy_name]
                    if strategy_results:
                        strategy_pnls = [r.final_pnl for r in strategy_results]
                        print(f"  {strategy_name}:")
                        print(f"    Simulations: {len(strategy_results)}")
                        print(f"    Mean PnL: ${np.mean(strategy_pnls):.2f}")
                        print(f"    Win Rate: {len([p for p in strategy_pnls if p > 0])/len(strategy_pnls)*100:.1f}%")
                print()
        
        # Get summary statistics from engine
        print("Engine Summary Statistics:")
        summary = sim_engine.get_summary_statistics()
        if summary:
            if 'pnl_statistics' in summary:
                pnl_stats = summary['pnl_statistics']
                print(f"  Total Simulations: {summary.get('total_simulations', 0)}")
                print(f"  Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
                print(f"  Mean PnL: ${pnl_stats.get('mean', 0):.2f}")
                print(f"  PnL 95th Percentile: ${pnl_stats.get('percentiles', {}).get('95th', 0):.2f}")
                print(f"  PnL 5th Percentile: ${pnl_stats.get('percentiles', {}).get('5th', 0):.2f}")
        print()
        
        # Save results
        print("Saving results...")
        results_file = "results/simple_simulation_results.csv"
        os.makedirs("results", exist_ok=True)
        sim_engine.save_results(results_file)
        print(f"Results saved to: {results_file}")
        print()
        
        print("=" * 60)
        print("Simulation completed successfully!")
        print("Try modifying the parameters and running again.")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def create_sample_config():
    """Create a sample configuration file."""
    from hpc_quantsim.config import create_default_config, save_config
    
    config = create_default_config()
    
    # Customize for example
    config.simulation.num_simulations = 1000
    config.simulation.max_parallel = 50
    config.simulation.symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
    
    config.hpc.use_gpu = False  # Start with CPU-only
    config.hpc.use_mpi = False
    
    config.metrics.collect_pnl = True
    config.metrics.collect_sharpe = True
    config.metrics.collect_drawdown = True
    
    # Save config
    os.makedirs("examples", exist_ok=True)
    save_config(config, "examples/sample_config.yaml")
    print("Sample configuration saved to: examples/sample_config.yaml")


if __name__ == "__main__":
    # Check if we should create sample config
    if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
        create_sample_config()
    else:
        exit_code = main()
        sys.exit(exit_code)
