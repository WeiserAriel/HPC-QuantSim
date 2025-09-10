"""
Command-line interface for HPC QuantSim.

Provides easy-to-use commands for running simulations:
- Single-node simulation execution
- Distributed cluster execution
- Configuration management
- System diagnostics
- Result analysis
"""

import click
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """HPC QuantSim - High-Frequency Market Simulator with Distributed GPU Compute"""
    pass


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file path')
@click.option('--scenarios', '-s', type=int, default=100, 
              help='Number of simulation scenarios')
@click.option('--data-path', '-d', type=click.Path(exists=True),
              help='Market data file path')
@click.option('--output-path', '-o', type=click.Path(), default='results',
              help='Output directory for results')
@click.option('--workers', '-w', type=int, default=4,
              help='Number of parallel workers')
@click.option('--gpu/--no-gpu', default=False,
              help='Use GPU acceleration')
def run(config, scenarios, data_path, output_path, workers, gpu):
    """Run simulation on single node."""
    
    click.echo("🚀 Starting HPC QuantSim simulation...")
    
    try:
        # Import components
        from . import create_simulation, print_system_info
        from .config import load_config, create_default_config
        from .core import MovingAverageStrategy, MeanReversionStrategy
        from .market import MarketReplay, ReplayMode
        
        # Print system info
        print_system_info()
        click.echo()
        
        # Load configuration
        if config:
            click.echo(f"Loading configuration from: {config}")
            config_obj = load_config(config)
        else:
            click.echo("Using default configuration")
            config_obj = create_default_config()
        
        # Override with command line options
        if scenarios:
            config_obj.simulation.num_simulations = scenarios
        if gpu:
            config_obj.hpc.use_gpu = True
        
        # Create simulation engine
        click.echo("Initializing simulation engine...")
        sim_engine = create_simulation(config=config_obj)
        
        # Handle market data
        if data_path:
            click.echo(f"Loading market data from: {data_path}")
            sim_engine.load_market_data(data_path)
        else:
            click.echo("Generating synthetic market data...")
            market_replay = MarketReplay(replay_mode=ReplayMode.BATCH)
            for symbol in config_obj.simulation.symbols:
                market_replay.create_synthetic_data(symbol)
            sim_engine.market_replay = market_replay
        
        # Setup default strategies if none configured
        if not config_obj.simulation.strategy_plugins:
            click.echo("Setting up default strategies...")
            
            # Moving Average Strategy
            ma_strategy = MovingAverageStrategy()
            ma_strategy.initialize({'short_window': 20, 'long_window': 50})
            sim_engine.register_strategy(ma_strategy, "MovingAverage")
            
            # Mean Reversion Strategy  
            mr_strategy = MeanReversionStrategy()
            mr_strategy.initialize({'lookback_window': 50, 'entry_threshold': 2.0})
            sim_engine.register_strategy(mr_strategy, "MeanReversion")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Run simulation
        click.echo(f"Running {scenarios} scenarios with {workers} workers...")
        with click.progressbar(length=scenarios, label='Simulations') as bar:
            def progress_callback(completed):
                bar.update(completed - bar.pos)
            
            results = sim_engine.run_simulation(max_workers=workers)
        
        # Save results
        results_file = Path(output_path) / f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        sim_engine.save_results(results_file)
        
        # Print summary
        click.echo("\n" + "="*60)
        click.echo("SIMULATION COMPLETE")
        click.echo("="*60)
        
        if results:
            successful = [r for r in results if r.success]
            failed = len(results) - len(successful)
            
            click.echo(f"Total simulations: {len(results)}")
            click.echo(f"Successful: {len(successful)}")
            click.echo(f"Failed: {failed}")
            click.echo(f"Success rate: {len(successful)/len(results)*100:.1f}%")
            
            if successful:
                import numpy as np
                pnls = [r.final_pnl for r in successful]
                click.echo(f"\nPerformance Summary:")
                click.echo(f"  Mean PnL: ${np.mean(pnls):.2f}")
                click.echo(f"  Std PnL: ${np.std(pnls):.2f}")
                click.echo(f"  Best PnL: ${np.max(pnls):.2f}")
                click.echo(f"  Worst PnL: ${np.min(pnls):.2f}")
        
        click.echo(f"\nResults saved to: {results_file}")
        click.echo("="*60)
        
    except Exception as e:
        click.echo(f"❌ Simulation failed: {e}", err=True)
        logger.exception("Simulation error")
        sys.exit(1)


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file path')
@click.option('--scenarios', '-s', type=int, required=True,
              help='Number of simulation scenarios')
@click.option('--data-path', '-d', type=click.Path(exists=True), required=True,
              help='Market data file path')
@click.option('--output-path', '-o', type=click.Path(), required=True,
              help='Output directory for results')
@click.option('--job-name', type=str,
              help='Job name for cluster scheduler')
def run_distributed(config, scenarios, data_path, output_path, job_name):
    """Run distributed simulation on HPC cluster."""
    
    click.echo("🌐 Starting distributed HPC QuantSim simulation...")
    
    try:
        from .hpc import get_mpi_info, initialize_mpi
        from .config import load_config, create_default_config
        from . import create_simulation
        
        # Check MPI environment
        mpi_info = get_mpi_info()
        if not mpi_info['available']:
            click.echo(f"❌ MPI not available: {mpi_info['reason']}", err=True)
            sys.exit(1)
        
        click.echo(f"MPI initialized: rank {mpi_info['rank']}/{mpi_info['size']}")
        
        # Initialize MPI collectives
        mpi_collectives = initialize_mpi()
        if not mpi_collectives:
            click.echo("❌ Failed to initialize MPI collectives", err=True)
            sys.exit(1)
        
        # Load configuration (broadcast from root)
        if mpi_collectives.is_root:
            if config:
                config_obj = load_config(config)
            else:
                config_obj = create_default_config()
            
            # Override with command line options
            config_obj.simulation.num_simulations = scenarios
            config_obj.hpc.use_mpi = True
        else:
            config_obj = None
        
        # Broadcast configuration to all ranks
        config_obj = mpi_collectives.broadcast_config(config_obj)
        
        # Create simulation engine
        sim_engine = create_simulation(config=config_obj)
        sim_engine.mpi_comm = mpi_collectives.comm
        sim_engine.mpi_rank = mpi_collectives.rank
        sim_engine.mpi_size = mpi_collectives.size
        
        # Load market data (all ranks)
        sim_engine.load_market_data(data_path)
        
        # Generate scenarios (on root) and scatter
        if mpi_collectives.is_root:
            click.echo(f"Generating {scenarios} scenarios...")
            all_scenarios = sim_engine.generate_scenarios()
        else:
            all_scenarios = None
        
        # Distribute scenarios across ranks
        local_scenarios = mpi_collectives.scatter_scenarios(all_scenarios)
        
        click.echo(f"Rank {mpi_collectives.rank}: Processing {len(local_scenarios)} scenarios")
        
        # Run local simulations
        local_results = []
        for scenario in local_scenarios:
            for strategy_name, strategy in sim_engine.strategies.items():
                result = sim_engine._run_single_scenario(scenario, strategy_name, strategy)
                local_results.append(result)
        
        # Gather results at root
        all_results = mpi_collectives.gather_results(local_results)
        
        # Aggregate metrics across ranks
        local_metrics = {
            'completed_scenarios': len(local_scenarios),
            'successful_runs': len([r for r in local_results if r.success]),
            'total_pnl': sum(r.final_pnl for r in local_results if r.success),
        }
        
        global_metrics = mpi_collectives.allreduce_metrics(local_metrics, 'sum')
        
        if mpi_collectives.is_root:
            click.echo("\n" + "="*60)
            click.echo("DISTRIBUTED SIMULATION COMPLETE")
            click.echo("="*60)
            
            click.echo(f"Total scenarios processed: {global_metrics['completed_scenarios']}")
            click.echo(f"Successful runs: {global_metrics['successful_runs']}")
            click.echo(f"Combined PnL: ${global_metrics['total_pnl']:.2f}")
            
            # Save results
            if all_results:
                os.makedirs(output_path, exist_ok=True)
                results_file = Path(output_path) / f"distributed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                # Convert results to DataFrame and save
                import pandas as pd
                data = []
                for result in all_results:
                    data.append({
                        'scenario_id': result.scenario_id,
                        'strategy_name': result.strategy_name,
                        'final_pnl': result.final_pnl,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'total_trades': result.total_trades,
                        'success': result.success,
                    })
                
                df = pd.DataFrame(data)
                df.to_csv(results_file, index=False)
                
                click.echo(f"Results saved to: {results_file}")
            
            # Performance statistics
            perf_stats = mpi_collectives.get_performance_stats()
            click.echo(f"\nMPI Performance:")
            for op, stats in perf_stats.items():
                if isinstance(stats, dict) and 'count' in stats:
                    click.echo(f"  {op}: {stats['count']} ops, {stats['avg_time_ms']:.1f}ms avg")
            
        click.echo("="*60)
        
    except Exception as e:
        click.echo(f"❌ Distributed simulation failed: {e}", err=True)
        logger.exception("Distributed simulation error")
        sys.exit(1)


@main.command()
@click.option('--scenarios', type=int, default=1000,
              help='Number of scenarios for cluster job')
@click.option('--data-path', '-d', type=click.Path(exists=True), required=True,
              help='Market data file path')
@click.option('--output-path', '-o', type=click.Path(), required=True,
              help='Output directory for results')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file path')
@click.option('--job-name', type=str,
              help='Custom job name')
def submit_cluster(scenarios, data_path, output_path, config, job_name):
    """Submit simulation job to HPC cluster."""
    
    try:
        from .hpc.cluster_manager import get_cluster_manager
        from .config import load_config, create_default_config
        
        click.echo("📋 Submitting job to HPC cluster...")
        
        # Load configuration
        if config:
            config_obj = load_config(config)
        else:
            config_obj = create_default_config()
        
        # Create cluster manager
        cluster_manager = get_cluster_manager(config_obj.deployment)
        if not cluster_manager:
            click.echo("❌ Failed to initialize cluster manager", err=True)
            sys.exit(1)
        
        # Check cluster availability
        cluster_info = cluster_manager.get_cluster_info()
        if not cluster_info['available']:
            click.echo(f"❌ Cluster not available: {cluster_info.get('error', 'Unknown error')}", err=True)
            sys.exit(1)
        
        click.echo(f"Cluster info: {cluster_info['total_nodes']} nodes, {cluster_info['scheduler']} scheduler")
        
        # Submit job
        strategy_configs = [
            {'type': 'moving_average', 'short_window': 20, 'long_window': 50},
            {'type': 'mean_reversion', 'lookback_window': 50, 'entry_threshold': 2.0}
        ]
        
        job_id = cluster_manager.submit_simulation_job(
            num_scenarios=scenarios,
            strategy_configs=strategy_configs,
            market_data_path=str(data_path),
            output_path=str(output_path),
            job_name=job_name
        )
        
        if job_id:
            click.echo(f"✅ Job submitted successfully: {job_id}")
            click.echo(f"Monitor with: hpc-quantsim status {job_id}")
        else:
            click.echo("❌ Job submission failed", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Cluster submission failed: {e}", err=True)
        logger.exception("Cluster submission error")
        sys.exit(1)


@main.command()
@click.argument('job_id', required=False)
def status(job_id):
    """Check status of cluster jobs."""
    
    try:
        from .hpc.cluster_manager import get_cluster_manager
        from .config import create_default_config
        
        config = create_default_config()
        cluster_manager = get_cluster_manager(config.deployment)
        
        if not cluster_manager:
            click.echo("❌ Failed to initialize cluster manager", err=True)
            sys.exit(1)
        
        if job_id:
            # Check specific job
            job_status = cluster_manager.get_job_status(job_id)
            click.echo(f"Job {job_id} status: {job_status}")
        else:
            # List all active jobs
            active_jobs = cluster_manager.list_active_jobs()
            
            if not active_jobs:
                click.echo("No active jobs found.")
                return
            
            click.echo("Active Jobs:")
            click.echo("-" * 80)
            for job in active_jobs:
                click.echo(f"ID: {job.get('name', 'unknown')}")
                click.echo(f"  Status: {job.get('status', 'unknown')}")
                click.echo(f"  Scenarios: {job.get('scenarios', 0)}")
                click.echo(f"  Submitted: {job.get('submitted_at', 'unknown')}")
                click.echo()
                
    except Exception as e:
        click.echo(f"❌ Status check failed: {e}", err=True)
        sys.exit(1)


@main.command()
def info():
    """Display system information and capabilities."""
    
    click.echo("HPC QuantSim System Information")
    click.echo("=" * 50)
    
    # Print system capabilities
    from . import print_system_info, get_capabilities
    print_system_info()
    
    # GPU information
    try:
        from .gpu import check_gpu_availability, print_gpu_info
        if check_gpu_availability():
            click.echo("\nGPU Information:")
            click.echo("-" * 30)
            print_gpu_info()
        else:
            click.echo("\nGPU: Not available")
    except ImportError:
        click.echo("\nGPU: CUDA libraries not installed")
    
    # MPI information
    try:
        from .hpc import get_mpi_info
        mpi_info = get_mpi_info()
        click.echo(f"\nMPI Information:")
        click.echo("-" * 30)
        if mpi_info['available']:
            click.echo(f"Available: Yes")
            click.echo(f"Version: {mpi_info.get('mpi_version', 'unknown')}")
            click.echo(f"Processes: {mpi_info.get('size', 1)}")
            click.echo(f"Current rank: {mpi_info.get('rank', 0)}")
        else:
            click.echo(f"Available: No ({mpi_info.get('reason', 'unknown')})")
    except ImportError:
        click.echo("\nMPI: Not available")
    
    # Cluster information
    try:
        from .hpc.cluster_manager import get_cluster_manager
        from .config import create_default_config
        
        config = create_default_config()
        cluster_manager = get_cluster_manager(config.deployment)
        
        if cluster_manager:
            cluster_info = cluster_manager.get_cluster_info()
            click.echo(f"\nCluster Information:")
            click.echo("-" * 30)
            click.echo(f"Scheduler: {cluster_info['scheduler']}")
            click.echo(f"Available: {cluster_info['available']}")
            if cluster_info['available']:
                click.echo(f"Total nodes: {cluster_info.get('total_nodes', 'unknown')}")
                click.echo(f"Available nodes: {cluster_info.get('available_nodes', 'unknown')}")
        else:
            click.echo("\nCluster: Not available")
            
    except Exception:
        click.echo("\nCluster: Information unavailable")


@main.command()
@click.option('--output', '-o', type=click.Path(), default='config.yaml',
              help='Output configuration file path')
def create_config(output):
    """Create a sample configuration file."""
    
    try:
        from .config import create_default_config, save_config
        
        config = create_default_config()
        
        # Customize with reasonable defaults
        config.simulation.num_simulations = 1000
        config.simulation.max_parallel = 50
        config.simulation.symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        
        # Save configuration
        save_config(config, output)
        
        click.echo(f"✅ Sample configuration created: {output}")
        click.echo("Edit the file to customize your simulation parameters.")
        
    except Exception as e:
        click.echo(f"❌ Failed to create configuration: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--benchmark-gpu/--no-benchmark-gpu', default=False,
              help='Run GPU benchmark')
@click.option('--benchmark-mpi/--no-benchmark-mpi', default=False, 
              help='Run MPI benchmark')
def benchmark(benchmark_gpu, benchmark_mpi):
    """Run system performance benchmarks."""
    
    click.echo("🔥 Running HPC QuantSim benchmarks...")
    
    if benchmark_gpu:
        try:
            from .gpu import check_gpu_availability, benchmark_all_devices
            if check_gpu_availability():
                click.echo("\nGPU Benchmark:")
                click.echo("=" * 30)
                benchmark_all_devices()
            else:
                click.echo("GPU benchmark skipped: CUDA not available")
        except ImportError:
            click.echo("GPU benchmark skipped: CUDA libraries not installed")
    
    if benchmark_mpi:
        try:
            from .hpc import initialize_mpi
            mpi_collectives = initialize_mpi()
            
            if mpi_collectives:
                click.echo("\nMPI Benchmark:")
                click.echo("=" * 30)
                
                # Simple MPI performance test
                import numpy as np
                import time
                
                # Test allreduce performance
                data_sizes = [1000, 10000, 100000]
                for size in data_sizes:
                    test_data = np.random.random(size).astype(np.float64)
                    
                    start_time = time.time()
                    result = mpi_collectives.allreduce_arrays(test_data, 'sum')
                    end_time = time.time()
                    
                    bandwidth = (size * 8 * 2) / (end_time - start_time) / (1024**2)  # MB/s
                    click.echo(f"AllReduce {size} elements: {(end_time-start_time)*1000:.1f}ms, {bandwidth:.1f} MB/s")
                
                # Print performance stats
                stats = mpi_collectives.get_performance_stats()
                click.echo(f"MPI Performance Summary: {stats}")
            else:
                click.echo("MPI benchmark skipped: MPI not available")
                
        except ImportError:
            click.echo("MPI benchmark skipped: MPI4PY not installed")
    
    if not benchmark_gpu and not benchmark_mpi:
        click.echo("Use --benchmark-gpu or --benchmark-mpi to run specific benchmarks")


if __name__ == '__main__':
    main()
