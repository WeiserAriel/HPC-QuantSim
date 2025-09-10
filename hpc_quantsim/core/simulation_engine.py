"""
Main simulation engine for HPC QuantSim.

The SimulationEngine orchestrates the entire simulation process:
- Loading market data and strategies
- Managing parallel execution across multiple scenarios
- Coordinating with HPC components (MPI, GPU, collectives)
- Collecting and aggregating metrics
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

from ..config import Config, SimulationConfig
from ..market import MarketReplay, TickData
from ..metrics import MetricAggregator, PerformanceMetrics
from .strategy_interface import Strategy, StrategyResult
from .order_book import OrderBook


@dataclass
class SimulationScenario:
    """Individual simulation scenario configuration."""
    scenario_id: int
    random_seed: int
    market_noise_level: float = 0.0
    execution_delay_ms: float = 0.0
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None

@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    scenario_id: int
    strategy_name: str
    execution_time_ms: float
    final_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    metrics: PerformanceMetrics
    success: bool = True
    error_message: Optional[str] = None

class SimulationEngine:
    """
    High-performance simulation engine for market replay and strategy testing.
    
    Features:
    - Parallel execution of multiple simulation scenarios
    - Integration with HPC-X components (MPI, GPU, UCX)
    - Real-time metrics collection and aggregation
    - Plugin-based strategy system
    - Tick-level market replay with microstructure simulation
    """
    
    def __init__(self, config: Union[Config, SimulationConfig, dict]):
        """Initialize the simulation engine."""
        if isinstance(config, dict):
            self.config = SimulationConfig(**config)
        elif isinstance(config, Config):
            self.config = config.simulation
            self.full_config = config
        else:
            self.config = config
            self.full_config = None
        
        # Core components
        self.market_replay: Optional[MarketReplay] = None
        self.strategies: Dict[str, Strategy] = {}
        self.metric_aggregator: Optional[MetricAggregator] = None
        self.order_book: Optional[OrderBook] = None
        
        # Execution state
        self.is_running = False
        self.current_scenarios: List[SimulationScenario] = []
        self.results: List[SimulationResult] = []
        
        # HPC components (initialized if available)
        self._init_hpc_components()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _init_hpc_components(self):
        """Initialize HPC components if available and configured."""
        self.mpi_comm = None
        self.mpi_rank = 0
        self.mpi_size = 1
        self.gpu_available = False
        
        # Try to initialize MPI
        try:
            if hasattr(self, 'full_config') and self.full_config and self.full_config.hpc.use_mpi:
                from mpi4py import MPI
                self.mpi_comm = MPI.COMM_WORLD
                self.mpi_rank = self.mpi_comm.Get_rank()
                self.mpi_size = self.mpi_comm.Get_size()
                self.logger.info(f"MPI initialized: rank {self.mpi_rank}/{self.mpi_size}")
        except ImportError:
            self.logger.warning("MPI not available")
        
        # Try to initialize GPU
        try:
            if hasattr(self, 'full_config') and self.full_config and self.full_config.hpc.use_gpu:
                import cupy as cp
                self.gpu_available = True
                self.logger.info(f"GPU available: {cp.cuda.Device().id}")
        except ImportError:
            self.logger.warning("CUDA/CuPy not available")
    
    def load_market_data(self, data_path: Union[str, Path], 
                        symbols: Optional[List[str]] = None) -> None:
        """Load historical market data for simulation."""
        from ..market import MarketReplay
        
        self.market_replay = MarketReplay()
        self.market_replay.load_data(data_path, symbols=symbols or self.config.symbols)
        
        # Initialize order book with first symbol
        if self.config.symbols:
            self.order_book = OrderBook(symbol=self.config.symbols[0])
        
        self.logger.info(f"Loaded market data from {data_path}")
    
    def register_strategy(self, strategy: Strategy, name: Optional[str] = None) -> None:
        """Register a trading strategy for simulation."""
        strategy_name = name or strategy.__class__.__name__
        self.strategies[strategy_name] = strategy
        self.logger.info(f"Registered strategy: {strategy_name}")
    
    def load_strategy_plugin(self, plugin_path: Union[str, Path], 
                           strategy_name: str) -> None:
        """Load a strategy from a Python plugin file."""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(strategy_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for Strategy class in the module
        strategy_class = getattr(module, 'Strategy', None)
        if not strategy_class:
            # Try to find any class that inherits from our Strategy base
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, Strategy) and 
                    attr != Strategy):
                    strategy_class = attr
                    break
        
        if not strategy_class:
            raise ValueError(f"No Strategy class found in {plugin_path}")
        
        strategy = strategy_class()
        self.register_strategy(strategy, strategy_name)
    
    def generate_scenarios(self) -> List[SimulationScenario]:
        """Generate simulation scenarios with parameter variations."""
        scenarios = []
        
        base_seed = self.config.random_seed
        num_sims = self.config.num_simulations
        
        # Generate parameter combinations
        for i in range(num_sims):
            scenario = SimulationScenario(
                scenario_id=i,
                random_seed=base_seed + i,
                market_noise_level=np.random.uniform(0.0, 0.05),  # 0-5% noise
                execution_delay_ms=np.random.uniform(0.1, 10.0),  # 0.1-10ms delay
                strategy_params=self._generate_strategy_params(i),
                start_time=self.config.start_time,
                end_time=self.config.end_time
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_strategy_params(self, scenario_id: int) -> Dict[str, Any]:
        """Generate randomized strategy parameters for scenario."""
        # This can be overridden or configured per strategy
        np.random.seed(self.config.random_seed + scenario_id)
        
        return {
            'lookback_window': np.random.randint(10, 100),
            'volatility_threshold': np.random.uniform(0.01, 0.1),
            'position_size': np.random.uniform(0.01, 0.1),
            'stop_loss_pct': np.random.uniform(0.02, 0.05),
            'take_profit_pct': np.random.uniform(0.03, 0.08),
        }
    
    def run_simulation(self, max_workers: Optional[int] = None) -> List[SimulationResult]:
        """Run the complete simulation with all scenarios."""
        if not self.market_replay:
            raise ValueError("Market data not loaded. Call load_market_data() first.")
        
        if not self.strategies:
            raise ValueError("No strategies registered. Call register_strategy() first.")
        
        self.is_running = True
        start_time = time.time()
        
        # Generate scenarios
        scenarios = self.generate_scenarios()
        self.current_scenarios = scenarios
        
        # Initialize metrics aggregator
        from ..metrics import MetricAggregator
        self.metric_aggregator = MetricAggregator()
        
        self.logger.info(f"Starting {len(scenarios)} simulations across {len(self.strategies)} strategies")
        
        # Determine execution mode
        max_workers = max_workers or min(self.config.max_parallel, len(scenarios))
        
        if self.mpi_size > 1:
            # MPI parallel execution
            results = self._run_mpi_parallel(scenarios)
        elif self.gpu_available and len(scenarios) > 100:
            # GPU batch execution for large scenario counts
            results = self._run_gpu_batch(scenarios)
        else:
            # Standard multiprocessing
            results = self._run_multiprocess(scenarios, max_workers)
        
        self.results = results
        execution_time = time.time() - start_time
        
        self.logger.info(f"Simulation completed in {execution_time:.2f}s")
        self.logger.info(f"Successful runs: {sum(1 for r in results if r.success)}/{len(results)}")
        
        self.is_running = False
        return results
    
    def _run_multiprocess(self, scenarios: List[SimulationScenario], 
                         max_workers: int) -> List[SimulationResult]:
        """Run scenarios using multiprocessing."""
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scenario/strategy combinations
            futures = []
            for scenario in scenarios:
                for strategy_name, strategy in self.strategies.items():
                    future = executor.submit(
                        self._run_single_scenario, 
                        scenario, strategy_name, strategy
                    )
                    futures.append(future)
            
            # Collect results as they complete
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Simulation failed: {e}")
                    # Create failed result
                    results.append(SimulationResult(
                        scenario_id=-1,
                        strategy_name="unknown",
                        execution_time_ms=0,
                        final_pnl=0,
                        sharpe_ratio=0,
                        max_drawdown=0,
                        total_trades=0,
                        metrics=PerformanceMetrics(),
                        success=False,
                        error_message=str(e)
                    ))
        
        return results
    
    def _run_mpi_parallel(self, scenarios: List[SimulationScenario]) -> List[SimulationResult]:
        """Run scenarios using MPI parallel execution."""
        # Distribute scenarios across MPI ranks
        local_scenarios = scenarios[self.mpi_rank::self.mpi_size]
        
        # Run local scenarios
        local_results = []
        for scenario in local_scenarios:
            for strategy_name, strategy in self.strategies.items():
                result = self._run_single_scenario(scenario, strategy_name, strategy)
                local_results.append(result)
        
        # Gather all results at rank 0
        if self.mpi_comm:
            all_results = self.mpi_comm.gather(local_results, root=0)
            if self.mpi_rank == 0:
                # Flatten results
                results = []
                for rank_results in all_results:
                    results.extend(rank_results)
                return results
            else:
                return []
        
        return local_results
    
    def _run_gpu_batch(self, scenarios: List[SimulationScenario]) -> List[SimulationResult]:
        """Run scenarios using GPU batch processing."""
        # This is a placeholder for GPU implementation
        # In practice, this would use CUDA kernels for parallel execution
        self.logger.info("GPU batch execution not yet implemented, falling back to CPU")
        return self._run_multiprocess(scenarios, self.config.max_parallel)
    
    def _run_single_scenario(self, scenario: SimulationScenario, 
                           strategy_name: str, strategy: Strategy) -> SimulationResult:
        """Execute a single simulation scenario."""
        try:
            start_time = time.time()
            
            # Set random seed for reproducibility
            np.random.seed(scenario.random_seed)
            
            # Initialize strategy with scenario parameters
            strategy.initialize(scenario.strategy_params)
            
            # Create order book for this scenario
            order_book = OrderBook(symbol=self.config.symbols[0])
            
            # Initialize metrics collection
            metrics = PerformanceMetrics()
            
            # Run tick-by-tick simulation
            tick_count = 0
            total_pnl = 0.0
            position = 0.0
            cash = 100000.0  # Starting capital
            
            for tick in self.market_replay.iterate_ticks(
                start_time=scenario.start_time,
                end_time=scenario.end_time
            ):
                # Update order book
                order_book.update(tick)
                
                # Add market noise if configured
                if scenario.market_noise_level > 0:
                    tick = self._add_market_noise(tick, scenario.market_noise_level)
                
                # Get strategy decision
                strategy_result = strategy.on_tick(tick, order_book, position, cash)
                
                # Execute trades (with simulated delay)
                if strategy_result.orders:
                    time.sleep(scenario.execution_delay_ms / 1000.0)  # Simulate latency
                    
                    for order in strategy_result.orders:
                        # Simulate order execution
                        fill_price = order_book.get_mid_price()
                        
                        if order.side.value == "buy":
                            position += order.quantity
                            cash -= order.quantity * fill_price
                        else:
                            position -= order.quantity
                            cash += order.quantity * fill_price
                        
                        # Update metrics
                        metrics.record_trade(
                            timestamp=tick.timestamp,
                            price=fill_price,
                            quantity=order.quantity,
                            side=order.side.value
                        )
                
                # Update PnL
                current_price = order_book.get_mid_price()
                unrealized_pnl = position * current_price
                total_pnl = cash + unrealized_pnl - 100000.0
                
                metrics.update_pnl(total_pnl, tick.timestamp)
                
                tick_count += 1
                
                # Break if simulation time exceeded
                if tick_count > 100000:  # Limit for testing
                    break
            
            # Finalize metrics
            metrics.finalize()
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return SimulationResult(
                scenario_id=scenario.scenario_id,
                strategy_name=strategy_name,
                execution_time_ms=execution_time,
                final_pnl=total_pnl,
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown,
                total_trades=metrics.total_trades,
                metrics=metrics,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Scenario {scenario.scenario_id} failed: {e}")
            return SimulationResult(
                scenario_id=scenario.scenario_id,
                strategy_name=strategy_name,
                execution_time_ms=0,
                final_pnl=0,
                sharpe_ratio=0,
                max_drawdown=0,
                total_trades=0,
                metrics=PerformanceMetrics(),
                success=False,
                error_message=str(e)
            )
    
    def _add_market_noise(self, tick: TickData, noise_level: float) -> TickData:
        """Add market microstructure noise to tick data."""
        # Add random noise to bid/ask prices
        noise = np.random.normal(0, tick.price * noise_level)
        
        # Create modified tick (this would need proper TickData implementation)
        return tick  # Placeholder
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all simulation results."""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            return {"error": "No successful simulation runs"}
        
        pnls = [r.final_pnl for r in successful_results]
        sharpe_ratios = [r.sharpe_ratio for r in successful_results]
        drawdowns = [r.max_drawdown for r in successful_results]
        
        return {
            "total_simulations": len(self.results),
            "successful_simulations": len(successful_results),
            "success_rate": len(successful_results) / len(self.results),
            "pnl_statistics": {
                "mean": np.mean(pnls),
                "std": np.std(pnls),
                "min": np.min(pnls),
                "max": np.max(pnls),
                "percentiles": {
                    "5th": np.percentile(pnls, 5),
                    "25th": np.percentile(pnls, 25),
                    "50th": np.percentile(pnls, 50),
                    "75th": np.percentile(pnls, 75),
                    "95th": np.percentile(pnls, 95),
                }
            },
            "sharpe_statistics": {
                "mean": np.mean(sharpe_ratios),
                "std": np.std(sharpe_ratios),
                "min": np.min(sharpe_ratios),
                "max": np.max(sharpe_ratios),
            },
            "risk_statistics": {
                "mean_max_drawdown": np.mean(drawdowns),
                "worst_drawdown": np.min(drawdowns),  # Most negative
            }
        }
    
    def save_results(self, output_path: Union[str, Path]) -> None:
        """Save simulation results to file."""
        import pandas as pd
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            row = {
                'scenario_id': result.scenario_id,
                'strategy_name': result.strategy_name,
                'execution_time_ms': result.execution_time_ms,
                'final_pnl': result.final_pnl,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades,
                'success': result.success,
                'error_message': result.error_message
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save as parquet for efficiency
        output_path = Path(output_path)
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=False)
        
        self.logger.info(f"Results saved to {output_path}")
