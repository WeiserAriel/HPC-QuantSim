"""
MPI collective operations for distributed HPC QuantSim computations.

Provides high-performance distributed computing primitives:
- AllReduce operations for metric aggregation
- Scatter/Gather for data distribution
- Broadcast for configuration distribution
- Custom collectives for financial computations
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pickle
import time

# Try to import MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


class MPICollectives:
    """
    High-performance MPI collective operations for distributed quantitative computing.
    
    Features:
    - Optimized collective operations for financial data
    - Custom reduction operations for metrics
    - Memory-efficient data distribution
    - Error handling and fault tolerance
    - Performance monitoring
    """
    
    def __init__(self, comm=None):
        """Initialize MPI collectives."""
        if not HAS_MPI:
            raise ImportError("MPI4PY not available")
        
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.is_root = (self.rank == 0)
        
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.operation_stats = {
            'allreduce': {'count': 0, 'total_time': 0.0, 'total_bytes': 0},
            'broadcast': {'count': 0, 'total_time': 0.0, 'total_bytes': 0},
            'scatter': {'count': 0, 'total_time': 0.0, 'total_bytes': 0},
            'gather': {'count': 0, 'total_time': 0.0, 'total_bytes': 0},
            'reduce': {'count': 0, 'total_time': 0.0, 'total_bytes': 0},
        }
        
        # Custom MPI operations
        self._setup_custom_operations()
        
        self.logger.info(f"MPI Collectives initialized: rank {self.rank}/{self.size}")
    
    def _setup_custom_operations(self):
        """Setup custom MPI reduction operations for financial data."""
        # Sharpe ratio aggregation (numerator and denominator separately)
        def sharpe_reduce_func(invec, inoutvec, datatype):
            """Custom reduction for Sharpe ratio calculation."""
            # Input format: [sum_returns, sum_squared_returns, count]
            for i in range(0, len(invec), 3):
                # Combine statistics
                inoutvec[i] += invec[i]         # sum_returns
                inoutvec[i+1] += invec[i+1]     # sum_squared_returns  
                inoutvec[i+2] += invec[i+2]     # count
        
        self.sharpe_op = MPI.Op.Create(sharpe_reduce_func, commute=True)
        
        # Portfolio statistics aggregation
        def portfolio_reduce_func(invec, inoutvec, datatype):
            """Custom reduction for portfolio statistics."""
            # Input format: [total_pnl, total_volume, max_drawdown, min_drawdown, trade_count]
            for i in range(0, len(invec), 5):
                inoutvec[i] += invec[i]                           # total_pnl
                inoutvec[i+1] += invec[i+1]                       # total_volume
                inoutvec[i+2] = max(inoutvec[i+2], invec[i+2])    # max_drawdown
                inoutvec[i+3] = min(inoutvec[i+3], invec[i+3])    # min_drawdown
                inoutvec[i+4] += invec[i+4]                       # trade_count
        
        self.portfolio_op = MPI.Op.Create(portfolio_reduce_func, commute=True)
    
    def allreduce_metrics(self, local_metrics: Dict[str, float], 
                         operation: str = 'sum') -> Dict[str, float]:
        """
        Perform AllReduce operation on performance metrics.
        
        Args:
            local_metrics: Local metrics dictionary
            operation: Reduction operation ('sum', 'mean', 'max', 'min')
            
        Returns:
            Globally reduced metrics
        """
        if self.size == 1:
            return local_metrics
        
        start_time = time.time()
        
        try:
            # Convert metrics to arrays for MPI operations
            metric_names = sorted(local_metrics.keys())
            local_values = np.array([local_metrics.get(name, 0.0) for name in metric_names], 
                                  dtype=np.float64)
            
            # Perform reduction based on operation type
            mpi_op = {
                'sum': MPI.SUM,
                'max': MPI.MAX,
                'min': MPI.MIN,
                'mean': MPI.SUM,  # Will divide by size later
            }.get(operation, MPI.SUM)
            
            global_values = np.zeros_like(local_values)
            self.comm.Allreduce(local_values, global_values, op=mpi_op)
            
            # Handle mean operation
            if operation == 'mean':
                global_values /= self.size
            
            # Convert back to dictionary
            global_metrics = {name: float(value) for name, value in zip(metric_names, global_values)}
            
            # Update statistics
            elapsed_time = time.time() - start_time
            self.operation_stats['allreduce']['count'] += 1
            self.operation_stats['allreduce']['total_time'] += elapsed_time
            self.operation_stats['allreduce']['total_bytes'] += local_values.nbytes * 2  # send + receive
            
            return global_metrics
            
        except Exception as e:
            self.logger.error(f"AllReduce metrics failed: {e}")
            return local_metrics
    
    def allreduce_arrays(self, local_array: np.ndarray, 
                        operation: str = 'sum') -> np.ndarray:
        """
        Perform AllReduce operation on numpy arrays.
        
        Args:
            local_array: Local array data
            operation: Reduction operation
            
        Returns:
            Globally reduced array
        """
        if self.size == 1:
            return local_array
        
        start_time = time.time()
        
        try:
            mpi_op = {
                'sum': MPI.SUM,
                'max': MPI.MAX,
                'min': MPI.MIN,
                'mean': MPI.SUM,
            }.get(operation, MPI.SUM)
            
            global_array = np.zeros_like(local_array)
            self.comm.Allreduce(local_array, global_array, op=mpi_op)
            
            if operation == 'mean':
                global_array /= self.size
            
            # Update statistics
            elapsed_time = time.time() - start_time
            self.operation_stats['allreduce']['count'] += 1
            self.operation_stats['allreduce']['total_time'] += elapsed_time
            self.operation_stats['allreduce']['total_bytes'] += local_array.nbytes * 2
            
            return global_array
            
        except Exception as e:
            self.logger.error(f"AllReduce arrays failed: {e}")
            return local_array
    
    def reduce_portfolio_stats(self, local_stats: Dict[str, float]) -> Dict[str, float]:
        """
        Reduce portfolio statistics using custom reduction operation.
        
        Args:
            local_stats: Local portfolio statistics
            
        Returns:
            Global portfolio statistics (only valid on root)
        """
        if self.size == 1:
            return local_stats
        
        start_time = time.time()
        
        try:
            # Pack statistics into array format expected by custom operation
            local_data = np.array([
                local_stats.get('total_pnl', 0.0),
                local_stats.get('total_volume', 0.0),
                local_stats.get('max_drawdown', 0.0),
                local_stats.get('min_drawdown', 0.0),
                local_stats.get('trade_count', 0),
            ], dtype=np.float64)
            
            global_data = np.zeros_like(local_data)
            
            # Use custom portfolio reduction operation
            self.comm.Allreduce(local_data, global_data, op=self.portfolio_op)
            
            global_stats = {
                'total_pnl': float(global_data[0]),
                'total_volume': float(global_data[1]),
                'max_drawdown': float(global_data[2]),
                'min_drawdown': float(global_data[3]),
                'trade_count': int(global_data[4]),
            }
            
            # Update statistics
            elapsed_time = time.time() - start_time
            self.operation_stats['allreduce']['count'] += 1
            self.operation_stats['allreduce']['total_time'] += elapsed_time
            self.operation_stats['allreduce']['total_bytes'] += local_data.nbytes * 2
            
            return global_stats
            
        except Exception as e:
            self.logger.error(f"Reduce portfolio stats failed: {e}")
            return local_stats
    
    def broadcast_config(self, config_data: Any) -> Any:
        """
        Broadcast configuration data from root to all processes.
        
        Args:
            config_data: Configuration data (on root), ignored on other ranks
            
        Returns:
            Broadcasted configuration data
        """
        start_time = time.time()
        
        try:
            # Serialize data for transmission
            if self.is_root:
                serialized_data = pickle.dumps(config_data)
                data_size = len(serialized_data)
            else:
                data_size = None
                serialized_data = None
            
            # First broadcast the size
            data_size = self.comm.bcast(data_size, root=0)
            
            # Create buffer on non-root processes
            if not self.is_root:
                serialized_data = bytearray(data_size)
            
            # Broadcast the actual data
            self.comm.Bcast(serialized_data, root=0)
            
            # Deserialize
            config_data = pickle.loads(serialized_data)
            
            # Update statistics
            elapsed_time = time.time() - start_time
            self.operation_stats['broadcast']['count'] += 1
            self.operation_stats['broadcast']['total_time'] += elapsed_time
            self.operation_stats['broadcast']['total_bytes'] += data_size
            
            return config_data
            
        except Exception as e:
            self.logger.error(f"Broadcast config failed: {e}")
            return config_data if self.is_root else None
    
    def scatter_scenarios(self, scenario_list: Optional[List] = None) -> List:
        """
        Scatter simulation scenarios from root to all processes.
        
        Args:
            scenario_list: List of scenarios (on root), ignored on other ranks
            
        Returns:
            Local scenarios for this process
        """
        start_time = time.time()
        
        try:
            if self.is_root:
                if scenario_list is None:
                    scenario_list = []
                
                # Distribute scenarios evenly across processes
                scenarios_per_rank = len(scenario_list) // self.size
                remainder = len(scenario_list) % self.size
                
                scattered_scenarios = []
                start_idx = 0
                
                for rank in range(self.size):
                    # Give extra scenarios to first 'remainder' processes
                    rank_scenarios = scenarios_per_rank + (1 if rank < remainder else 0)
                    end_idx = start_idx + rank_scenarios
                    
                    rank_data = scenario_list[start_idx:end_idx]
                    scattered_scenarios.append(rank_data)
                    start_idx = end_idx
            else:
                scattered_scenarios = None
            
            # Scatter the scenario lists
            local_scenarios = self.comm.scatter(scattered_scenarios, root=0)
            
            # Update statistics
            elapsed_time = time.time() - start_time
            self.operation_stats['scatter']['count'] += 1
            self.operation_stats['scatter']['total_time'] += elapsed_time
            if self.is_root and scenario_list:
                self.operation_stats['scatter']['total_bytes'] += len(pickle.dumps(scenario_list))
            
            return local_scenarios or []
            
        except Exception as e:
            self.logger.error(f"Scatter scenarios failed: {e}")
            return []
    
    def gather_results(self, local_results: List) -> Optional[List]:
        """
        Gather simulation results from all processes to root.
        
        Args:
            local_results: Results from local process
            
        Returns:
            All results combined (only valid on root)
        """
        start_time = time.time()
        
        try:
            all_results = self.comm.gather(local_results, root=0)
            
            if self.is_root:
                # Flatten results from all processes
                combined_results = []
                for rank_results in all_results:
                    if rank_results:
                        combined_results.extend(rank_results)
                
                # Update statistics
                elapsed_time = time.time() - start_time
                self.operation_stats['gather']['count'] += 1
                self.operation_stats['gather']['total_time'] += elapsed_time
                self.operation_stats['gather']['total_bytes'] += len(pickle.dumps(combined_results))
                
                return combined_results
            
            return None
            
        except Exception as e:
            self.logger.error(f"Gather results failed: {e}")
            return local_results if self.is_root else None
    
    def barrier(self, timeout: Optional[float] = None):
        """
        Synchronization barrier for all processes.
        
        Args:
            timeout: Optional timeout in seconds
        """
        try:
            if timeout:
                # MPI doesn't support timeout directly, so we implement a simple check
                start_time = time.time()
                
            self.comm.Barrier()
            
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(f"Barrier timeout exceeded: {timeout}s")
                
        except Exception as e:
            self.logger.error(f"Barrier failed: {e}")
    
    def reduce_percentiles(self, local_data: np.ndarray, 
                          percentiles: List[float] = [5, 25, 50, 75, 95]) -> Dict[str, float]:
        """
        Calculate percentiles across all processes.
        
        Args:
            local_data: Local data array
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary of percentile values (only valid on root)
        """
        if self.size == 1:
            return {f'p{p}': np.percentile(local_data, p) for p in percentiles}
        
        try:
            # Gather all data to root
            all_data = self.comm.gather(local_data, root=0)
            
            if self.is_root:
                # Combine all data
                combined_data = np.concatenate(all_data)
                
                # Calculate percentiles
                result = {}
                for p in percentiles:
                    result[f'p{p}'] = np.percentile(combined_data, p)
                
                return result
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Reduce percentiles failed: {e}")
            return {}
    
    def allgather_metadata(self, local_metadata: Dict) -> List[Dict]:
        """
        All-gather metadata from all processes.
        
        Args:
            local_metadata: Local process metadata
            
        Returns:
            List of metadata from all processes
        """
        try:
            all_metadata = self.comm.allgather(local_metadata)
            return all_metadata
            
        except Exception as e:
            self.logger.error(f"AllGather metadata failed: {e}")
            return [local_metadata]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get MPI collective operation performance statistics."""
        stats = {}
        
        for op_name, op_stats in self.operation_stats.items():
            if op_stats['count'] > 0:
                avg_time = op_stats['total_time'] / op_stats['count']
                avg_bandwidth = (op_stats['total_bytes'] / op_stats['total_time'] / (1024**2) 
                               if op_stats['total_time'] > 0 else 0)
                
                stats[op_name] = {
                    'count': op_stats['count'],
                    'total_time_ms': op_stats['total_time'] * 1000,
                    'avg_time_ms': avg_time * 1000,
                    'total_bytes': op_stats['total_bytes'],
                    'avg_bandwidth_mb_s': avg_bandwidth,
                }
        
        stats['mpi_info'] = {
            'rank': self.rank,
            'size': self.size,
            'is_root': self.is_root,
        }
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        for op_stats in self.operation_stats.values():
            op_stats.update({'count': 0, 'total_time': 0.0, 'total_bytes': 0})
    
    def __del__(self):
        """Cleanup MPI resources."""
        try:
            if hasattr(self, 'sharpe_op'):
                self.sharpe_op.Free()
            if hasattr(self, 'portfolio_op'):
                self.portfolio_op.Free()
        except Exception:
            pass


def get_mpi_info() -> Dict[str, Any]:
    """Get MPI environment information."""
    if not HAS_MPI:
        return {'available': False, 'reason': 'MPI4PY not installed'}
    
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        processor_name = MPI.Get_processor_name()
        
        return {
            'available': True,
            'rank': rank,
            'size': size,
            'processor_name': processor_name,
            'mpi_version': '.'.join(map(str, MPI.Get_version())),
        }
        
    except Exception as e:
        return {'available': False, 'reason': str(e)}


def initialize_mpi() -> Optional[MPICollectives]:
    """Initialize MPI collectives if available."""
    if not HAS_MPI:
        return None
    
    try:
        return MPICollectives()
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to initialize MPI: {e}")
        return None


if __name__ == "__main__":
    # Simple MPI test
    mpi_info = get_mpi_info()
    print(f"MPI Info: {mpi_info}")
    
    if mpi_info['available']:
        collectives = initialize_mpi()
        if collectives:
            print(f"MPI Collectives initialized on rank {collectives.rank}/{collectives.size}")
            
            # Simple test
            local_data = {'test_metric': collectives.rank * 10}
            global_data = collectives.allreduce_metrics(local_data, 'sum')
            
            if collectives.is_root:
                print(f"Test AllReduce result: {global_data}")
                print(f"Performance stats: {collectives.get_performance_stats()}")
        else:
            print("Failed to initialize MPI collectives")
    else:
        print("MPI not available")
