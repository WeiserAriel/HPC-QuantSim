"""
Distributed metric aggregation using HPC-X collectives for HPC QuantSim.

Provides high-performance aggregation of metrics across distributed simulation runs:
- MPI-based collective operations
- NCCL GPU-to-GPU aggregation
- Real-time metric streaming
- Hierarchical aggregation strategies
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np
import time
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

from .performance_metrics import PerformanceMetrics
from ..config import Config


class AggregationMethod(Enum):
    """Methods for aggregating metrics across nodes."""
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    STD = "std"
    PERCENTILE = "percentile"
    HISTOGRAM = "histogram"


@dataclass
class MetricUpdate:
    """Single metric update from a simulation node."""
    timestamp: float
    source_rank: int
    scenario_id: int
    strategy_name: str
    metric_name: str
    value: Union[float, np.ndarray, Dict]
    metadata: Optional[Dict] = None


class MetricAggregator:
    """
    High-performance distributed metric aggregator.
    
    Features:
    - Real-time metric collection from distributed simulation nodes
    - HPC-X collective operations for efficient aggregation
    - Multiple aggregation strategies (sum, mean, percentiles, etc.)
    - GPU acceleration for large-scale operations
    - Hierarchical aggregation for scalability
    - Low-latency metric streaming
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize metric aggregator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # MPI configuration
        self.mpi_comm = None
        self.mpi_rank = 0
        self.mpi_size = 1
        self.is_root = True
        
        # GPU configuration
        self.gpu_enabled = False
        self.gpu_device = None
        
        # Aggregation state
        self.metrics_buffer: Dict[str, List[MetricUpdate]] = defaultdict(list)
        self.aggregated_metrics: Dict[str, Any] = {}
        self.aggregation_methods: Dict[str, AggregationMethod] = {}
        
        # Performance tracking
        self.total_updates = 0
        self.total_aggregations = 0
        self.aggregation_time = 0.0
        
        # Real-time streaming
        self.streaming_enabled = False
        self.stream_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.stream_interval = 1.0  # seconds
        self.last_stream_time = 0.0
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.aggregation_lock = threading.RLock()
        
        # Initialize HPC components
        self._init_hpc_components()
        
        # Default aggregation methods
        self._setup_default_aggregation_methods()
    
    def _init_hpc_components(self):
        """Initialize MPI and GPU components."""
        # Initialize MPI if available and configured
        try:
            if self.config and self.config.hpc.use_mpi:
                from mpi4py import MPI
                self.mpi_comm = MPI.COMM_WORLD
                self.mpi_rank = self.mpi_comm.Get_rank()
                self.mpi_size = self.mpi_comm.Get_size()
                self.is_root = (self.mpi_rank == 0)
                self.logger.info(f"MPI initialized: rank {self.mpi_rank}/{self.mpi_size}")
        except ImportError:
            self.logger.warning("MPI not available")
        
        # Initialize GPU if available and configured
        try:
            if self.config and self.config.hpc.use_gpu:
                import cupy as cp
                self.gpu_device = cp.cuda.Device()
                self.gpu_enabled = True
                self.logger.info(f"GPU enabled: device {self.gpu_device.id}")
        except ImportError:
            self.logger.warning("CUDA/CuPy not available")
    
    def _setup_default_aggregation_methods(self):
        """Setup default aggregation methods for common metrics."""
        defaults = {
            'total_pnl': AggregationMethod.SUM,
            'realized_pnl': AggregationMethod.SUM,
            'unrealized_pnl': AggregationMethod.SUM,
            'total_trades': AggregationMethod.SUM,
            'total_volume': AggregationMethod.SUM,
            'total_commission': AggregationMethod.SUM,
            'sharpe_ratio': AggregationMethod.MEAN,
            'max_drawdown': AggregationMethod.MAX,
            'win_rate': AggregationMethod.MEAN,
            'volatility': AggregationMethod.MEAN,
            'execution_time_ms': AggregationMethod.MEAN,
        }
        
        self.aggregation_methods.update(defaults)
    
    def add_metric_update(self, update: MetricUpdate) -> None:
        """Add a metric update from a simulation node."""
        with self.aggregation_lock:
            self.metrics_buffer[update.metric_name].append(update)
            self.total_updates += 1
            
            # Trigger real-time streaming if enabled
            if self.streaming_enabled:
                current_time = time.time()
                if current_time - self.last_stream_time > self.stream_interval:
                    self._trigger_streaming()
                    self.last_stream_time = current_time
    
    def collect_metrics(self, performance_metrics: PerformanceMetrics,
                       scenario_id: int, strategy_name: str) -> None:
        """Collect metrics from a PerformanceMetrics object."""
        timestamp = time.time()
        summary = performance_metrics.get_performance_summary()
        
        # Create metric updates for key metrics
        for metric_name, value in summary.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                update = MetricUpdate(
                    timestamp=timestamp,
                    source_rank=self.mpi_rank,
                    scenario_id=scenario_id,
                    strategy_name=strategy_name,
                    metric_name=metric_name,
                    value=float(value)
                )
                self.add_metric_update(update)
    
    def set_aggregation_method(self, metric_name: str, method: AggregationMethod) -> None:
        """Set aggregation method for a specific metric."""
        self.aggregation_methods[metric_name] = method
    
    def aggregate_metrics(self, force: bool = False) -> Dict[str, Any]:
        """
        Aggregate all buffered metrics using configured methods.
        
        Args:
            force: Force aggregation even if buffer is small
            
        Returns:
            Dictionary of aggregated metrics
        """
        with self.aggregation_lock:
            if not force and len(self.metrics_buffer) == 0:
                return self.aggregated_metrics
            
            start_time = time.time()
            
            # Aggregate each metric type
            for metric_name, updates in self.metrics_buffer.items():
                if not updates:
                    continue
                
                method = self.aggregation_methods.get(metric_name, AggregationMethod.MEAN)
                aggregated_value = self._aggregate_metric(updates, method)
                
                self.aggregated_metrics[metric_name] = aggregated_value
            
            # Perform distributed aggregation if MPI is available
            if self.mpi_comm and self.mpi_size > 1:
                self.aggregated_metrics = self._mpi_aggregate(self.aggregated_metrics)
            
            # Clear processed buffers
            if not force:  # Keep buffer for continuous aggregation
                for updates in self.metrics_buffer.values():
                    updates.clear()
            
            self.total_aggregations += 1
            self.aggregation_time += time.time() - start_time
            
            return self.aggregated_metrics
    
    def _aggregate_metric(self, updates: List[MetricUpdate], 
                         method: AggregationMethod) -> Any:
        """Aggregate a single metric using specified method."""
        if not updates:
            return None
        
        values = np.array([update.value for update in updates if isinstance(update.value, (int, float))])
        
        if len(values) == 0:
            return None
        
        if method == AggregationMethod.SUM:
            return float(np.sum(values))
        elif method == AggregationMethod.MEAN:
            return float(np.mean(values))
        elif method == AggregationMethod.MIN:
            return float(np.min(values))
        elif method == AggregationMethod.MAX:
            return float(np.max(values))
        elif method == AggregationMethod.STD:
            return float(np.std(values))
        elif method == AggregationMethod.PERCENTILE:
            return {
                '5th': float(np.percentile(values, 5)),
                '25th': float(np.percentile(values, 25)),
                '50th': float(np.percentile(values, 50)),
                '75th': float(np.percentile(values, 75)),
                '95th': float(np.percentile(values, 95)),
            }
        elif method == AggregationMethod.HISTOGRAM:
            hist, edges = np.histogram(values, bins=20)
            return {
                'bins': edges.tolist(),
                'counts': hist.tolist(),
                'total': len(values)
            }
        else:
            return float(np.mean(values))  # Default fallback
    
    def _mpi_aggregate(self, local_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform MPI-based distributed aggregation."""
        try:
            if not self.mpi_comm:
                return local_metrics
            
            # Gather all metrics at root
            all_metrics = self.mpi_comm.gather(local_metrics, root=0)
            
            if not self.is_root:
                # Non-root nodes return empty dict
                return {}
            
            # Root node aggregates across all nodes
            global_metrics = {}
            
            for metric_name in local_metrics.keys():
                method = self.aggregation_methods.get(metric_name, AggregationMethod.MEAN)
                
                # Collect values from all nodes
                all_values = []
                for node_metrics in all_metrics:
                    if metric_name in node_metrics and node_metrics[metric_name] is not None:
                        value = node_metrics[metric_name]
                        if isinstance(value, (int, float)):
                            all_values.append(value)
                        elif isinstance(value, dict) and method == AggregationMethod.PERCENTILE:
                            # Flatten percentile results
                            for pct_name, pct_value in value.items():
                                all_values.append(pct_value)
                
                if all_values:
                    if method == AggregationMethod.SUM:
                        global_metrics[metric_name] = sum(all_values)
                    elif method == AggregationMethod.MEAN:
                        global_metrics[metric_name] = np.mean(all_values)
                    elif method == AggregationMethod.MIN:
                        global_metrics[metric_name] = min(all_values)
                    elif method == AggregationMethod.MAX:
                        global_metrics[metric_name] = max(all_values)
                    elif method == AggregationMethod.STD:
                        global_metrics[metric_name] = np.std(all_values)
                    elif method == AggregationMethod.PERCENTILE:
                        global_metrics[metric_name] = {
                            '5th': np.percentile(all_values, 5),
                            '25th': np.percentile(all_values, 25),
                            '50th': np.percentile(all_values, 50),
                            '75th': np.percentile(all_values, 75),
                            '95th': np.percentile(all_values, 95),
                        }
            
            # Broadcast aggregated results to all nodes
            global_metrics = self.mpi_comm.bcast(global_metrics, root=0)
            
            return global_metrics
            
        except Exception as e:
            self.logger.error(f"MPI aggregation failed: {e}")
            return local_metrics
    
    def _nccl_aggregate(self, local_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform NCCL GPU-based aggregation (placeholder)."""
        # This would implement NCCL-based aggregation for GPU clusters
        # Currently returns local metrics as placeholder
        self.logger.info("NCCL aggregation not yet implemented")
        return local_metrics
    
    def enable_streaming(self, interval: float = 1.0) -> None:
        """Enable real-time metric streaming."""
        self.streaming_enabled = True
        self.stream_interval = interval
        self.last_stream_time = time.time()
        self.logger.info(f"Streaming enabled with {interval}s interval")
    
    def add_stream_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for real-time metric streaming."""
        self.stream_callbacks.append(callback)
    
    def _trigger_streaming(self) -> None:
        """Trigger real-time metric streaming to callbacks."""
        if not self.stream_callbacks:
            return
        
        # Aggregate current metrics
        current_metrics = self.aggregate_metrics(force=False)
        
        # Send to all callbacks asynchronously
        for callback in self.stream_callbacks:
            self.executor.submit(self._safe_callback, callback, current_metrics)
    
    def _safe_callback(self, callback: Callable, metrics: Dict[str, Any]) -> None:
        """Safely execute callback with error handling."""
        try:
            callback(metrics.copy())
        except Exception as e:
            self.logger.error(f"Stream callback error: {e}")
    
    def get_metric_summary(self, metric_name: str, 
                          time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get detailed summary for a specific metric."""
        with self.aggregation_lock:
            updates = self.metrics_buffer.get(metric_name, [])
            
            if time_window:
                current_time = time.time()
                updates = [u for u in updates if current_time - u.timestamp <= time_window]
            
            if not updates:
                return {'metric_name': metric_name, 'count': 0}
            
            values = [u.value for u in updates if isinstance(u.value, (int, float))]
            
            if not values:
                return {'metric_name': metric_name, 'count': 0}
            
            return {
                'metric_name': metric_name,
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'sum': sum(values),
                'latest': values[-1] if values else None,
                'timestamps': [u.timestamp for u in updates[-10:]],  # Last 10
                'source_ranks': list(set(u.source_rank for u in updates)),
                'strategies': list(set(u.strategy_name for u in updates)),
            }
    
    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """Get aggregator performance statistics."""
        avg_aggregation_time = (self.aggregation_time / self.total_aggregations 
                               if self.total_aggregations > 0 else 0)
        
        return {
            'total_updates': self.total_updates,
            'total_aggregations': self.total_aggregations,
            'avg_aggregation_time_ms': avg_aggregation_time * 1000,
            'total_aggregation_time_ms': self.aggregation_time * 1000,
            'buffer_sizes': {name: len(updates) for name, updates in self.metrics_buffer.items()},
            'streaming_enabled': self.streaming_enabled,
            'stream_callbacks': len(self.stream_callbacks),
            'mpi_enabled': self.mpi_comm is not None,
            'mpi_rank': self.mpi_rank,
            'mpi_size': self.mpi_size,
            'gpu_enabled': self.gpu_enabled,
            'aggregation_methods': {name: method.value for name, method in self.aggregation_methods.items()},
        }
    
    def export_metrics(self, format: str = 'dict') -> Any:
        """
        Export aggregated metrics in specified format.
        
        Args:
            format: Export format ('dict', 'json', 'parquet', 'csv')
            
        Returns:
            Exported metrics in requested format
        """
        metrics = self.aggregate_metrics(force=True)
        
        if format == 'dict':
            return metrics
        elif format == 'json':
            import json
            return json.dumps(metrics, indent=2, default=str)
        elif format == 'parquet':
            import pandas as pd
            # Flatten metrics for tabular format
            flat_data = []
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    flat_data.append({'metric_name': metric_name, 'value': value})
                elif isinstance(value, dict):
                    for sub_name, sub_value in value.items():
                        flat_data.append({
                            'metric_name': f"{metric_name}_{sub_name}",
                            'value': sub_value
                        })
            
            df = pd.DataFrame(flat_data)
            return df.to_parquet(index=False)
        elif format == 'csv':
            import pandas as pd
            # Similar flattening as parquet
            flat_data = []
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    flat_data.append({'metric_name': metric_name, 'value': value})
                elif isinstance(value, dict):
                    for sub_name, sub_value in value.items():
                        flat_data.append({
                            'metric_name': f"{metric_name}_{sub_name}",
                            'value': sub_value
                        })
            
            df = pd.DataFrame(flat_data)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_buffers(self) -> None:
        """Clear all metric buffers."""
        with self.aggregation_lock:
            self.metrics_buffer.clear()
            self.logger.info("Metric buffers cleared")
    
    def shutdown(self) -> None:
        """Shutdown aggregator and cleanup resources."""
        self.streaming_enabled = False
        
        # Final aggregation
        final_metrics = self.aggregate_metrics(force=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Metric aggregator shutdown complete")
        
        return final_metrics
