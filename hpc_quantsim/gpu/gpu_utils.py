"""
GPU utilities and helper functions for HPC QuantSim.

Provides device management, memory utilities, and performance monitoring
for CUDA-based acceleration components.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time

# Try to import CUDA libraries
try:
    import cupy as cp
    import cupyx.profiler
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    # Create a dummy cp module for type annotations
    class DummyCuPy:
        float32 = float
        ndarray = object
    cp = DummyCuPy()

class GPUUtils:
    """Utility functions for GPU operations and management."""
    
    @staticmethod
    def check_gpu_availability() -> bool:
        """Check if CUDA GPU is available."""
        if not HAS_CUDA:
            return False
        
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            return device_count > 0
        except Exception:
            return False
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get detailed information about available GPU devices."""
        if not HAS_CUDA:
            return {'available': False, 'reason': 'CUDA not installed'}
        
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            
            if device_count == 0:
                return {'available': False, 'reason': 'No CUDA devices found'}
            
            devices = []
            for device_id in range(device_count):
                with cp.cuda.Device(device_id):
                    props = cp.cuda.runtime.getDeviceProperties(device_id)
                    
                    device_info = {
                        'id': device_id,
                        'name': props['name'].decode('utf-8'),
                        'compute_capability': f"{props['major']}.{props['minor']}",
                        'total_memory_gb': props['totalGlobalMem'] / (1024**3),
                        'multiprocessor_count': props['multiProcessorCount'],
                        'max_threads_per_block': props['maxThreadsPerBlock'],
                        'max_block_dim': props['maxThreadsDim'],
                        'max_grid_dim': props['maxGridSize'],
                        'clock_rate_khz': props['clockRate'],
                        'memory_clock_rate_khz': props['memoryClockRate'],
                        'memory_bus_width': props['memoryBusWidth'],
                        'l2_cache_size': props['l2CacheSize'],
                        'concurrent_kernels': bool(props['concurrentKernels']),
                        'ecc_enabled': bool(props['ECCEnabled']),
                    }
                    devices.append(device_info)
            
            return {
                'available': True,
                'device_count': device_count,
                'devices': devices
            }
            
        except Exception as e:
            return {'available': False, 'reason': str(e)}
    
    @staticmethod
    def get_memory_info(device_id: int = 0) -> Dict[str, int]:
        """Get memory information for specified device."""
        if not HAS_CUDA:
            return {'free': 0, 'total': 0, 'used': 0}
        
        try:
            with cp.cuda.Device(device_id):
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                used_mem = total_mem - free_mem
                
                return {
                    'free': free_mem,
                    'total': total_mem,
                    'used': used_mem,
                    'free_gb': free_mem / (1024**3),
                    'total_gb': total_mem / (1024**3),
                    'used_gb': used_mem / (1024**3),
                    'utilization_pct': (used_mem / total_mem) * 100
                }
        except Exception:
            return {'free': 0, 'total': 0, 'used': 0}
    
    @staticmethod
    def benchmark_device(device_id: int = 0, size: int = 10000000) -> Dict[str, float]:
        """Benchmark GPU device performance."""
        if not HAS_CUDA:
            return {}
        
        try:
            with cp.cuda.Device(device_id):
                # Memory bandwidth test
                data = cp.random.random(size, dtype=cp.float32)
                
                # GPU to GPU copy
                start_time = time.time()
                data_copy = cp.copy(data)
                cp.cuda.Stream.null.synchronize()
                gpu_copy_time = time.time() - start_time
                
                # Element-wise operations
                start_time = time.time()
                result = cp.sqrt(data * data + data_copy * data_copy)
                cp.cuda.Stream.null.synchronize()
                compute_time = time.time() - start_time
                
                # Reduction operations
                start_time = time.time()
                sum_result = cp.sum(result)
                cp.cuda.Stream.null.synchronize()
                reduction_time = time.time() - start_time
                
                # Host to device transfer
                host_data = np.random.random(size).astype(np.float32)
                start_time = time.time()
                device_data = cp.asarray(host_data)
                cp.cuda.Stream.null.synchronize()
                h2d_time = time.time() - start_time
                
                # Device to host transfer
                start_time = time.time()
                host_result = cp.asnumpy(device_data)
                cp.cuda.Stream.null.synchronize()
                d2h_time = time.time() - start_time
                
                # Calculate metrics
                data_size_gb = (size * 4) / (1024**3)  # 4 bytes per float32
                
                return {
                    'device_id': device_id,
                    'data_size_gb': data_size_gb,
                    'gpu_copy_time_ms': gpu_copy_time * 1000,
                    'gpu_copy_bandwidth_gb_s': data_size_gb / gpu_copy_time,
                    'compute_time_ms': compute_time * 1000,
                    'compute_throughput_gflops': (size * 3) / compute_time / 1e9,  # 3 ops per element
                    'reduction_time_ms': reduction_time * 1000,
                    'h2d_time_ms': h2d_time * 1000,
                    'h2d_bandwidth_gb_s': data_size_gb / h2d_time,
                    'd2h_time_ms': d2h_time * 1000,
                    'd2h_bandwidth_gb_s': data_size_gb / d2h_time,
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def optimize_block_size(func, data_size: int, max_block_size: int = 1024) -> Tuple[int, int]:
        """Optimize CUDA block size for given function and data size."""
        if not HAS_CUDA:
            return 1, data_size
        
        # Use heuristics to determine optimal block size
        # This is a simplified version - production code would benchmark different sizes
        
        # Start with reasonable defaults based on data size
        if data_size <= 1024:
            block_size = min(data_size, max_block_size)
            grid_size = 1
        else:
            # Use multiple of 32 (warp size) for best performance
            block_size = min(256, max_block_size)  # Good default for most cases
            grid_size = (data_size + block_size - 1) // block_size
        
        return block_size, grid_size
    
    @staticmethod
    def create_streams(num_streams: int = 4) -> List:
        """Create CUDA streams for concurrent execution."""
        if not HAS_CUDA:
            return []
        
        try:
            streams = []
            for _ in range(num_streams):
                stream = cp.cuda.Stream()
                streams.append(stream)
            return streams
        except Exception:
            return []
    
    @staticmethod
    def synchronize_streams(streams: List) -> None:
        """Synchronize all CUDA streams."""
        if not HAS_CUDA or not streams:
            return
        
        try:
            for stream in streams:
                stream.synchronize()
        except Exception:
            pass
    
    @staticmethod
    def profile_kernel(func, *args, **kwargs) -> Dict[str, Any]:
        """Profile CUDA kernel execution."""
        if not HAS_CUDA:
            return {}
        
        try:
            # Use CuPy profiler
            with cupyx.profiler.profile():
                start_time = time.time()
                result = func(*args, **kwargs)
                cp.cuda.Stream.null.synchronize()
                end_time = time.time()
            
            return {
                'execution_time_ms': (end_time - start_time) * 1000,
                'result_shape': result.shape if hasattr(result, 'shape') else None,
                'result_dtype': str(result.dtype) if hasattr(result, 'dtype') else None,
            }
            
        except Exception as e:
            return {'error': str(e)}


class GPUTimer:
    """High-precision GPU timer using CUDA events."""
    
    def __init__(self):
        self.start_event = None
        self.end_event = None
        
        if HAS_CUDA:
            self.start_event = cp.cuda.Event()
            self.end_event = cp.cuda.Event()
    
    def start(self):
        """Start timing."""
        if HAS_CUDA and self.start_event:
            self.start_event.record()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time in milliseconds."""
        if not HAS_CUDA or not self.end_event:
            return 0.0
        
        self.end_event.record()
        self.end_event.synchronize()
        
        return cp.cuda.get_elapsed_time(self.start_event, self.end_event)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.stop()


def check_gpu_availability() -> bool:
    """Check if GPU is available (standalone function)."""
    return GPUUtils.check_gpu_availability()


def print_gpu_info():
    """Print GPU information to console."""
    info = GPUUtils.get_device_info()
    
    if not info['available']:
        print(f"GPU not available: {info.get('reason', 'Unknown')}")
        return
    
    print(f"Found {info['device_count']} CUDA device(s):")
    
    for device in info['devices']:
        print(f"\nDevice {device['id']}: {device['name']}")
        print(f"  Compute Capability: {device['compute_capability']}")
        print(f"  Total Memory: {device['total_memory_gb']:.1f} GB")
        print(f"  Multiprocessors: {device['multiprocessor_count']}")
        print(f"  Max Threads/Block: {device['max_threads_per_block']}")
        print(f"  Clock Rate: {device['clock_rate_khz']/1000:.0f} MHz")
        print(f"  Memory Clock: {device['memory_clock_rate_khz']/1000:.0f} MHz")
        print(f"  Memory Bus Width: {device['memory_bus_width']} bits")
        print(f"  L2 Cache: {device['l2_cache_size']/1024:.0f} KB")
        print(f"  Concurrent Kernels: {'Yes' if device['concurrent_kernels'] else 'No'}")
        print(f"  ECC Enabled: {'Yes' if device['ecc_enabled'] else 'No'}")
        
        # Memory info
        mem_info = GPUUtils.get_memory_info(device['id'])
        print(f"  Memory Usage: {mem_info.get('used_gb', 0):.1f}/{mem_info.get('total_gb', 0):.1f} GB ({mem_info.get('utilization_pct', 0):.1f}%)")


def benchmark_all_devices():
    """Benchmark all available GPU devices."""
    info = GPUUtils.get_device_info()
    
    if not info['available']:
        print(f"GPU not available: {info.get('reason', 'Unknown')}")
        return
    
    print("GPU Performance Benchmark")
    print("=" * 50)
    
    for device in info['devices']:
        print(f"\nBenchmarking Device {device['id']}: {device['name']}")
        print("-" * 40)
        
        benchmark = GPUUtils.benchmark_device(device['id'])
        
        if 'error' in benchmark:
            print(f"Benchmark failed: {benchmark['error']}")
            continue
        
        print(f"Data Size: {benchmark['data_size_gb']:.2f} GB")
        print(f"GPU Copy Time: {benchmark['gpu_copy_time_ms']:.2f} ms")
        print(f"GPU Copy Bandwidth: {benchmark['gpu_copy_bandwidth_gb_s']:.1f} GB/s")
        print(f"Compute Time: {benchmark['compute_time_ms']:.2f} ms")
        print(f"Compute Throughput: {benchmark['compute_throughput_gflops']:.1f} GFLOPS")
        print(f"Reduction Time: {benchmark['reduction_time_ms']:.2f} ms")
        print(f"Host→Device: {benchmark['h2d_time_ms']:.2f} ms ({benchmark['h2d_bandwidth_gb_s']:.1f} GB/s)")
        print(f"Device→Host: {benchmark['d2h_time_ms']:.2f} ms ({benchmark['d2h_bandwidth_gb_s']:.1f} GB/s)")


if __name__ == "__main__":
    # Command-line utility for GPU diagnostics
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_all_devices()
    else:
        print_gpu_info()
