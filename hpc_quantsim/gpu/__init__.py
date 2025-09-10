"""
GPU acceleration components for HPC QuantSim.

This module provides CUDA-based acceleration for:
- Parallel strategy execution
- Statistical computations
- Matrix operations for portfolio optimization
- Memory management and data transfers
"""

# Try to import GPU components, fallback gracefully if CUDA not available
try:
    from .cuda_kernels import CUDAKernels
    from .gpu_memory_pool import GPUMemoryPool
    from .gpu_utils import GPUUtils, check_gpu_availability
    HAS_CUDA = True
except ImportError:
    # Fallback implementations
    class CUDAKernels:
        def __init__(self):
            raise ImportError("CUDA not available")
    
    class GPUMemoryPool:
        def __init__(self):
            raise ImportError("CUDA not available")
    
    class GPUUtils:
        @staticmethod
        def check_gpu_availability():
            return False
    
    def check_gpu_availability():
        return False
    
    HAS_CUDA = False

__all__ = [
    'CUDAKernels', 'GPUMemoryPool', 'GPUUtils', 
    'check_gpu_availability', 'HAS_CUDA'
]
