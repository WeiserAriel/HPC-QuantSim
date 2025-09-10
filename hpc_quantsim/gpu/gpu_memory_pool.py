"""
GPU memory pool manager for HPC QuantSim.

Provides efficient GPU memory allocation and management:
- Pre-allocated memory pools
- Memory reuse to avoid allocation overhead
- Memory fragmentation prevention
- Multi-stream memory management
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
from collections import defaultdict

# Try to import CUDA libraries
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = None


class GPUMemoryPool:
    """
    Efficient GPU memory pool for high-performance computing.
    
    Features:
    - Pre-allocated memory blocks to reduce allocation overhead
    - Automatic memory reuse and recycling
    - Multi-size pool management
    - Thread-safe operations
    - Memory usage tracking and statistics
    """
    
    def __init__(self, device_id: int = 0, initial_pool_size_mb: int = 1024):
        """Initialize GPU memory pool."""
        self.device_id = device_id
        self.initial_pool_size = initial_pool_size_mb * 1024 * 1024  # Convert to bytes
        
        self.logger = logging.getLogger(__name__)
        
        if not HAS_CUDA:
            raise ImportError("CUDA/CuPy not available")
        
        # Set device context
        self.device = cp.cuda.Device(device_id)
        
        # Memory pools organized by size
        self.pools: Dict[int, List[cp.ndarray]] = defaultdict(list)
        self.allocated_blocks: Dict[id, Tuple[int, cp.ndarray]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_allocated_bytes': 0,
            'total_freed_bytes': 0,
            'current_usage_bytes': 0,
            'allocation_count': 0,
            'free_count': 0,
            'pool_hit_count': 0,
            'pool_miss_count': 0,
            'peak_usage_bytes': 0,
        }
        
        # Standard block sizes (powers of 2 for efficient alignment)
        self.standard_sizes = [
            1024,      # 1 KB
            4096,      # 4 KB  
            16384,     # 16 KB
            65536,     # 64 KB
            262144,    # 256 KB
            1048576,   # 1 MB
            4194304,   # 4 MB
            16777216,  # 16 MB
            67108864,  # 64 MB
        ]
        
        # Initialize pools with device context
        with self.device:
            self._initialize_pools()
    
    def _initialize_pools(self):
        """Initialize memory pools with pre-allocated blocks."""
        try:
            # Get available memory
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            
            # Use portion of available memory for pool
            pool_budget = min(self.initial_pool_size, free_mem // 2)
            
            self.logger.info(f"Initializing GPU memory pool with {pool_budget / (1024**2):.1f} MB budget")
            
            # Allocate blocks for each standard size
            remaining_budget = pool_budget
            
            for size in sorted(self.standard_sizes, reverse=True):  # Start with largest
                if remaining_budget < size:
                    continue
                
                # Allocate multiple blocks of this size
                blocks_of_size = min(4, remaining_budget // size)  # Up to 4 blocks per size
                
                for _ in range(blocks_of_size):
                    if remaining_budget >= size:
                        try:
                            block = cp.zeros(size // 4, dtype=cp.float32)  # Allocate as float32 array
                            self.pools[size].append(block)
                            remaining_budget -= size
                            self.stats['total_allocated_bytes'] += size
                            
                        except cp.cuda.memory.OutOfMemoryError:
                            self.logger.warning(f"Failed to allocate {size} byte block")
                            break
                
                if remaining_budget < min(self.standard_sizes):
                    break
            
            self.logger.info(f"Initialized pools with {len([b for pool in self.pools.values() for b in pool])} blocks")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory pools: {e}")
    
    def allocate(self, size_bytes: int, dtype=cp.float32) -> Optional[cp.ndarray]:
        """
        Allocate GPU memory block.
        
        Args:
            size_bytes: Required size in bytes
            dtype: Data type for allocation
            
        Returns:
            GPU memory block or None if allocation fails
        """
        with self.lock:
            with self.device:
                # Find appropriate pool size
                pool_size = self._find_pool_size(size_bytes)
                
                # Try to get from pool first
                if pool_size in self.pools and self.pools[pool_size]:
                    block = self.pools[pool_size].pop()
                    
                    # Resize if needed
                    elements_needed = size_bytes // dtype().nbytes
                    if block.size != elements_needed:
                        # Create view with correct size
                        block = block.view()
                        block = block.reshape(-1)[:elements_needed]
                    
                    self.allocated_blocks[id(block)] = (pool_size, block)
                    self.stats['pool_hit_count'] += 1
                    self.stats['allocation_count'] += 1
                    self.stats['current_usage_bytes'] += pool_size
                    self.stats['peak_usage_bytes'] = max(self.stats['peak_usage_bytes'], 
                                                       self.stats['current_usage_bytes'])
                    
                    return block
                
                # Pool miss - allocate new block
                try:
                    elements_needed = size_bytes // dtype().nbytes
                    block = cp.zeros(elements_needed, dtype=dtype)
                    
                    actual_size = block.nbytes
                    self.allocated_blocks[id(block)] = (actual_size, block)
                    
                    self.stats['pool_miss_count'] += 1
                    self.stats['allocation_count'] += 1
                    self.stats['total_allocated_bytes'] += actual_size
                    self.stats['current_usage_bytes'] += actual_size
                    self.stats['peak_usage_bytes'] = max(self.stats['peak_usage_bytes'], 
                                                       self.stats['current_usage_bytes'])
                    
                    return block
                    
                except cp.cuda.memory.OutOfMemoryError:
                    self.logger.error(f"Out of GPU memory allocating {size_bytes} bytes")
                    return None
                
                except Exception as e:
                    self.logger.error(f"GPU allocation failed: {e}")
                    return None
    
    def free(self, block: cp.ndarray) -> bool:
        """
        Free GPU memory block back to pool.
        
        Args:
            block: GPU memory block to free
            
        Returns:
            True if successfully freed
        """
        with self.lock:
            block_id = id(block)
            
            if block_id not in self.allocated_blocks:
                self.logger.warning("Attempting to free untracked memory block")
                return False
            
            pool_size, original_block = self.allocated_blocks[block_id]
            
            # Return to appropriate pool if it's a standard size
            if pool_size in self.standard_sizes and len(self.pools[pool_size]) < 10:
                # Reset block content and return to pool
                original_block.fill(0)  # Clear data
                self.pools[pool_size].append(original_block)
            
            # Update statistics
            del self.allocated_blocks[block_id]
            self.stats['free_count'] += 1
            self.stats['total_freed_bytes'] += pool_size
            self.stats['current_usage_bytes'] -= pool_size
            
            return True
    
    def _find_pool_size(self, required_size: int) -> int:
        """Find the best pool size for required allocation."""
        # Find smallest standard size that fits the requirement
        for size in self.standard_sizes:
            if size >= required_size:
                return size
        
        # If larger than largest standard size, return exact size
        return required_size
    
    def allocate_array(self, shape: Tuple[int, ...], dtype=cp.float32) -> Optional[cp.ndarray]:
        """
        Allocate GPU array with specified shape and dtype.
        
        Args:
            shape: Array shape
            dtype: Data type
            
        Returns:
            GPU array or None if allocation fails
        """
        import numpy as np
        
        total_elements = np.prod(shape)
        size_bytes = total_elements * dtype().nbytes
        
        block = self.allocate(size_bytes, dtype)
        if block is None:
            return None
        
        # Reshape to desired shape
        try:
            return block.reshape(shape)
        except Exception as e:
            self.logger.error(f"Failed to reshape allocated block: {e}")
            self.free(block)
            return None
    
    def allocate_like(self, array) -> Optional[cp.ndarray]:
        """Allocate GPU array with same shape and dtype as input array."""
        if hasattr(array, 'shape') and hasattr(array, 'dtype'):
            return self.allocate_array(array.shape, array.dtype)
        else:
            self.logger.error("Input array must have shape and dtype attributes")
            return None
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory pool information."""
        with self.lock:
            with self.device:
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                used_mem = total_mem - free_mem
                
                pool_info = {}
                total_pool_blocks = 0
                total_pool_bytes = 0
                
                for size, blocks in self.pools.items():
                    pool_info[f"pool_{size}_bytes"] = {
                        'block_count': len(blocks),
                        'total_bytes': len(blocks) * size
                    }
                    total_pool_blocks += len(blocks)
                    total_pool_bytes += len(blocks) * size
                
                return {
                    'device_id': self.device_id,
                    'device_memory': {
                        'total_gb': total_mem / (1024**3),
                        'used_gb': used_mem / (1024**3),
                        'free_gb': free_mem / (1024**3),
                        'utilization_pct': (used_mem / total_mem) * 100
                    },
                    'pool_stats': {
                        'total_pool_blocks': total_pool_blocks,
                        'total_pool_bytes': total_pool_bytes,
                        'total_pool_mb': total_pool_bytes / (1024**2),
                        'allocated_blocks': len(self.allocated_blocks),
                        'current_usage_mb': self.stats['current_usage_bytes'] / (1024**2),
                        'peak_usage_mb': self.stats['peak_usage_bytes'] / (1024**2),
                    },
                    'pool_breakdown': pool_info,
                    'statistics': self.stats.copy()
                }
    
    def defragment(self) -> int:
        """
        Defragment memory pools by consolidating free blocks.
        
        Returns:
            Number of blocks consolidated
        """
        with self.lock:
            consolidated = 0
            
            # This is a simplified defragmentation
            # In practice, you'd implement more sophisticated algorithms
            
            for size in list(self.pools.keys()):
                blocks = self.pools[size]
                
                # Remove empty references
                valid_blocks = [b for b in blocks if b is not None]
                removed = len(blocks) - len(valid_blocks)
                
                if removed > 0:
                    self.pools[size] = valid_blocks
                    consolidated += removed
            
            # Remove empty pools
            empty_pools = [size for size, blocks in self.pools.items() if not blocks]
            for size in empty_pools:
                del self.pools[size]
                
            if consolidated > 0:
                self.logger.info(f"Defragmentation consolidated {consolidated} blocks")
            
            return consolidated
    
    def clear_pools(self) -> None:
        """Clear all memory pools and release GPU memory."""
        with self.lock:
            with self.device:
                # Free all pooled memory
                total_freed = 0
                for size, blocks in self.pools.items():
                    total_freed += len(blocks) * size
                    for block in blocks:
                        del block  # Let Python GC handle cleanup
                
                self.pools.clear()
                
                # Update stats
                self.stats['total_freed_bytes'] += total_freed
                
                self.logger.info(f"Cleared memory pools, freed {total_freed / (1024**2):.1f} MB")
    
    def resize_pools(self, new_size_mb: int) -> None:
        """Resize memory pools to new target size."""
        with self.lock:
            self.logger.info(f"Resizing memory pools to {new_size_mb} MB")
            
            # Clear current pools
            self.clear_pools()
            
            # Update pool size
            self.initial_pool_size = new_size_mb * 1024 * 1024
            
            # Reinitialize
            with self.device:
                self._initialize_pools()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory pool statistics."""
        with self.lock:
            pool_efficiency = 0.0
            if self.stats['allocation_count'] > 0:
                pool_efficiency = (self.stats['pool_hit_count'] / 
                                 self.stats['allocation_count']) * 100
            
            return {
                **self.stats,
                'pool_hit_rate_pct': pool_efficiency,
                'average_allocation_size_bytes': (self.stats['total_allocated_bytes'] / 
                                                 max(1, self.stats['allocation_count'])),
                'memory_utilization_pct': (self.stats['current_usage_bytes'] / 
                                          max(1, self.stats['peak_usage_bytes'])) * 100,
                'fragmentation_ratio': len(self.allocated_blocks) / max(1, 
                    sum(len(blocks) for blocks in self.pools.values())),
            }
    
    def __del__(self):
        """Cleanup when pool is destroyed."""
        try:
            self.clear_pools()
        except Exception:
            pass  # Ignore cleanup errors


# Global memory pool instance
_global_pool: Optional[GPUMemoryPool] = None
_pool_lock = threading.Lock()


def get_global_pool(device_id: int = 0, pool_size_mb: int = 1024) -> Optional[GPUMemoryPool]:
    """Get or create global GPU memory pool instance."""
    global _global_pool
    
    if not HAS_CUDA:
        return None
    
    with _pool_lock:
        if _global_pool is None:
            try:
                _global_pool = GPUMemoryPool(device_id, pool_size_mb)
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to create global GPU memory pool: {e}")
                return None
        
        return _global_pool


def clear_global_pool():
    """Clear the global GPU memory pool."""
    global _global_pool
    
    with _pool_lock:
        if _global_pool:
            _global_pool.clear_pools()
            _global_pool = None
