"""
HPC (High Performance Computing) integration for HPC QuantSim.

This module provides distributed computing capabilities using:
- OpenMPI for multi-node parallelization
- UCX for high-performance communication
- NCCL for GPU-to-GPU communication
- Cluster management and job scheduling
"""

# Try to import HPC components, fallback gracefully if not available
try:
    from .mpi_collectives import MPICollectives
    from .cluster_manager import ClusterManager
    HAS_MPI = True
except ImportError:
    # Fallback implementations
    class MPICollectives:
        def __init__(self):
            raise ImportError("MPI not available")
    
    class ClusterManager:
        def __init__(self):
            raise ImportError("MPI not available")
    
    HAS_MPI = False

try:
    from .ucx_transport import UCXTransport
    HAS_UCX = True
except ImportError:
    class UCXTransport:
        def __init__(self):
            raise ImportError("UCX not available")
    HAS_UCX = False

__all__ = [
    'MPICollectives', 'ClusterManager', 'UCXTransport',
    'HAS_MPI', 'HAS_UCX'
]
