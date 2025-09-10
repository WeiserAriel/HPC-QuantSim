# HPC QuantSim
## High-Frequency Market Simulator with Distributed GPU Compute

HPC QuantSim is a high-frequency market simulation platform built for stress-testing and benchmarking algorithmic trading strategies. It simulates tick-by-tick market behavior using real historical order book and trade data, running thousands of scenario variations across a distributed GPU cluster.

## Architecture Overview

### Core Components
- **Strategy Sandbox**: Python/C++ strategy plugins with stateless/stateful agents
- **Market Replay Engine**: Tick-level replay from historical LOB data
- **Scenario Simulation Core**: 1,000+ parallel simulations with parameter variations
- **Metric Aggregator**: Real-time PnL and risk metrics collection using HPC-X collectives
- **Real-Time Visualizer**: Web-based dashboard for live monitoring

### HPC-X Integration
- **OpenMPI**: Distributed simulation nodes
- **UCX**: Fast inter-node communication and GPU transfers
- **NCCL**: GPU-to-GPU collectives for metric reduction
- **CUDA**: Strategy execution and statistical computation
- **HCOLL/UCC**: Collective operations for metric aggregation
- **GPUDirect**: Direct GPU↔NIC data paths
- **ClusterKit**: Deployment automation and cluster management

## Performance Goals
- 5,000+ parallel simulations in under 5 minutes
- GPU↔GPU latency < 5µs with GPUDirect
- Real-time PnL collection with <0.5% variation
- Strategy integration in <10 minutes

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build C++ extensions
python setup.py build_ext --inplace

# Run sample simulation
python examples/simple_simulation.py
```

## Directory Structure
```
hpc_quantsim/
├── core/                 # Core simulation engine
├── strategies/           # Strategy plugins
├── market/              # Market data and replay engine
├── metrics/             # Performance and risk metrics
├── gpu/                 # CUDA kernels and GPU utilities
├── hpc/                 # HPC-X integration and MPI
├── visualization/       # Web-based dashboard
├── deployment/          # ClusterKit and Docker configs
└── examples/           # Sample strategies and simulations
```

## License
Apache 2.0
