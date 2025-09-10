"""
Configuration management for HPC QuantSim.

Handles loading and validation of simulation configurations, HPC settings,
and system parameters.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
# from omegaconf import OmegaConf, DictConfig  # Optional dependency

@dataclass
class SimulationConfig:
    """Core simulation configuration."""
    # Simulation parameters
    num_simulations: int = 1000
    max_parallel: int = 100
    random_seed: int = 42
    simulation_time_ms: int = 3600000  # 1 hour in milliseconds
    
    # Market data
    market_data_path: str = "data/market_data.parquet"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    symbols: list = None
    
    # Strategy configuration
    strategy_plugins: list = None
    strategy_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["SPY", "QQQ", "IWM"]
        if self.strategy_plugins is None:
            self.strategy_plugins = []
        if self.strategy_params is None:
            self.strategy_params = {}

@dataclass
class HPCConfig:
    """HPC and distributed computing configuration."""
    # MPI settings
    use_mpi: bool = False
    mpi_ranks: int = 1
    nodes_per_rank: int = 1
    
    # GPU settings
    use_gpu: bool = False
    gpus_per_node: int = 1
    gpu_memory_pool_mb: int = 2048
    
    # Communication settings
    use_ucx: bool = False
    use_sharp: bool = False
    use_gpudirect: bool = False
    
    # Collective operations
    collective_backend: str = "nccl"  # nccl, hcoll, ucc
    allreduce_algorithm: str = "tree"
    
    # Performance tuning
    cpu_affinity: bool = True
    numa_binding: bool = True
    thread_pinning: bool = True

@dataclass
class MetricsConfig:
    """Metrics collection and aggregation configuration."""
    # Performance metrics
    collect_pnl: bool = True
    collect_sharpe: bool = True
    collect_drawdown: bool = True
    collect_latency: bool = True
    
    # Risk metrics
    collect_var: bool = True
    collect_cvar: bool = True
    var_confidence: float = 0.95
    
    # Aggregation settings
    aggregation_interval_ms: int = 1000
    real_time_updates: bool = True
    
    # Storage
    metrics_output_path: str = "results/metrics"
    save_intermediate: bool = False

@dataclass
class VisualizationConfig:
    """Real-time visualization configuration."""
    enable_dashboard: bool = False
    dashboard_port: int = 8080
    dashboard_host: str = "localhost"
    
    # Chart settings
    update_interval_ms: int = 1000
    max_data_points: int = 10000
    
    # Export settings
    save_plots: bool = False
    plot_format: str = "html"

@dataclass
class DeploymentConfig:
    """Cluster deployment configuration."""
    # Cluster settings
    scheduler: str = "slurm"  # slurm, pbs, sge
    job_name: str = "hpc_quantsim"
    walltime: str = "01:00:00"
    
    # Resource allocation
    nodes: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int = 1
    memory_per_node_gb: int = 16
    
    # Software environment
    modules_to_load: list = None
    conda_env: Optional[str] = None
    singularity_image: Optional[str] = None
    
    def __post_init__(self):
        if self.modules_to_load is None:
            self.modules_to_load = ["openmpi", "cuda", "hpcx"]

@dataclass
class Config:
    """Main configuration container."""
    simulation: SimulationConfig
    hpc: HPCConfig
    metrics: MetricsConfig
    visualization: VisualizationConfig
    deployment: DeploymentConfig

def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file format
    suffix = config_path.suffix.lower()
    
    if suffix == '.json':
        with open(config_path, 'r') as f:
            data = json.load(f)
    elif suffix in ['.yml', '.yaml']:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")
    
    # Simple configuration handling (fallback without OmegaConf)
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(data)
        return Config(
            simulation=SimulationConfig(**cfg.get('simulation', {})),
            hpc=HPCConfig(**cfg.get('hpc', {})),
            metrics=MetricsConfig(**cfg.get('metrics', {})),
            visualization=VisualizationConfig(**cfg.get('visualization', {})),
            deployment=DeploymentConfig(**cfg.get('deployment', {}))
        )
    except ImportError:
        # Fallback without OmegaConf
        return Config(
            simulation=SimulationConfig(**data.get('simulation', {})),
            hpc=HPCConfig(**data.get('hpc', {})),
            metrics=MetricsConfig(**data.get('metrics', {})),
            visualization=VisualizationConfig(**data.get('visualization', {})),
            deployment=DeploymentConfig(**data.get('deployment', {}))
        )

def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to file."""
    config_path = Path(config_path)
    
    # Convert to dictionary
    data = {
        'simulation': asdict(config.simulation),
        'hpc': asdict(config.hpc),
        'metrics': asdict(config.metrics),
        'visualization': asdict(config.visualization),
        'deployment': asdict(config.deployment)
    }
    
    # Determine file format
    suffix = config_path.suffix.lower()
    
    if suffix == '.json':
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif suffix in ['.yml', '.yaml']:
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")

def create_default_config() -> Config:
    """Create default configuration."""
    return Config(
        simulation=SimulationConfig(),
        hpc=HPCConfig(),
        metrics=MetricsConfig(),
        visualization=VisualizationConfig(),
        deployment=DeploymentConfig()
    )

def merge_configs(base_config: Config, override_config: dict) -> Config:
    """Merge configuration with overrides."""
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.structured(base_config)
        override_cfg = OmegaConf.create(override_config)
        merged = OmegaConf.merge(cfg, override_cfg)
        return OmegaConf.to_object(merged)
    except ImportError:
        # Simple fallback merge
        import copy
        merged_config = copy.deepcopy(base_config)
        
        for section, values in override_config.items():
            if hasattr(merged_config, section):
                section_obj = getattr(merged_config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        
        return merged_config

# Environment-based configuration
def load_config_from_env() -> Dict[str, Any]:
    """Load configuration overrides from environment variables."""
    env_config = {}
    
    # HPC settings
    if 'QUANTSIM_USE_GPU' in os.environ:
        env_config['hpc'] = {'use_gpu': os.environ['QUANTSIM_USE_GPU'].lower() == 'true'}
    
    if 'QUANTSIM_MPI_RANKS' in os.environ:
        if 'hpc' not in env_config:
            env_config['hpc'] = {}
        env_config['hpc']['mpi_ranks'] = int(os.environ['QUANTSIM_MPI_RANKS'])
    
    # Simulation settings
    if 'QUANTSIM_NUM_SIMS' in os.environ:
        env_config['simulation'] = {'num_simulations': int(os.environ['QUANTSIM_NUM_SIMS'])}
    
    if 'QUANTSIM_DATA_PATH' in os.environ:
        if 'simulation' not in env_config:
            env_config['simulation'] = {}
        env_config['simulation']['market_data_path'] = os.environ['QUANTSIM_DATA_PATH']
    
    return env_config
