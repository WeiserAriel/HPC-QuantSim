#!/usr/bin/env python3
"""
HPC QuantSim Cluster Deployment Management Tool

Command-line interface for deploying and managing HPC QuantSim 
across different cluster environments including SLURM, PBS, and Kubernetes.
"""

import argparse
import json
import sys
import os
import yaml
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..config import load_config, create_default_config
from ..hpc.cluster_manager import ClusterManager, get_cluster_manager


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging for the deployment tool."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_cluster_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load cluster configuration from YAML file."""
    if config_path is None:
        # Use default cluster config from package
        config_path = Path(__file__).parent / "cluster_config.yml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_slurm_script(
    cluster_name: str, 
    job_template: str,
    config: Dict[str, Any],
    scenarios: int = 1000,
    data_path: Optional[str] = None
) -> str:
    """Generate SLURM job script from template."""
    
    cluster_config = load_cluster_config()
    cluster_info = cluster_config['clusters'].get(cluster_name, cluster_config['default'])
    
    # Load SLURM template
    template_path = Path(__file__).parent / "slurm_template.json"
    with open(template_path, 'r') as f:
        slurm_config = json.load(f)
    
    template = slurm_config['job_templates'].get(job_template, slurm_config['job_templates']['default'])
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build SLURM script
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={template['job_name'].format(timestamp=timestamp)}",
        f"#SBATCH --nodes={cluster_info.get('nodes', template['nodes'])}",
        f"#SBATCH --ntasks-per-node={cluster_info.get('tasks_per_node', template['ntasks_per_node'])}",
        f"#SBATCH --cpus-per-task={cluster_info.get('cpus_per_task', template['cpus_per_task'])}",
        f"#SBATCH --mem={cluster_info.get('memory_per_node', template['mem'])}",
        f"#SBATCH --time={cluster_info.get('walltime', template['time'])}",
        f"#SBATCH --output={template['output']}",
        f"#SBATCH --error={template['error']}"
    ]
    
    # Add GPU requirements if specified
    if cluster_info.get('gpu_per_node', 0) > 0:
        gpu_spec = f"gpu:{cluster_info['gpu_per_node']}"
        if cluster_info.get('gpu_type'):
            gpu_spec = f"gpu:{cluster_info['gpu_type']}:{cluster_info['gpu_per_node']}"
        script_lines.append(f"#SBATCH --gres={gpu_spec}")
    
    # Add partition/account info
    if cluster_info.get('partition'):
        script_lines.append(f"#SBATCH --partition={cluster_info['partition']}")
    if cluster_info.get('account'):
        script_lines.append(f"#SBATCH --account={cluster_info['account']}")
    
    script_lines.extend([
        "",
        "# Load modules",
        "module purge"
    ])
    
    # Add module loads
    for module in cluster_info.get('modules', []):
        script_lines.append(f"module load {module}")
    
    script_lines.extend([
        "",
        "# Set environment variables"
    ])
    
    # Add environment variables
    for key, value in cluster_info.get('environment', {}).items():
        script_lines.append(f"export {key}={value}")
    
    script_lines.extend([
        "",
        "# Print job information",
        "echo \"Job started at $(date)\"",
        "echo \"Running on nodes: $SLURM_JOB_NODELIST\"",
        "echo \"Number of tasks: $SLURM_NTASKS\"",
        "",
        "# Change to working directory",
        "cd $SLURM_SUBMIT_DIR",
        "",
        "# Run HPC QuantSim"
    ])
    
    # Build the main command
    cmd_parts = [
        "mpirun",
        "-np $SLURM_NTASKS", 
        "python -m hpc_quantsim.cli run",
        f"--scenarios {scenarios}",
        "--mpi"
    ]
    
    if data_path:
        cmd_parts.append(f"--data-path {data_path}")
    
    if cluster_info.get('gpu_per_node', 0) > 0:
        cmd_parts.append("--gpu")
    
    script_lines.append(" ".join(cmd_parts))
    
    script_lines.extend([
        "",
        "echo \"Job completed at $(date)\"",
        "echo \"Exit status: $?\""
    ])
    
    return "\n".join(script_lines)


def deploy_to_cluster(args) -> None:
    """Deploy HPC QuantSim job to cluster."""
    logger = setup_logging(args.log_level)
    
    try:
        # Load configuration
        config = load_config(args.config) if args.config else create_default_config()
        cluster_manager = get_cluster_manager(config.deployment)
        
        if not cluster_manager:
            logger.error("No cluster manager available")
            sys.exit(1)
        
        # Generate job script
        if args.scheduler == "slurm":
            script_content = generate_slurm_script(
                cluster_name=args.cluster or "default",
                job_template=args.template,
                config=config.to_dict(),
                scenarios=args.scenarios,
                data_path=args.data_path
            )
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(script_content)
                script_path = f.name
            
            logger.info(f"Generated SLURM script: {script_path}")
            
            if args.dry_run:
                print("Generated SLURM script:")
                print("-" * 50)
                print(script_content)
                print("-" * 50)
            else:
                # Submit job
                job_id = cluster_manager.submit_job(
                    script_path=script_path,
                    job_name=f"quantsim-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    resources={
                        "nodes": args.nodes,
                        "tasks_per_node": args.tasks_per_node,
                        "walltime": args.walltime
                    }
                )
                logger.info(f"Submitted job with ID: {job_id}")
                
                # Clean up temporary script
                os.unlink(script_path)
                
        else:
            logger.error(f"Scheduler '{args.scheduler}' not yet implemented")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


def list_clusters(args) -> None:
    """List available cluster configurations."""
    try:
        cluster_config = load_cluster_config(args.config_file)
        
        print("Available cluster configurations:")
        print("=" * 50)
        
        for name, config in cluster_config['clusters'].items():
            print(f"\nCluster: {name}")
            print(f"  Scheduler: {config.get('scheduler', 'unknown')}")
            print(f"  Nodes: {config.get('nodes', 'N/A')}")
            print(f"  GPUs per node: {config.get('gpu_per_node', 0)}")
            print(f"  Memory per node: {config.get('memory_per_node', 'N/A')}")
            print(f"  Walltime: {config.get('walltime', 'N/A')}")
            
    except Exception as e:
        print(f"Error loading cluster configurations: {e}")
        sys.exit(1)


def check_cluster_status(args) -> None:
    """Check status of cluster and running jobs."""
    logger = setup_logging(args.log_level)
    
    try:
        config = load_config(args.config) if args.config else create_default_config()
        cluster_manager = get_cluster_manager(config.deployment)
        
        if not cluster_manager:
            logger.error("No cluster manager available")
            sys.exit(1)
        
        # Get job status
        if args.job_id:
            status = cluster_manager.get_job_status(args.job_id)
            print(f"Job {args.job_id} status: {status}")
        else:
            # List all quantsim jobs
            jobs = cluster_manager.list_jobs(user_filter=True)
            
            print("HPC QuantSim Jobs:")
            print("-" * 60)
            print(f"{'Job ID':<12} {'Name':<20} {'State':<12} {'Runtime':<10}")
            print("-" * 60)
            
            for job in jobs:
                if 'quantsim' in job.get('name', '').lower():
                    print(f"{job.get('id', 'N/A'):<12} "
                          f"{job.get('name', 'N/A'):<20} "
                          f"{job.get('state', 'N/A'):<12} "
                          f"{job.get('runtime', 'N/A'):<10}")
                          
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        sys.exit(1)


def cancel_jobs(args) -> None:
    """Cancel cluster jobs."""
    logger = setup_logging(args.log_level)
    
    try:
        config = load_config(args.config) if args.config else create_default_config()
        cluster_manager = get_cluster_manager(config.deployment)
        
        if not cluster_manager:
            logger.error("No cluster manager available")
            sys.exit(1)
        
        for job_id in args.job_ids:
            result = cluster_manager.cancel_job(job_id)
            if result:
                logger.info(f"Cancelled job {job_id}")
            else:
                logger.error(f"Failed to cancel job {job_id}")
                
    except Exception as e:
        logger.error(f"Job cancellation failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for cluster deployment tool."""
    parser = argparse.ArgumentParser(
        description="HPC QuantSim Cluster Deployment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy job to cluster')
    deploy_parser.add_argument('--config', help='Configuration file path')
    deploy_parser.add_argument('--cluster', help='Target cluster name')
    deploy_parser.add_argument('--scheduler', default='slurm', choices=['slurm', 'pbs'], help='Job scheduler')
    deploy_parser.add_argument('--template', default='default', help='Job template to use')
    deploy_parser.add_argument('--scenarios', type=int, default=1000, help='Number of scenarios')
    deploy_parser.add_argument('--data-path', help='Path to market data')
    deploy_parser.add_argument('--nodes', type=int, default=4, help='Number of nodes')
    deploy_parser.add_argument('--tasks-per-node', type=int, default=8, help='Tasks per node')
    deploy_parser.add_argument('--walltime', default='02:00:00', help='Wall time limit')
    deploy_parser.add_argument('--dry-run', action='store_true', help='Generate script without submitting')
    deploy_parser.set_defaults(func=deploy_to_cluster)
    
    # List clusters command
    list_parser = subparsers.add_parser('list', help='List available clusters')
    list_parser.add_argument('--config-file', help='Cluster configuration file')
    list_parser.set_defaults(func=list_clusters)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check cluster/job status')
    status_parser.add_argument('--config', help='Configuration file path')
    status_parser.add_argument('--job-id', help='Specific job ID to check')
    status_parser.set_defaults(func=check_cluster_status)
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel cluster jobs')
    cancel_parser.add_argument('--config', help='Configuration file path')
    cancel_parser.add_argument('job_ids', nargs='+', help='Job IDs to cancel')
    cancel_parser.set_defaults(func=cancel_jobs)
    
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
