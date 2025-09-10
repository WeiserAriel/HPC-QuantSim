"""
Cluster management for distributed HPC QuantSim execution.

Provides cluster-aware job management:
- SLURM/PBS job submission and monitoring
- Node resource management
- Distributed simulation orchestration
- Fault tolerance and recovery
- Performance monitoring
"""

import os
import subprocess
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile

from ..config import DeploymentConfig


class ClusterManager:
    """
    Cluster management and job orchestration for HPC QuantSim.
    
    Features:
    - Multi-scheduler support (SLURM, PBS, SGE)
    - Dynamic resource allocation
    - Job monitoring and recovery
    - Node health checking
    - Performance profiling
    """
    
    def __init__(self, config: DeploymentConfig):
        """Initialize cluster manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Detect available job scheduler
        self.scheduler = self._detect_scheduler()
        self.logger.info(f"Detected job scheduler: {self.scheduler}")
        
        # Job tracking
        self.active_jobs = {}
        self.job_history = []
        
        # Node information
        self.node_info = {}
        self.last_node_update = None
        
        # Performance tracking
        self.cluster_stats = {
            'jobs_submitted': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_compute_hours': 0.0,
            'total_cost': 0.0,
        }
    
    def _detect_scheduler(self) -> str:
        """Auto-detect available job scheduler."""
        # Check for SLURM
        if self._command_exists('sbatch'):
            return 'slurm'
        
        # Check for PBS/Torque
        if self._command_exists('qsub'):
            return 'pbs'
        
        # Check for SGE
        if self._command_exists('qsub') and os.path.exists('/opt/sge'):
            return 'sge'
        
        # No scheduler found
        return 'none'
    
    def _command_exists(self, command: str) -> bool:
        """Check if command exists in PATH."""
        try:
            subprocess.run(['which', command], check=True, 
                         capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def submit_simulation_job(self, num_scenarios: int, 
                            strategy_configs: List[Dict],
                            market_data_path: str,
                            output_path: str,
                            job_name: Optional[str] = None) -> Optional[str]:
        """
        Submit distributed simulation job to cluster.
        
        Args:
            num_scenarios: Total number of scenarios to run
            strategy_configs: List of strategy configurations
            market_data_path: Path to market data files
            output_path: Output directory for results
            job_name: Optional job name
            
        Returns:
            Job ID if successful, None otherwise
        """
        if self.scheduler == 'none':
            self.logger.error("No job scheduler available")
            return None
        
        job_name = job_name or f"quantsim_{int(time.time())}"
        
        # Calculate resource requirements
        nodes_needed = min(self.config.nodes, max(1, num_scenarios // 100))
        total_cores = nodes_needed * self.config.cpus_per_task
        estimated_runtime = self._estimate_runtime(num_scenarios, len(strategy_configs))
        
        self.logger.info(f"Submitting job: {job_name}")
        self.logger.info(f"Resources: {nodes_needed} nodes, {total_cores} cores")
        self.logger.info(f"Estimated runtime: {estimated_runtime}")
        
        # Generate job script
        job_script = self._generate_job_script(
            job_name=job_name,
            nodes=nodes_needed,
            runtime=estimated_runtime,
            num_scenarios=num_scenarios,
            strategy_configs=strategy_configs,
            market_data_path=market_data_path,
            output_path=output_path
        )
        
        # Submit job
        job_id = self._submit_job_script(job_script, job_name)
        
        if job_id:
            # Track job
            self.active_jobs[job_id] = {
                'name': job_name,
                'submitted_at': datetime.now(),
                'nodes': nodes_needed,
                'scenarios': num_scenarios,
                'strategies': len(strategy_configs),
                'status': 'submitted',
                'output_path': output_path,
            }
            
            self.cluster_stats['jobs_submitted'] += 1
            self.logger.info(f"Job submitted successfully: {job_id}")
        else:
            self.logger.error("Job submission failed")
        
        return job_id
    
    def _estimate_runtime(self, num_scenarios: int, num_strategies: int) -> str:
        """Estimate job runtime based on scenarios and strategies."""
        # Simple heuristic: ~1 second per scenario-strategy combination
        # Plus overhead for data loading and result aggregation
        
        total_combinations = num_scenarios * num_strategies
        estimated_seconds = total_combinations * 1.0 + 300  # Base + 5min overhead
        
        # Add scaling factor for smaller jobs (less efficient)
        if total_combinations < 1000:
            estimated_seconds *= 2
        
        # Convert to hours:minutes:seconds
        hours = int(estimated_seconds // 3600)
        minutes = int((estimated_seconds % 3600) // 60)
        seconds = int(estimated_seconds % 60)
        
        # Add safety margin
        hours += 1
        
        # Cap at walltime limit
        max_hours = int(self.config.walltime.split(':')[0])
        hours = min(hours, max_hours)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _generate_job_script(self, job_name: str, nodes: int, runtime: str,
                           num_scenarios: int, strategy_configs: List[Dict],
                           market_data_path: str, output_path: str) -> str:
        """Generate job script for the specified scheduler."""
        
        if self.scheduler == 'slurm':
            return self._generate_slurm_script(
                job_name, nodes, runtime, num_scenarios, 
                strategy_configs, market_data_path, output_path
            )
        elif self.scheduler == 'pbs':
            return self._generate_pbs_script(
                job_name, nodes, runtime, num_scenarios,
                strategy_configs, market_data_path, output_path
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")
    
    def _generate_slurm_script(self, job_name: str, nodes: int, runtime: str,
                              num_scenarios: int, strategy_configs: List[Dict],
                              market_data_path: str, output_path: str) -> str:
        """Generate SLURM job script."""
        
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={self.config.tasks_per_node}
#SBATCH --cpus-per-task={self.config.cpus_per_task}
#SBATCH --mem={self.config.memory_per_node_gb}GB
#SBATCH --time={runtime}
#SBATCH --output={output_path}/slurm-%j.out
#SBATCH --error={output_path}/slurm-%j.err
"""
        
        # Add GPU resources if needed
        if hasattr(self.config, 'gpus_per_node') and self.config.gpus_per_node > 0:
            script += f"#SBATCH --gres=gpu:{self.config.gpus_per_node}\n"
        
        # Add modules to load
        if self.config.modules_to_load:
            script += "\n# Load required modules\n"
            for module in self.config.modules_to_load:
                script += f"module load {module}\n"
        
        # Add conda environment activation
        if self.config.conda_env:
            script += f"\n# Activate conda environment\nconda activate {self.config.conda_env}\n"
        
        # Add Singularity setup if specified
        if self.config.singularity_image:
            script += f"\n# Singularity setup\nSINGULARITY_IMAGE={self.config.singularity_image}\n"
        
        # Main execution command
        script += f"""
# Create output directory
mkdir -p {output_path}

# Export environment variables
export QUANTSIM_NUM_SIMS={num_scenarios}
export QUANTSIM_DATA_PATH={market_data_path}
export QUANTSIM_OUTPUT_PATH={output_path}
export QUANTSIM_USE_MPI=true

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Run simulation
echo "Starting HPC QuantSim simulation at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"

mpirun -np $SLURM_NTASKS python -m hpc_quantsim.cli run-distributed \\
    --scenarios {num_scenarios} \\
    --data-path {market_data_path} \\
    --output-path {output_path} \\
    --config-path job_configs/{job_name}_config.yaml

echo "Simulation completed at $(date)"

# Collect performance statistics
if command -v sacct &> /dev/null; then
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State > {output_path}/job_stats.txt
fi
"""
        
        return script
    
    def _generate_pbs_script(self, job_name: str, nodes: int, runtime: str,
                            num_scenarios: int, strategy_configs: List[Dict],
                            market_data_path: str, output_path: str) -> str:
        """Generate PBS job script."""
        
        total_cores = nodes * self.config.tasks_per_node
        
        script = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -l nodes={nodes}:ppn={self.config.tasks_per_node}
#PBS -l walltime={runtime}
#PBS -l mem={self.config.memory_per_node_gb}gb
#PBS -o {output_path}/pbs_output.txt
#PBS -e {output_path}/pbs_error.txt
#PBS -V
"""
        
        # Add modules and environment setup
        if self.config.modules_to_load:
            script += "\n# Load required modules\n"
            for module in self.config.modules_to_load:
                script += f"module load {module}\n"
        
        if self.config.conda_env:
            script += f"\n# Activate conda environment\nsource activate {self.config.conda_env}\n"
        
        # Main execution
        script += f"""
# Create output directory
mkdir -p {output_path}

# Export environment variables
export QUANTSIM_NUM_SIMS={num_scenarios}
export QUANTSIM_DATA_PATH={market_data_path}
export QUANTSIM_OUTPUT_PATH={output_path}
export QUANTSIM_USE_MPI=true

# Change to working directory
cd $PBS_O_WORKDIR

# Run simulation
echo "Starting HPC QuantSim simulation at $(date)"
echo "Job ID: $PBS_JOBID"

mpirun -np {total_cores} python -m hpc_quantsim.cli run-distributed \\
    --scenarios {num_scenarios} \\
    --data-path {market_data_path} \\
    --output-path {output_path} \\
    --config-path job_configs/{job_name}_config.yaml

echo "Simulation completed at $(date)"
"""
        
        return script
    
    def _submit_job_script(self, script: str, job_name: str) -> Optional[str]:
        """Submit job script to scheduler."""
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            if self.scheduler == 'slurm':
                result = subprocess.run(['sbatch', script_path], 
                                      capture_output=True, text=True, check=True)
                # Extract job ID from SLURM output
                job_id = result.stdout.strip().split()[-1]
                
            elif self.scheduler == 'pbs':
                result = subprocess.run(['qsub', script_path],
                                      capture_output=True, text=True, check=True)
                job_id = result.stdout.strip()
            
            else:
                raise ValueError(f"Unsupported scheduler: {self.scheduler}")
            
            return job_id
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Job submission failed: {e.stderr}")
            return None
            
        finally:
            # Clean up temporary script file
            try:
                os.unlink(script_path)
            except Exception:
                pass
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of submitted job."""
        
        if self.scheduler == 'slurm':
            return self._get_slurm_job_status(job_id)
        elif self.scheduler == 'pbs':
            return self._get_pbs_job_status(job_id)
        else:
            return {'status': 'unknown', 'reason': 'No scheduler available'}
    
    def _get_slurm_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get SLURM job status."""
        try:
            result = subprocess.run(['squeue', '-j', job_id, '--format=%T,%R,%S,%M'],
                                  capture_output=True, text=True, check=True)
            
            if len(result.stdout.strip().split('\n')) < 2:
                # Job not in queue, check completed jobs
                result = subprocess.run(['sacct', '-j', job_id, '--format=State,Elapsed,ExitCode', '-n'],
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    fields = result.stdout.strip().split()
                    status = fields[0] if fields else 'UNKNOWN'
                    elapsed = fields[1] if len(fields) > 1 else '00:00:00'
                    exit_code = fields[2] if len(fields) > 2 else '0:0'
                    
                    return {
                        'status': status.lower(),
                        'elapsed_time': elapsed,
                        'exit_code': exit_code,
                        'completed': True
                    }
                else:
                    return {'status': 'not_found', 'completed': True}
            
            # Parse squeue output
            fields = result.stdout.strip().split('\n')[1].split(',')
            status = fields[0]
            reason = fields[1]
            start_time = fields[2]
            time_used = fields[3]
            
            return {
                'status': status.lower(),
                'reason': reason,
                'start_time': start_time,
                'time_used': time_used,
                'completed': False
            }
            
        except subprocess.CalledProcessError:
            return {'status': 'error', 'reason': 'Failed to query job status'}
    
    def _get_pbs_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get PBS job status."""
        try:
            result = subprocess.run(['qstat', job_id], 
                                  capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            if len(lines) < 3:
                return {'status': 'completed', 'completed': True}
            
            # Parse qstat output
            job_line = lines[2]  # Skip header lines
            fields = job_line.split()
            status = fields[4] if len(fields) > 4 else 'U'  # Unknown
            
            status_map = {
                'Q': 'pending',
                'R': 'running', 
                'C': 'completed',
                'E': 'exiting',
                'H': 'held',
            }
            
            return {
                'status': status_map.get(status, 'unknown'),
                'completed': status in ['C', 'E']
            }
            
        except subprocess.CalledProcessError:
            return {'status': 'not_found', 'completed': True}
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel running job."""
        try:
            if self.scheduler == 'slurm':
                subprocess.run(['scancel', job_id], check=True)
            elif self.scheduler == 'pbs':
                subprocess.run(['qdel', job_id], check=True)
            else:
                return False
            
            # Update job tracking
            if job_id in self.active_jobs:
                self.active_jobs[job_id]['status'] = 'cancelled'
            
            self.logger.info(f"Job {job_id} cancelled successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """List all active jobs."""
        active = []
        
        for job_id, job_info in self.active_jobs.items():
            # Update status
            current_status = self.get_job_status(job_id)
            job_info.update(current_status)
            
            # Move completed jobs to history
            if current_status.get('completed', False):
                self.job_history.append(job_info.copy())
                
                # Update statistics
                if current_status.get('status') == 'completed':
                    self.cluster_stats['jobs_completed'] += 1
                else:
                    self.cluster_stats['jobs_failed'] += 1
                    
                del self.active_jobs[job_id]
            else:
                active.append(job_info)
        
        return active
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information and resources."""
        if self.scheduler == 'slurm':
            return self._get_slurm_cluster_info()
        elif self.scheduler == 'pbs':
            return self._get_pbs_cluster_info()
        else:
            return {'scheduler': 'none', 'nodes': 0, 'available': False}
    
    def _get_slurm_cluster_info(self) -> Dict[str, Any]:
        """Get SLURM cluster information."""
        try:
            # Get node information
            result = subprocess.run(['sinfo', '--format=%N,%C,%m,%f,%T', '--noheader'],
                                  capture_output=True, text=True, check=True)
            
            nodes = []
            total_cores = 0
            total_memory = 0
            available_nodes = 0
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    fields = line.split(',')
                    if len(fields) >= 5:
                        node_name = fields[0]
                        cores_info = fields[1]  # Format: allocated/idle/other/total
                        memory = fields[2]
                        features = fields[3]
                        state = fields[4]
                        
                        # Parse cores (extract total)
                        cores = cores_info.split('/')[-1] if '/' in cores_info else cores_info
                        
                        nodes.append({
                            'name': node_name,
                            'cores': cores,
                            'memory': memory,
                            'features': features,
                            'state': state
                        })
                        
                        if cores.isdigit():
                            total_cores += int(cores)
                        if state in ['idle', 'mix', 'alloc']:
                            available_nodes += 1
            
            return {
                'scheduler': 'slurm',
                'total_nodes': len(nodes),
                'available_nodes': available_nodes,
                'total_cores': total_cores,
                'nodes': nodes,
                'available': True
            }
            
        except subprocess.CalledProcessError:
            return {'scheduler': 'slurm', 'available': False, 'error': 'Failed to query cluster'}
    
    def _get_pbs_cluster_info(self) -> Dict[str, Any]:
        """Get PBS cluster information."""
        try:
            result = subprocess.run(['pbsnodes'], capture_output=True, text=True, check=True)
            
            # Parse pbsnodes output (simplified)
            nodes = []
            current_node = None
            
            for line in result.stdout.split('\n'):
                line = line.strip()
                if not line.startswith(' ') and line:
                    # New node
                    if current_node:
                        nodes.append(current_node)
                    current_node = {'name': line, 'state': 'unknown'}
                elif line.startswith('state = '):
                    if current_node:
                        current_node['state'] = line.split(' = ')[1]
            
            if current_node:
                nodes.append(current_node)
            
            available_nodes = sum(1 for n in nodes if n['state'] in ['free', 'job-exclusive'])
            
            return {
                'scheduler': 'pbs',
                'total_nodes': len(nodes),
                'available_nodes': available_nodes,
                'nodes': nodes,
                'available': True
            }
            
        except subprocess.CalledProcessError:
            return {'scheduler': 'pbs', 'available': False, 'error': 'Failed to query cluster'}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cluster management statistics."""
        return {
            'cluster_stats': self.cluster_stats.copy(),
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.job_history),
            'scheduler': self.scheduler,
            'last_update': datetime.now().isoformat(),
        }
    
    def cleanup_old_jobs(self, max_age_days: int = 7):
        """Clean up old job history entries."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        old_count = len(self.job_history)
        self.job_history = [
            job for job in self.job_history 
            if job.get('submitted_at', datetime.now()) > cutoff_date
        ]
        
        removed = old_count - len(self.job_history)
        if removed > 0:
            self.logger.info(f"Cleaned up {removed} old job history entries")


def get_cluster_manager(config: Optional[DeploymentConfig] = None) -> Optional[ClusterManager]:
    """Get cluster manager instance."""
    if config is None:
        from ..config import create_default_config
        config = create_default_config().deployment
    
    try:
        return ClusterManager(config)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create cluster manager: {e}")
        return None
