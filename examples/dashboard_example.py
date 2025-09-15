#!/usr/bin/env python3
"""
Example script to demonstrate the HPC QuantSim Dashboard.

This script shows how to:
1. Start the dashboard server
2. Run simulations with real-time monitoring
3. View results through the web interface
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hpc_quantsim.config import create_default_config
from hpc_quantsim.core.simulation_engine import SimulationEngine
from hpc_quantsim.dashboard.app import create_dashboard_app
import uvicorn


def create_sample_config():
    """Create a sample configuration for dashboard testing."""
    config = create_default_config()
    
    # Configure simulation parameters
    config.simulation.num_simulations = 50
    config.simulation.strategy = "MovingAverage"
    config.simulation.start_date = "2023-01-01"
    config.simulation.end_date = "2023-12-31"
    
    # Configure market data
    config.market.data_path = "examples/sample_data.parquet"  # You'd need to provide this
    config.market.replay_speed = 1.0
    config.market.use_microstructure = False
    
    # Configure performance settings
    config.hpc.use_gpu = False  # Set to True if you have CUDA
    config.hpc.use_mpi = False  # Set to True for distributed execution
    config.hpc.num_workers = 4
    
    return config


def run_dashboard_with_simulation():
    """
    Run the dashboard with a pre-configured simulation engine.
    
    This demonstrates how to integrate the dashboard with your simulation system.
    """
    print("🚀 Starting HPC QuantSim Dashboard Example")
    print("=" * 50)
    
    # Create configuration
    config = create_sample_config()
    print(f"✓ Created sample configuration")
    
    # Initialize simulation engine
    try:
        simulation_engine = SimulationEngine(config)
        print(f"✓ Initialized simulation engine")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize simulation engine: {e}")
        print("   Dashboard will start without simulation integration")
        simulation_engine = None
    
    # Create dashboard app
    app = create_dashboard_app(simulation_engine)
    print(f"✓ Created dashboard application")
    
    # Start the server
    print("\n🌐 Starting dashboard server...")
    print("   Dashboard URL: http://localhost:8000")
    print("   Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down dashboard...")


def run_simulation_in_background(simulation_engine: SimulationEngine):
    """
    Run a simulation in the background to demonstrate real-time updates.
    
    In a real deployment, this would be triggered through the dashboard UI.
    """
    import threading
    import time
    
    def simulation_worker():
        print("Starting background simulation...")
        try:
            # This would run your actual simulation
            results = simulation_engine.run_simulation()
            print(f"Simulation completed. Generated {len(results)} results.")
        except Exception as e:
            print(f"Simulation error: {e}")
    
    # Start simulation in background thread
    thread = threading.Thread(target=simulation_worker)
    thread.daemon = True
    thread.start()


if __name__ == "__main__":
    print("HPC QuantSim Dashboard Example")
    print("Choose an option:")
    print("1. Start dashboard only (no simulation engine)")
    print("2. Start dashboard with simulation engine integration")
    print("3. Run simulation and dashboard together")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Simple dashboard without simulation engine
        from hpc_quantsim.dashboard.app import run_dashboard
        run_dashboard(host="0.0.0.0", port=8000)
        
    elif choice == "2":
        # Dashboard with simulation engine integration
        run_dashboard_with_simulation()
        
    elif choice == "3":
        # Full integration with background simulation
        config = create_sample_config()
        simulation_engine = SimulationEngine(config)
        
        # Start dashboard
        app = create_dashboard_app(simulation_engine)
        
        # Start background simulation
        run_simulation_in_background(simulation_engine)
        
        # Start dashboard server
        print("🌐 Starting integrated dashboard...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

