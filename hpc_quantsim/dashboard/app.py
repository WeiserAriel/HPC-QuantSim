"""
FastAPI dashboard application for HPC QuantSim.

Provides real-time monitoring, interactive charts, and strategy comparison.
"""

import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from datetime import datetime, timedelta
import logging

from ..core.simulation_engine import SimulationEngine, SimulationResult
from ..metrics.performance_metrics import PerformanceMetrics
from ..metrics.metric_aggregator import MetricAggregator
from ..gpu.gpu_utils import GPUUtils
from ..hpc.mpi_collectives import HAS_MPI
from .websocket_manager import WebSocketManager
from .chart_generators import ChartGenerator

logger = logging.getLogger(__name__)


async def _run_simulation_background(
    simulation_engine: SimulationEngine,
    config: Dict[str, Any],
    simulation_id: str,
    websocket_manager: WebSocketManager,
    app_state: Any
) -> None:
    """Run simulation in background with progress updates."""
    try:
        # Update engine config if provided
        if config.get("num_simulations"):
            simulation_engine.config.num_simulations = config["num_simulations"]
        if config.get("max_parallel"):
            simulation_engine.config.max_parallel = config["max_parallel"]
            
        # Load market data if not already loaded
        if not simulation_engine.market_replay:
            from ..market import MarketReplay
            market_replay = MarketReplay()
            # Use default or example data if no specific data provided
            if config.get("data_path"):
                await market_replay.load_data(config["data_path"])
            else:
                # Load sample data for demo
                logger.info("Loading sample market data for simulation")
            simulation_engine.market_replay = market_replay
        
        # Register default strategies if none exist
        if not simulation_engine.strategies:
            from ..core.strategy_interface import MovingAverageStrategy, MeanReversionStrategy
            simulation_engine.register_strategy(MovingAverageStrategy(), "MovingAverage")
            simulation_engine.register_strategy(MeanReversionStrategy(), "MeanReversion")
        
        # Run simulation with periodic progress updates
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(simulation_engine.run_simulation)
            
            # Monitor progress
            while not future.done() and not app_state.stop_simulation_flag:
                await asyncio.sleep(1)  # Check every second
                
                # Send progress update
                if simulation_engine.is_running:
                    progress_info = simulation_engine.get_progress()
                    await websocket_manager.broadcast({
                        "type": "simulation_progress",
                        "simulation_id": simulation_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "progress": progress_info["progress"],
                        "completed_tasks": progress_info["completed_tasks"],
                        "total_tasks": progress_info["total_tasks"],
                        "status": "running"
                    })
                
                # Check if stop was requested
                if app_state.stop_simulation_flag:
                    logger.info(f"Simulation {simulation_id} stop requested")
                    break
            
            if app_state.stop_simulation_flag:
                # Cancel the future if possible
                future.cancel()
                simulation_engine.is_running = False
                await websocket_manager.broadcast({
                    "type": "simulation_cancelled",
                    "simulation_id": simulation_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                return
                
            # Get results
            results = future.result()
            app_state.simulation_results.extend(results)
            
            # Send completion notification
            await websocket_manager.broadcast({
                "type": "simulation_completed",
                "simulation_id": simulation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "results_count": len(results),
                "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0
            })
            
    except Exception as e:
        logger.error(f"Background simulation error: {e}")
        await websocket_manager.broadcast({
            "type": "simulation_error",
            "simulation_id": simulation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        })
    finally:
        # Clean up
        simulation_engine.is_running = False
        if simulation_id in app_state.background_tasks:
            del app_state.background_tasks[simulation_id]


def create_dashboard_app(simulation_engine: Optional[SimulationEngine] = None) -> FastAPI:
    """Create and configure the dashboard FastAPI application."""
    
    app = FastAPI(
        title="HPC QuantSim Dashboard",
        description="Real-time monitoring and analysis for quantitative simulations",
        version="1.0.0"
    )
    
    # Initialize dashboard components
    websocket_manager = WebSocketManager()
    chart_generator = ChartGenerator()
    
    # Store simulation engine reference
    app.state.simulation_engine = simulation_engine
    app.state.websocket_manager = websocket_manager
    app.state.chart_generator = chart_generator
    app.state.simulation_results: List[SimulationResult] = []
    app.state.live_metrics: Dict[str, Any] = {}
    app.state.background_tasks: Dict[str, asyncio.Task] = {}
    app.state.stop_simulation_flag = False
    
    # Configure templates and static files
    templates = Jinja2Templates(directory="hpc_quantsim/dashboard/templates")
    
    # Mount static files
    from pathlib import Path
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request):
        """Main dashboard page."""
        return templates.TemplateResponse(
            "dashboard.html", 
            {"request": request, "title": "HPC QuantSim Dashboard"}
        )
    
    @app.get("/api/status")
    async def get_status():
        """Get current system status."""
        engine = app.state.simulation_engine
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "engine_available": engine is not None,
            "active_connections": len(app.state.websocket_manager.active_connections),
            "total_simulations": len(app.state.simulation_results),
            "system_info": {
                "gpu_available": GPUUtils.check_gpu_availability(),
                "mpi_available": HAS_MPI,
                "gpu_info": GPUUtils.get_device_info() if GPUUtils.check_gpu_availability() else None
            }
        }
        
        if engine and hasattr(engine, 'execution_state'):
            status.update({
                "is_running": engine.execution_state.get("is_running", False),
                "progress": engine.execution_state.get("progress", 0.0),
                "current_scenario": engine.execution_state.get("current_scenario", 0),
                "total_scenarios": engine.execution_state.get("total_scenarios", 0),
            })
        
        return status
    
    @app.get("/api/metrics/summary")
    async def get_metrics_summary():
        """Get aggregated performance metrics summary."""
        results = app.state.simulation_results
        if not results:
            return {"message": "No simulation results available"}
        
        # Calculate summary statistics
        pnl_values = [r.final_pnl for r in results if r.success]
        sharpe_values = [r.sharpe_ratio for r in results if r.success and r.sharpe_ratio is not None]
        drawdown_values = [r.max_drawdown for r in results if r.success]
        
        summary = {
            "total_simulations": len(results),
            "successful_simulations": len(pnl_values),
            "success_rate": len(pnl_values) / len(results) if results else 0,
            "pnl_stats": {
                "mean": float(np.mean(pnl_values)) if pnl_values else 0,
                "std": float(np.std(pnl_values)) if pnl_values else 0,
                "min": float(np.min(pnl_values)) if pnl_values else 0,
                "max": float(np.max(pnl_values)) if pnl_values else 0,
            },
            "sharpe_stats": {
                "mean": float(np.mean(sharpe_values)) if sharpe_values else 0,
                "std": float(np.std(sharpe_values)) if sharpe_values else 0,
            } if sharpe_values else None,
            "drawdown_stats": {
                "mean": float(np.mean(drawdown_values)) if drawdown_values else 0,
                "max": float(np.max(drawdown_values)) if drawdown_values else 0,
            }
        }
        
        return summary
    
    @app.get("/api/charts/pnl-distribution")
    async def get_pnl_distribution_chart():
        """Generate PnL distribution chart data."""
        results = app.state.simulation_results
        chart_data = app.state.chart_generator.create_pnl_distribution(results)
        return chart_data
    
    @app.get("/api/charts/performance-over-time")
    async def get_performance_chart():
        """Generate performance over time chart data."""
        results = app.state.simulation_results
        chart_data = app.state.chart_generator.create_performance_timeline(results)
        return chart_data
    
    @app.get("/api/charts/strategy-comparison")
    async def get_strategy_comparison():
        """Generate strategy comparison chart data."""
        results = app.state.simulation_results
        chart_data = app.state.chart_generator.create_strategy_comparison(results)
        return chart_data
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await app.state.websocket_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif message.get("type") == "subscribe":
                    # Handle subscription to specific data streams
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "channels": message.get("channels", [])
                    }))
                    
        except WebSocketDisconnect:
            app.state.websocket_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            app.state.websocket_manager.disconnect(websocket)
    
    @app.post("/api/simulation/start")
    async def start_simulation(config: Dict[str, Any]):
        """Start a new simulation with given configuration."""
        if not app.state.simulation_engine:
            return {"error": "Simulation engine not available"}
        
        # Check if simulation is already running
        if app.state.simulation_engine.is_running:
            return {"error": "Simulation already running"}
        
        try:
            simulation_id = f"sim_{int(time.time())}"
            app.state.stop_simulation_flag = False
            
            # Start simulation in background task
            task = asyncio.create_task(
                _run_simulation_background(
                    app.state.simulation_engine,
                    config,
                    simulation_id,
                    app.state.websocket_manager,
                    app.state
                )
            )
            app.state.background_tasks[simulation_id] = task
            
            await app.state.websocket_manager.broadcast({
                "type": "simulation_started",
                "timestamp": datetime.utcnow().isoformat(),
                "simulation_id": simulation_id,
                "config": config
            })
            
            return {"message": "Simulation started", "simulation_id": simulation_id}
            
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            return {"error": str(e)}
    
    @app.post("/api/simulation/stop")
    async def stop_simulation():
        """Stop the current simulation."""
        if not app.state.simulation_engine:
            return {"error": "Simulation engine not available"}
        
        if not app.state.simulation_engine.is_running:
            return {"error": "No simulation running"}
        
        try:
            # Set stop flag
            app.state.stop_simulation_flag = True
            
            # Cancel background tasks
            for task_id, task in app.state.background_tasks.items():
                if not task.done():
                    task.cancel()
            
            # Clear completed tasks
            app.state.background_tasks = {k: v for k, v in app.state.background_tasks.items() if not v.done()}
            
            await app.state.websocket_manager.broadcast({
                "type": "simulation_stopped",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "User requested stop"
            })
            
            return {"message": "Simulation stop requested"}
            
        except Exception as e:
            logger.error(f"Failed to stop simulation: {e}")
            return {"error": str(e)}
    
    return app


def run_dashboard(host: str = "0.0.0.0", port: int = 8000, simulation_engine: Optional[SimulationEngine] = None):
    """Run the dashboard server."""
    app = create_dashboard_app(simulation_engine)
    
    logger.info(f"Starting HPC QuantSim Dashboard on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_dashboard()
