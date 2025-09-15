# HPC QuantSim Dashboard 📊

A real-time web-based visualization dashboard for monitoring and analyzing quantitative trading simulations.

## ✨ Features

### 🔄 Real-Time Monitoring
- **Live Metrics**: Real-time PnL, Sharpe ratio, drawdown tracking
- **Progress Monitoring**: Visual progress bars and execution status
- **WebSocket Integration**: Sub-second update latency

### 📈 Interactive Visualizations
- **PnL Distribution**: Histogram analysis of profit/loss across simulations
- **Performance Timeline**: Track performance metrics over time
- **Strategy Comparison**: Side-by-side comparison of trading strategies
- **Responsive Charts**: Built with Plotly.js for interactive exploration

### 🎛️ Control Interface
- **Simulation Control**: Start/stop simulations from the web UI
- **Configuration Management**: Dynamic parameter adjustment
- **System Status**: Monitor GPU, MPI, and cluster resources

### 🎨 Modern UI
- **Dark Theme**: Professional trading-focused design
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Real-Time Updates**: Live data streams without page refresh

## 🚀 Quick Start

### Method 1: Command Line Interface
```bash
# Start dashboard with default settings
hpc-quantsim dashboard

# Start with custom host/port
hpc-quantsim dashboard --host 0.0.0.0 --port 8080

# Start with simulation engine integration
hpc-quantsim dashboard --config examples/config.yaml
```

### Method 2: Python API
```python
from hpc_quantsim.dashboard.app import run_dashboard
from hpc_quantsim.core.simulation_engine import SimulationEngine
from hpc_quantsim.config import load_config

# Load configuration
config = load_config("config.yaml")
simulation_engine = SimulationEngine(config)

# Start dashboard
run_dashboard(
    host="0.0.0.0",
    port=8000,
    simulation_engine=simulation_engine
)
```

### Method 3: Example Script
```bash
# Run the complete example
python examples/dashboard_example.py
```

## 🌐 Dashboard Interface

Once started, the dashboard will be available at `http://localhost:8000`

### Main Components

#### 1. **Metrics Overview**
- Total simulations run
- Success rate percentage  
- Mean PnL across all simulations
- Average Sharpe ratio

#### 2. **Simulation Control Panel**
- **Start/Stop**: Control simulation execution
- **Progress Bar**: Visual progress indicator
- **Refresh**: Manually update data

#### 3. **Interactive Charts**
- **PnL Distribution**: Analyze profit/loss patterns
- **Performance Timeline**: Track metrics over time
- **Strategy Comparison**: Compare multiple strategies
- **Real-Time Updates**: Charts update automatically

#### 4. **System Log**
- Real-time system messages
- Error reporting
- Connection status
- Simulation events

## ⚙️ Configuration

### Dashboard Settings
```yaml
dashboard:
  host: "0.0.0.0"
  port: 8000
  auto_refresh: true
  max_log_entries: 100
  chart_theme: "plotly_dark"
```

### WebSocket Configuration  
```yaml
websocket:
  heartbeat_interval: 30
  max_connections: 100
  channels:
    - metrics
    - simulation  
    - results
```

## 🔌 API Endpoints

### REST API
- `GET /` - Dashboard home page
- `GET /api/status` - System status
- `GET /api/metrics/summary` - Performance summary
- `GET /api/charts/pnl-distribution` - PnL chart data
- `GET /api/charts/performance-over-time` - Timeline data
- `GET /api/charts/strategy-comparison` - Strategy comparison
- `POST /api/simulation/start` - Start simulation
- `POST /api/simulation/stop` - Stop simulation

### WebSocket
- `ws://host:port/ws` - Real-time data stream
- **Channels**: `metrics`, `simulation`, `results`
- **Message Types**: `metrics_update`, `simulation_progress`, `new_result`

## 🔧 Advanced Usage

### Custom Chart Integration
```python
from hpc_quantsim.dashboard.chart_generators import ChartGenerator

# Create custom chart generator
chart_gen = ChartGenerator(theme="plotly_white")

# Generate custom visualization
custom_chart = chart_gen.create_custom_chart(your_data)
```

### WebSocket Broadcasting
```python
from hpc_quantsim.dashboard.websocket_manager import WebSocketManager

# Broadcast custom metrics
websocket_manager = WebSocketManager()
await websocket_manager.broadcast_metrics_update({
    "custom_metric": value,
    "timestamp": datetime.utcnow().isoformat()
})
```

### Integration with Simulation Engine
```python
# Hook into simulation events for real-time updates
class DashboardSimulationEngine(SimulationEngine):
    def __init__(self, config, websocket_manager):
        super().__init__(config)
        self.websocket_manager = websocket_manager
    
    async def on_simulation_progress(self, progress_data):
        # Broadcast progress to dashboard
        await self.websocket_manager.broadcast_simulation_progress(progress_data)
    
    async def on_new_result(self, result):
        # Broadcast new results to dashboard
        await self.websocket_manager.broadcast_new_result(result.to_dict())
```

## 🐛 Troubleshooting

### Common Issues

#### Dashboard won't start
```bash
# Check if port is available
lsof -i :8000

# Try different port
hpc-quantsim dashboard --port 8080
```

#### WebSocket connection fails
- Check firewall settings
- Verify host/port configuration
- Check browser developer console for errors

#### Charts not loading
- Ensure Plotly.js is accessible
- Check browser compatibility
- Verify API endpoints are responding

### Performance Optimization

#### High-Frequency Updates
```yaml
# Reduce update frequency for better performance
dashboard:
  update_interval: 1000  # milliseconds
  max_data_points: 1000
  enable_chart_streaming: false
```

#### Memory Management
```yaml
# Limit data retention
dashboard:
  max_results_history: 10000
  cleanup_interval: 300  # seconds
  enable_data_compression: true
```

## 🚦 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["hpc-quantsim", "dashboard", "--host", "0.0.0.0"]
```

### NGINX Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Security Considerations
- Enable HTTPS in production
- Implement authentication/authorization
- Use environment variables for secrets
- Configure CORS appropriately
- Implement rate limiting

## 📝 Development

### Adding Custom Metrics
1. Extend the `PerformanceMetrics` class
2. Update the WebSocket message handlers
3. Add new chart types in `ChartGenerator`
4. Update the dashboard UI

### Custom Chart Types
```python
class CustomChartGenerator(ChartGenerator):
    def create_risk_heatmap(self, risk_data):
        """Create custom risk analysis heatmap."""
        # Your custom chart logic here
        return plotly_figure.to_dict()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This dashboard is part of the HPC QuantSim project and follows the same license terms.

