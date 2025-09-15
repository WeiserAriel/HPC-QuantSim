// HPC QuantSim Dashboard JavaScript

class QuantSimDashboard {
    constructor() {
        this.websocket = null;
        this.reconnectInterval = 5000;
        this.maxReconnectAttempts = 10;
        this.reconnectAttempts = 0;
        
        this.init();
    }
    
    init() {
        console.log('Initializing HPC QuantSim Dashboard...');
        this.setupEventListeners();
        this.connectWebSocket();
        this.updateConnectionStatus('connecting');
    }
    
    setupEventListeners() {
        // Start simulation button
        const startBtn = document.getElementById('start-simulation');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startSimulation());
        }
        
        // Stop simulation button
        const stopBtn = document.getElementById('stop-simulation');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopSimulation());
        }
        
        // Refresh button
        const refreshBtn = document.getElementById('refresh-data');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }
        
        // Clear logs button
        const clearLogsBtn = document.getElementById('clear-logs');
        if (clearLogsBtn) {
            clearLogsBtn.addEventListener('click', () => this.clearLogs());
        }
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('connected');
            this.refreshData();
        };
        
        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus('disconnected');
            this.attemptReconnect();
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('error');
        };
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => this.connectWebSocket(), this.reconnectInterval);
        } else {
            console.error('Max reconnection attempts reached');
            this.updateConnectionStatus('failed');
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'simulation_started':
                this.onSimulationStarted(data);
                break;
            case 'simulation_progress':
                this.onSimulationProgress(data);
                break;
            case 'simulation_completed':
                this.onSimulationCompleted(data);
                break;
            case 'simulation_stopped':
                this.onSimulationStopped(data);
                break;
            case 'simulation_error':
                this.onSimulationError(data);
                break;
            case 'metrics_update':
                this.onMetricsUpdate(data);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    onSimulationStarted(data) {
        this.logMessage('info', `Simulation started: ${data.simulation_id}`);
        this.updateSimulationStatus('running');
        this.updateProgressBar(0);
    }
    
    onSimulationProgress(data) {
        const progress = Math.round(data.progress * 100);
        this.updateProgressBar(progress);
        this.logMessage('info', `Progress: ${progress}%`);
    }
    
    onSimulationCompleted(data) {
        this.logMessage('success', `Simulation completed! Results: ${data.results_count}`);
        this.updateSimulationStatus('completed');
        this.updateProgressBar(100);
        this.refreshData();
    }
    
    onSimulationStopped(data) {
        this.logMessage('warning', `Simulation stopped: ${data.reason || 'Unknown reason'}`);
        this.updateSimulationStatus('stopped');
    }
    
    onSimulationError(data) {
        this.logMessage('error', `Simulation error: ${data.error}`);
        this.updateSimulationStatus('error');
    }
    
    onMetricsUpdate(data) {
        this.updateMetrics(data);
    }
    
    async startSimulation() {
        try {
            const config = {
                num_simulations: parseInt(document.getElementById('num-simulations')?.value) || 1000,
                max_parallel: parseInt(document.getElementById('max-parallel')?.value) || 16
            };
            
            const response = await fetch('/api/simulation/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.logMessage('success', result.message);
            } else {
                this.logMessage('error', result.error || 'Failed to start simulation');
            }
        } catch (error) {
            this.logMessage('error', `Failed to start simulation: ${error.message}`);
        }
    }
    
    async stopSimulation() {
        try {
            const response = await fetch('/api/simulation/stop', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.logMessage('success', result.message);
            } else {
                this.logMessage('error', result.error || 'Failed to stop simulation');
            }
        } catch (error) {
            this.logMessage('error', `Failed to stop simulation: ${error.message}`);
        }
    }
    
    async refreshData() {
        try {
            // Fetch system status
            const statusResponse = await fetch('/api/status');
            if (statusResponse.ok) {
                const status = await statusResponse.json();
                this.updateSystemInfo(status);
            }
            
            // Fetch metrics summary
            const metricsResponse = await fetch('/api/metrics/summary');
            if (metricsResponse.ok) {
                const metrics = await metricsResponse.json();
                this.updateMetrics(metrics);
            }
        } catch (error) {
            console.error('Failed to refresh data:', error);
            this.logMessage('error', 'Failed to refresh data');
        }
    }
    
    updateConnectionStatus(status) {
        const statusEl = document.getElementById('connection-status');
        if (statusEl) {
            statusEl.className = `connection-status connection-${status}`;
            
            const statusText = {
                connecting: 'Connecting...',
                connected: 'Connected',
                disconnected: 'Disconnected',
                error: 'Connection Error',
                failed: 'Connection Failed'
            };
            
            statusEl.textContent = statusText[status] || status;
        }
    }
    
    updateSimulationStatus(status) {
        const statusEl = document.getElementById('simulation-status');
        if (statusEl) {
            statusEl.className = `status status-${status}`;
            statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        }
        
        // Update buttons based on status
        const startBtn = document.getElementById('start-simulation');
        const stopBtn = document.getElementById('stop-simulation');
        
        if (startBtn && stopBtn) {
            if (status === 'running') {
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        }
    }
    
    updateProgressBar(progress) {
        const progressBar = document.getElementById('progress-bar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
        
        const progressText = document.getElementById('progress-text');
        if (progressText) {
            progressText.textContent = `${progress}%`;
        }
    }
    
    updateSystemInfo(status) {
        // Update various system info elements
        const elements = {
            'gpu-available': status.system_info?.gpu_available ? 'Yes' : 'No',
            'mpi-available': status.system_info?.mpi_available ? 'Yes' : 'No',
            'active-connections': status.active_connections || 0,
            'total-simulations': status.total_simulations || 0
        };
        
        for (const [id, value] of Object.entries(elements)) {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = value;
            }
        }
        
        // Update simulation status if available
        if (status.is_running !== undefined) {
            this.updateSimulationStatus(status.is_running ? 'running' : 'stopped');
        }
        
        if (status.progress !== undefined) {
            this.updateProgressBar(Math.round(status.progress * 100));
        }
    }
    
    updateMetrics(metrics) {
        if (!metrics || metrics.message) {
            return; // No metrics available
        }
        
        // Update metric cards
        const metricElements = {
            'mean-pnl': metrics.pnl_statistics?.mean?.toFixed(2) || '0.00',
            'sharpe-ratio': metrics.sharpe_statistics?.mean?.toFixed(2) || '0.00',
            'success-rate': `${((metrics.success_rate || 0) * 100).toFixed(1)}%`,
            'max-drawdown': `${((metrics.drawdown_statistics?.max || 0) * 100).toFixed(1)}%`
        };
        
        for (const [id, value] of Object.entries(metricElements)) {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = value;
            }
        }
    }
    
    logMessage(level, message) {
        const logContainer = document.getElementById('log-container');
        if (!logContainer) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${level}`;
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
        
        // Limit log entries to prevent memory issues
        const entries = logContainer.children;
        if (entries.length > 100) {
            logContainer.removeChild(entries[0]);
        }
    }
    
    clearLogs() {
        const logContainer = document.getElementById('log-container');
        if (logContainer) {
            logContainer.innerHTML = '';
        }
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new QuantSimDashboard();
});

// Export for potential use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = QuantSimDashboard;
}
