"""
Chart generation utilities for the HPC QuantSim dashboard.

Creates interactive Plotly charts for performance visualization.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging

from ..core.simulation_engine import SimulationResult
from ..metrics.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generates interactive charts for simulation results and metrics."""
    
    def __init__(self, theme: str = "plotly_dark"):
        """Initialize the chart generator with default styling."""
        self.theme = theme
        self.color_palette = [
            '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
            '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
        ]
    
    def create_pnl_distribution(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Create PnL distribution histogram."""
        if not results:
            return self._empty_chart("No results available")
        
        # Extract PnL values
        successful_results = [r for r in results if r.success]
        pnl_values = [r.final_pnl for r in successful_results]
        
        if not pnl_values:
            return self._empty_chart("No successful simulations")
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pnl_values,
            nbinsx=50,
            name="PnL Distribution",
            marker_color=self.color_palette[0],
            opacity=0.7
        ))
        
        # Add statistical lines
        mean_pnl = np.mean(pnl_values)
        median_pnl = np.median(pnl_values)
        
        fig.add_vline(x=mean_pnl, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: ${mean_pnl:,.2f}")
        fig.add_vline(x=median_pnl, line_dash="dash", line_color="green",
                     annotation_text=f"Median: ${median_pnl:,.2f}")
        
        fig.update_layout(
            title="PnL Distribution Across Simulations",
            xaxis_title="Final PnL ($)",
            yaxis_title="Frequency",
            template=self.theme,
            showlegend=False
        )
        
        return fig.to_dict()
    
    def create_performance_timeline(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Create performance metrics timeline chart."""
        if not results:
            return self._empty_chart("No results available")
        
        # Sort results by scenario_id to create timeline
        sorted_results = sorted(results, key=lambda x: x.scenario_id)
        successful_results = [r for r in sorted_results if r.success]
        
        if not successful_results:
            return self._empty_chart("No successful simulations")
        
        # Extract data for timeline
        scenario_ids = [r.scenario_id for r in successful_results]
        pnl_values = [r.final_pnl for r in successful_results]
        sharpe_values = [r.sharpe_ratio for r in successful_results if r.sharpe_ratio is not None]
        drawdown_values = [r.max_drawdown for r in successful_results]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=['PnL Over Time', 'Sharpe Ratio', 'Maximum Drawdown'],
            vertical_spacing=0.05
        )
        
        # PnL line chart
        fig.add_trace(go.Scatter(
            x=scenario_ids,
            y=pnl_values,
            mode='lines+markers',
            name='PnL',
            line=dict(color=self.color_palette[0]),
            marker=dict(size=3)
        ), row=1, col=1)
        
        # Sharpe ratio (if available)
        if len(sharpe_values) == len(scenario_ids):
            fig.add_trace(go.Scatter(
                x=scenario_ids,
                y=sharpe_values,
                mode='lines+markers',
                name='Sharpe Ratio',
                line=dict(color=self.color_palette[1]),
                marker=dict(size=3)
            ), row=2, col=1)
        
        # Max drawdown
        fig.add_trace(go.Scatter(
            x=scenario_ids,
            y=drawdown_values,
            mode='lines+markers',
            name='Max Drawdown',
            line=dict(color=self.color_palette[2]),
            marker=dict(size=3)
        ), row=3, col=1)
        
        fig.update_layout(
            title="Performance Metrics Timeline",
            template=self.theme,
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Scenario ID", row=3, col=1)
        
        return fig.to_dict()
    
    def create_strategy_comparison(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Create strategy performance comparison chart."""
        if not results:
            return self._empty_chart("No results available")
        
        # Group results by strategy
        strategy_groups = {}
        for result in results:
            if result.success:
                strategy_name = result.strategy_name
                if strategy_name not in strategy_groups:
                    strategy_groups[strategy_name] = []
                strategy_groups[strategy_name].append(result)
        
        if not strategy_groups:
            return self._empty_chart("No successful simulations")
        
        # Calculate statistics for each strategy
        strategy_stats = []
        for strategy_name, strategy_results in strategy_groups.items():
            pnl_values = [r.final_pnl for r in strategy_results]
            sharpe_values = [r.sharpe_ratio for r in strategy_results if r.sharpe_ratio is not None]
            
            stats = {
                'strategy': strategy_name,
                'count': len(strategy_results),
                'mean_pnl': np.mean(pnl_values),
                'std_pnl': np.std(pnl_values),
                'mean_sharpe': np.mean(sharpe_values) if sharpe_values else 0,
                'win_rate': sum(1 for pnl in pnl_values if pnl > 0) / len(pnl_values)
            }
            strategy_stats.append(stats)
        
        if len(strategy_groups) == 1:
            # Single strategy - show distribution
            strategy_name = list(strategy_groups.keys())[0]
            pnl_values = [r.final_pnl for r in strategy_groups[strategy_name]]
            
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=pnl_values,
                name=strategy_name,
                marker_color=self.color_palette[0]
            ))
            
            fig.update_layout(
                title=f"PnL Distribution - {strategy_name}",
                yaxis_title="PnL ($)",
                template=self.theme
            )
            
        else:
            # Multiple strategies - comparison
            strategies = [s['strategy'] for s in strategy_stats]
            mean_pnls = [s['mean_pnl'] for s in strategy_stats]
            win_rates = [s['win_rate'] * 100 for s in strategy_stats]  # Convert to percentage
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Mean PnL by Strategy', 'Win Rate by Strategy'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Mean PnL bar chart
            fig.add_trace(go.Bar(
                x=strategies,
                y=mean_pnls,
                name='Mean PnL',
                marker_color=self.color_palette[0]
            ), row=1, col=1)
            
            # Win rate bar chart
            fig.add_trace(go.Bar(
                x=strategies,
                y=win_rates,
                name='Win Rate (%)',
                marker_color=self.color_palette[1]
            ), row=1, col=2)
            
            fig.update_layout(
                title="Strategy Performance Comparison",
                template=self.theme,
                showlegend=False
            )
            
            fig.update_yaxes(title_text="Mean PnL ($)", row=1, col=1)
            fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
        
        return fig.to_dict()
    
    def create_real_time_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create real-time metrics dashboard."""
        # Create gauge charts for key metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Current PnL', 'Sharpe Ratio', 'Drawdown', 'Success Rate'],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Current PnL gauge
        current_pnl = metrics_data.get('current_pnl', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=current_pnl,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Current PnL ($)"},
            gauge={
                'axis': {'range': [-10000, 10000]},
                'bar': {'color': "green" if current_pnl >= 0 else "red"},
                'steps': [
                    {'range': [-10000, 0], 'color': "lightgray"},
                    {'range': [0, 10000], 'color': "lightgreen"}
                ]
            }
        ), row=1, col=1)
        
        # Add other metrics similarly...
        
        fig.update_layout(
            title="Real-Time Performance Metrics",
            template=self.theme,
            height=500
        )
        
        return fig.to_dict()
    
    def create_execution_timeline(self, execution_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create simulation execution timeline."""
        if not execution_data:
            return self._empty_chart("No execution data available")
        
        # Create Gantt-like chart showing simulation execution
        fig = go.Figure()
        
        for i, execution in enumerate(execution_data):
            start_time = execution.get('start_time')
            end_time = execution.get('end_time')
            scenario_id = execution.get('scenario_id', f'Scenario {i}')
            
            if start_time and end_time:
                fig.add_trace(go.Scatter(
                    x=[start_time, end_time],
                    y=[scenario_id, scenario_id],
                    mode='lines',
                    line=dict(width=10),
                    name=f'Scenario {scenario_id}'
                ))
        
        fig.update_layout(
            title="Simulation Execution Timeline",
            xaxis_title="Time",
            yaxis_title="Scenario",
            template=self.theme
        )
        
        return fig.to_dict()
    
    def _empty_chart(self, message: str) -> Dict[str, Any]:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            template=self.theme,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig.to_dict()

