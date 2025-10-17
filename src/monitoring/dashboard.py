"""
Real-time Monitoring Dashboard for Predictive Maintenance
Built with Plotly Dash for interactive visualization
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for equipment health
    """
    
    def __init__(self, port: int = 8050):
        """
        Initialize dashboard
        
        Args:
            port: Port number for the dashboard server
        """
        self.port = port
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Store for real-time data
        self.sensor_data_history = []
        self.predictions_history = []
        self.alerts_history = []
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🏭 Predictive Maintenance Dashboard", 
                           className="text-center mb-4 mt-4",
                           style={'color': '#2c3e50'}),
                    html.H5("Edge AI-Powered Equipment Monitoring System", 
                           className="text-center mb-4",
                           style={'color': '#7f8c8d'})
                ])
            ]),
            
            html.Hr(),
            
            # Status Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🟢 Normal", className="text-success"),
                            html.H2(id="normal-count", children="0"),
                            html.P("Equipment Operating Normally")
                        ])
                    ], className="mb-3")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🟡 Warning", className="text-warning"),
                            html.H2(id="warning-count", children="0"),
                            html.P("Anomalies Detected")
                        ])
                    ], className="mb-3")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🟠 High Risk", className="text-orange"),
                            html.H2(id="high-risk-count", children="0"),
                            html.P("High Failure Probability")
                        ])
                    ], className="mb-3")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🔴 Critical", className="text-danger"),
                            html.H2(id="critical-count", children="0"),
                            html.P("Immediate Action Required")
                        ])
                    ], className="mb-3")
                ], width=3),
            ]),
            
            html.Hr(),
            
            # Real-time Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Temperature Trends"),
                        dbc.CardBody([
                            dcc.Graph(id="temperature-chart")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Vibration Levels"),
                        dbc.CardBody([
                            dcc.Graph(id="vibration-chart")
                        ])
                    ])
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Failure Probability Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id="failure-probability-chart")
                        ])
                    ])
                ], width=12),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Equipment Health Status"),
                        dbc.CardBody([
                            dcc.Graph(id="equipment-status-chart")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Alerts"),
                        dbc.CardBody([
                            html.Div(id="alerts-table")
                        ])
                    ])
                ], width=6),
            ], className="mb-3"),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            ),
            
            # Store components
            dcc.Store(id='sensor-data-store', data=[]),
            dcc.Store(id='predictions-store', data=[]),
            dcc.Store(id='alerts-store', data=[]),
            
        ], fluid=True, style={'backgroundColor': '#ecf0f1'})
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('normal-count', 'children'),
             Output('warning-count', 'children'),
             Output('high-risk-count', 'children'),
             Output('critical-count', 'children'),
             Output('temperature-chart', 'figure'),
             Output('vibration-chart', 'figure'),
             Output('failure-probability-chart', 'figure'),
             Output('equipment-status-chart', 'figure'),
             Output('alerts-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components"""
            
            # Generate simulated data for demo
            sensor_data, predictions = self._generate_demo_data(n)
            
            # Count statuses
            normal_count = sum(1 for p in predictions if p['alert_level'] == 'LOW')
            warning_count = sum(1 for p in predictions if p['alert_level'] == 'MEDIUM')
            high_risk_count = sum(1 for p in predictions if p['alert_level'] == 'HIGH')
            critical_count = sum(1 for p in predictions if p['alert_level'] == 'CRITICAL')
            
            # Temperature chart
            temp_fig = self._create_time_series_chart(
                sensor_data, 
                'temperature',
                'Temperature (°C)',
                '#e74c3c',
                threshold=95
            )
            
            # Vibration chart
            vib_fig = self._create_time_series_chart(
                sensor_data,
                'vibration',
                'Vibration (mm/s)',
                '#3498db',
                threshold=8
            )
            
            # Failure probability chart
            failure_fig = self._create_failure_probability_chart(predictions)
            
            # Equipment status chart
            status_fig = self._create_equipment_status_chart(predictions)
            
            # Alerts table
            alerts_table = self._create_alerts_table(predictions)
            
            return (
                normal_count, warning_count, high_risk_count, critical_count,
                temp_fig, vib_fig, failure_fig, status_fig, alerts_table
            )
    
    def _generate_demo_data(self, n: int) -> tuple:
        """Generate demo sensor data and predictions"""
        
        num_equipment = 5
        timestamp = datetime.now()
        
        sensor_data = []
        predictions = []
        
        for i in range(num_equipment):
            equipment_id = f"Motor_{i+1:03d}"
            
            # Simulate different conditions
            if i == 0:  # Critical equipment
                temp = 98 + np.random.randn() * 2
                vib = 9 + np.random.randn() * 0.5
                failure_prob = 0.92
                alert_level = 'CRITICAL'
            elif i == 1:  # High risk
                temp = 88 + np.random.randn() * 2
                vib = 7 + np.random.randn() * 0.5
                failure_prob = 0.75
                alert_level = 'HIGH'
            elif i == 2:  # Warning
                temp = 75 + np.random.randn() * 2
                vib = 5 + np.random.randn() * 0.5
                failure_prob = 0.45
                alert_level = 'MEDIUM'
            else:  # Normal
                temp = 62 + np.random.randn() * 2
                vib = 2.5 + np.random.randn() * 0.3
                failure_prob = 0.15
                alert_level = 'LOW'
            
            sensor_data.append({
                'timestamp': timestamp,
                'equipment_id': equipment_id,
                'temperature': temp,
                'vibration': vib,
                'pressure': 7.5 + np.random.randn() * 0.5,
                'rpm': 1800 + np.random.randn() * 100
            })
            
            predictions.append({
                'equipment_id': equipment_id,
                'failure_probability': failure_prob,
                'alert_level': alert_level,
                'timestamp': timestamp
            })
        
        return sensor_data, predictions
    
    def _create_time_series_chart(
        self, 
        data: List[Dict], 
        column: str, 
        title: str, 
        color: str,
        threshold: float = None
    ):
        """Create time series chart"""
        
        df = pd.DataFrame(data)
        
        fig = go.Figure()
        
        for equipment_id in df['equipment_id'].unique():
            equipment_data = df[df['equipment_id'] == equipment_id]
            
            fig.add_trace(go.Scatter(
                x=equipment_data['timestamp'],
                y=equipment_data[column],
                mode='lines+markers',
                name=equipment_id,
                line=dict(width=2)
            ))
        
        if threshold:
            fig.add_hline(
                y=threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Threshold: {threshold}"
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=title,
            hovermode='x unified',
            height=300
        )
        
        return fig
    
    def _create_failure_probability_chart(self, predictions: List[Dict]):
        """Create failure probability chart"""
        
        df = pd.DataFrame(predictions)
        
        fig = px.line(
            df,
            x='timestamp',
            y='failure_probability',
            color='equipment_id',
            title='Failure Probability Trends',
            markers=True
        )
        
        fig.add_hline(
            y=0.7, 
            line_dash="dash", 
            line_color="orange",
            annotation_text="High Risk Threshold"
        )
        
        fig.add_hline(
            y=0.9,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Threshold"
        )
        
        fig.update_layout(
            yaxis_title="Failure Probability",
            xaxis_title="Time",
            hovermode='x unified',
            height=350
        )
        
        return fig
    
    def _create_equipment_status_chart(self, predictions: List[Dict]):
        """Create equipment status pie chart"""
        
        df = pd.DataFrame(predictions)
        status_counts = df['alert_level'].value_counts()
        
        colors = {
            'LOW': '#27ae60',
            'MEDIUM': '#f39c12',
            'HIGH': '#e67e22',
            'CRITICAL': '#e74c3c'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            marker=dict(colors=[colors.get(level, '#95a5a6') for level in status_counts.index]),
            hole=.3
        )])
        
        fig.update_layout(
            title="Equipment Status Distribution",
            height=350
        )
        
        return fig
    
    def _create_alerts_table(self, predictions: List[Dict]):
        """Create alerts table"""
        
        # Filter for alerts only
        alerts = [p for p in predictions if p['alert_level'] in ['HIGH', 'CRITICAL']]
        
        if not alerts:
            return html.P("No active alerts", className="text-success")
        
        table_rows = []
        for alert in alerts:
            color = 'danger' if alert['alert_level'] == 'CRITICAL' else 'warning'
            
            row = html.Tr([
                html.Td(alert['equipment_id']),
                html.Td(f"{alert['failure_probability']:.1%}"),
                html.Td(
                    dbc.Badge(alert['alert_level'], color=color),
                ),
                html.Td(alert['timestamp'].strftime('%H:%M:%S'))
            ])
            table_rows.append(row)
        
        table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Equipment"),
                html.Th("Failure Prob."),
                html.Th("Status"),
                html.Th("Time")
            ])),
            html.Tbody(table_rows)
        ], bordered=True, hover=True, size='sm')
        
        return table
    
    def run(self, debug: bool = False):
        """Run the dashboard server"""
        print(f"\n{'='*60}")
        print(f"🚀 Starting Monitoring Dashboard...")
        print(f"{'='*60}")
        print(f"📊 Dashboard URL: http://localhost:{self.port}")
        print(f"⏰ Auto-refresh: Every 5 seconds")
        print(f"{'='*60}\n")
        
        self.app.run_server(debug=debug, port=self.port, host='0.0.0.0')


def main():
    """Launch monitoring dashboard"""
    dashboard = MonitoringDashboard(port=8050)
    dashboard.run(debug=True)


if __name__ == '__main__':
    main()
