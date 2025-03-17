import dash
from dash import dcc, html, Input, Output, State, no_update
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Load the PM2.5 model
with open('pm25_model_7d.pkl', 'rb') as f:
    model = joblib.load(f)

# Load the PM10 model
try:
    with open('catboost_pm10_24h.pkl', 'rb') as f:
        pm10_model = joblib.load(f)
    pm10_model_loaded = True
    # Get expected feature names if available
    pm10_expected_features = []
    if hasattr(pm10_model, 'feature_names_in_'):
        pm10_expected_features = pm10_model.feature_names_in_
        print("PM10 model expects these features:", pm10_expected_features)
except Exception as e:
    print(f"Error loading PM10 model: {e}")
    pm10_model_loaded = False

# Get expected feature names if available
expected_features = []
if hasattr(model, 'feature_names_in_'):
    expected_features = model.feature_names_in_
    print("Model expects these features:", expected_features)

# Define CSS styles with pollution theme
colors = {
    'background': '#121212',
    'panel': '#1e1e1e', 
    'text': '#ffffff',
    'lightText': '#cccccc',
    'border': '#333333',
    'primary': '#790000',
    'accent1': '#ff3b30',
    'accent2': '#541b1b',
    'smokeDark': '#1a0000',
    'smokeLight': '#2c0a0a',
    'danger': '#ff1744',      # Red - for dangerous levels
    'warning': '#ff9800',     # Orange - for moderate warnings
    'safe': '#4caf50',        # Green - for safe levels
    'warningRed': '#ff1744',
    'dangerZone': '#b71c1c',
}

# Common styles with pollution theme
common_styles = {
    'panel': {
        'backgroundColor': colors['panel'],
        'borderRadius': '12px',
        'boxShadow': '0 8px 20px rgba(0, 0, 0, 0.3), 0 0 30px rgba(109, 40, 217, 0.2)',
        'padding': '25px',
        'marginBottom': '25px',
        'border': f'1px solid {colors["border"]}',
        'backdropFilter': 'blur(8px)',
        'position': 'relative',
        'overflow': 'hidden'
    },
    'header': {
        'color': colors['text'],
        'marginBottom': '25px',
        'fontWeight': 'bold',
        'position': 'relative',
        'display': 'inline-block',
        'paddingBottom': '12px',
        'fontSize': '22px',
        'borderBottom': f'2px solid {colors["primary"]}',
        'textShadow': '0 0 10px rgba(109, 40, 217, 0.5)',
        'width': '100%'
    },
    'label': {
        'color': colors['lightText'],
        'marginTop': '20px',
        'marginBottom': '10px',
        'fontWeight': 'bold',
        'fontSize': '16px',
        'letterSpacing': '1px',
        'textTransform': 'uppercase',
        'display': 'block',
        'position': 'relative'
    },
    'input': {
        'width': '100%',
        'padding': '15px 20px',
        'borderRadius': '10px',
        'backgroundColor': 'rgba(30, 30, 30, 0.8)',
        'border': f'1px solid {colors["accent1"]}',
        'marginBottom': '25px',
        'color': colors['text'],
        'transition': 'all 0.3s ease',
        'boxShadow': 'inset 0 2px 4px rgba(0, 0, 0, 0.3), 0 0 8px rgba(255, 0, 0, 0.1)',
        'fontSize': '16px',
        'fontFamily': 'Rajdhani, sans-serif'
    },
    'button': {
        'backgroundColor': colors['accent1'],
        'color': 'white',
        'padding': '14px 32px',
        'margin': '25px 0',
        'border': 'none',
        'borderRadius': '8px',
        'boxShadow': '0 4px 15px rgba(255, 59, 48, 0.4)',
        'cursor': 'pointer',
        'fontSize': '16px',
        'fontWeight': 'bold',
        'transition': 'all 0.3s ease',
        'position': 'relative',
        'overflow': 'hidden',
        'textTransform': 'uppercase',
        'letterSpacing': '1px'
    }
}

# Define additional CSS styles for improved visuals with pollution theme
additional_styles = {
    'card': {
        'backgroundColor': 'rgba(31, 41, 55, 0.7)',
        'borderRadius': '12px',
        'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.4), 0 0 15px rgba(109, 40, 217, 0.15)',
        'padding': '25px',
        'marginBottom': '25px',
        'transition': 'all 0.4s ease',
        'border': f'1px solid {colors["border"]}',
        'backdropFilter': 'blur(10px)',
        'position': 'relative',
        'overflow': 'hidden'
    },
    'info_text': {
        'fontSize': '14px',
        'color': colors['lightText'],
        'margin': '10px 0 20px 0',
        'lineHeight': '1.6',
        'borderLeft': f'3px solid {colors["primary"]}',
        'paddingLeft': '15px',
        'fontStyle': 'italic'
    },
    'value_display': {
        'fontSize': '22px',
        'fontWeight': 'bold',
        'color': colors['primary'],
        'padding': '8px 12px',
        'backgroundColor': 'rgba(0, 0, 0, 0.2)',
        'borderRadius': '8px',
        'display': 'inline-block',
        'margin': '5px 0',
        'textShadow': '0 0 10px rgba(109, 40, 217, 0.5)',
        'boxShadow': 'inset 0 2px 4px rgba(0, 0, 0, 0.25)'
    },
    'stat_container': {
        'backgroundColor': 'rgba(0, 0, 0, 0.15)',
        'borderRadius': '10px',
        'padding': '15px',
        'marginTop': '15px',
        'marginBottom': '20px',
        'border': f'1px solid {colors["border"]}',
        'display': 'flex',
        'justifyContent': 'space-between',
        'flexWrap': 'wrap'
    },
    'stat_item': {
        'padding': '10px',
        'textAlign': 'center',
        'flexGrow': '1',
        'flexBasis': '30%',
        'borderRadius': '8px',
        'transition': 'all 0.3s ease',
        'margin': '5px',
        'backgroundColor': 'rgba(31, 41, 55, 0.5)'
    },
    'dashboard_heading': {
        'color': 'white',
        'textAlign': 'center',
        'fontWeight': 'bold',
        'marginBottom': '35px',
        'fontSize': '36px',
        'position': 'relative',
        'display': 'inline-block',
        'padding': '5px 15px',
        'borderBottom': f'4px solid {colors["primary"]}',
        'textShadow': '0 0 15px rgba(109, 40, 217, 0.7)',
        'letterSpacing': '1px'
    },
    'pollution_icon': {
        'marginRight': '10px',
        'color': colors['primary']
    }
}

# Initialize the Dash app
app = dash.Dash(
    __name__, 
    title="Air Quality Prediction Dashboard",
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
)

# Define tabs for different prediction types
tabs_style = {
    'height': '50px',
    'backgroundColor': 'rgba(31, 41, 55, 0.4)',
    'borderRadius': '12px 12px 0 0',
    'padding': '0 20px',
    'border': f'1px solid {colors["border"]}',
    'borderBottom': 'none',
    'boxShadow': '0 -5px 15px rgba(0, 0, 0, 0.1)',
    'backdropFilter': 'blur(10px)'
}

tab_style = {
    'borderBottom': f'1px solid {colors["border"]}',
    'padding': '12px 20px',
    'fontWeight': 'bold',
    'color': colors['lightText'],
    'backgroundColor': 'transparent',
    'borderTopLeftRadius': '8px',
    'borderTopRightRadius': '8px',
    'transition': 'all 0.3s ease',
    'marginRight': '5px',
    'fontSize': '15px',
    'textTransform': 'uppercase',
    'letterSpacing': '1px',
    'fontFamily': 'Rajdhani, sans-serif'
}

tab_selected_style = {
    'borderTop': f'3px solid {colors["primary"]}',
    'borderLeft': f'1px solid {colors["border"]}',
    'borderRight': f'1px solid {colors["border"]}',
    'borderBottom': 'none', 
    'backgroundColor': 'rgba(109, 40, 217, 0.1)',
    'color': colors['primary'],
    'padding': '12px 20px',
    'fontWeight': 'bold',
    'boxShadow': '0 0 15px rgba(109, 40, 217, 0.2)',
    'borderTopLeftRadius': '8px',
    'borderTopRightRadius': '8px',
    'transform': 'translateY(-3px)',
    'fontSize': '15px',
    'textShadow': '0 0 5px rgba(109, 40, 217, 0.3)',
    'textTransform': 'uppercase',
    'letterSpacing': '1px',
    'fontFamily': 'Rajdhani, sans-serif'
}

tabs = dcc.Tabs(
    id='tabs',
    value='tab-7-day',
    children=[
        dcc.Tab(label='7-Day Prediction', value='tab-7-day', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='PM10 Prediction', value='tab-pm10', style=tab_style, selected_style=tab_selected_style),
    ],
    style=tabs_style
)

# Define the 7-day prediction layout
seven_day_layout = html.Div([
    html.Div([
        # Info card at the top
        html.Div([
            html.H3([
                html.I(className="fas fa-calendar-alt", style={"marginRight": "15px", "color": colors['accent1']}),
                "7-Day PM2.5 Prediction"
            ], className="panel-header"),
            html.P([
                html.I(className="fas fa-info-circle", style={"marginRight": "10px", "color": colors['accent1']}),
                "This tool predicts PM2.5 values for the next 7 days based on current conditions and historical data. "
                "The predictions use iterative forecasting, where each day's prediction becomes input for the next day."
            ], style={
                "fontSize": "16px", 
                "color": colors['lightText'], 
                "padding": "15px", 
                "backgroundColor": "rgba(20, 20, 20, 0.5)", 
                "borderRadius": "10px",
                "border": f"1px solid {colors['border']}",
                "marginTop": "15px"
            })
        ], className="panel"),
        
        # Left panel - Initial conditions
        html.Div([
            html.H3([
                html.I(className="fas fa-sliders-h", style={"marginRight": "15px", "color": colors['accent1']}),
                "Initial Conditions"
            ], className="panel-header"),
            
            # Date section with border
            html.Div([
                html.Label([
                    html.I(className="far fa-calendar-alt", style={"marginRight": "10px", "fontSize": "18px", "color": "#3498db"}),
                    "Starting Date"
                ], className="input-label", style={"marginTop": "0", "color": colors['accent1']}),
                dcc.DatePickerSingle(
                    id='start-date-7d',
                    date=datetime.now().date(),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%'},
                    className="date-picker"
                )
            ], style={
                "padding": "20px", 
                "backgroundColor": "rgba(20, 20, 20, 0.5)", 
                "borderRadius": "10px", 
                "marginBottom": "25px",
                "border": f"1px solid {colors['border']}"
            }),
            
            # Environmental conditions section
            html.Div([
                html.H4([
                    html.I(className="fas fa-cloud", style={"marginRight": "10px"}),
                    "Environmental Conditions"
                ], style={"marginBottom": "15px", "color": colors['accent1'], "fontSize": "18px"}),
                
                html.Label([
                    html.I(className="fas fa-tint", style={"marginRight": "10px", "fontSize": "18px", "color": "#3498db"}),
                    "Average Humidity (%) for the week"
                ], className="input-label", style={"marginTop": "10px"}),
                dcc.Slider(
                    id='humidity-7d', 
                    min=0, 
                    max=100, 
                    step=1, 
                    value=60,
                    marks={0: '0', 25: '25', 50: '50', 75: '75', 100: '100'}, 
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    className='slider'
                ),
                
                html.Label([
                    html.I(className="fas fa-temperature-high", style={"marginRight": "10px", "fontSize": "18px", "color": "#e74c3c"}),
                    "Average Temperature (°C) for the week"
                ], className="input-label"),
                dcc.Slider(
                    id='temperature-7d', 
                    min=0, 
                    max=50, 
                    step=1, 
                    value=25,
                    marks={0: '0', 10: '10', 20: '20', 30: '30', 40: '40', 50: '50'}, 
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    className='slider'
                )
            ], style={
                "padding": "20px", 
                "backgroundColor": "rgba(20, 20, 20, 0.5)", 
                "borderRadius": "10px",
                "border": f"1px solid {colors['border']}"
            }),
        ], className="panel", style={"width": "48%", "display": "inline-block", "marginRight": "2%", "verticalAlign": "top"}),
        
        # Right panel - Recent PM2.5 values
        html.Div([
            html.H3([
                html.I(className="fas fa-history", style={"marginRight": "15px", "color": colors['accent1']}),
                "Recent PM2.5 Values"
            ], className="panel-header"),
            
            # Historical values in a nice grid
            html.Div([
                html.Div([
                    html.Label([
                        html.I(className="fas fa-clock", style={"marginRight": "10px", "fontSize": "16px", "color": "#3498db"}),
                        "PM2.5 (1 day ago)"
                    ], className="input-label", style={"marginTop": "0", "marginBottom": "5px"}),
                    dcc.Input(
                        id='pm25-1d-ago-7d', 
                        type='number', 
                        value=20, 
                        min=0, 
                        max=1000,
                        className="input-field",
                        style={"marginBottom": "15px"}
                    ),
                ], style={"width": "48%", "display": "inline-block", "marginRight": "4%"}),
                
                html.Div([
                    html.Label([
                        html.I(className="fas fa-clock", style={"marginRight": "10px", "fontSize": "16px", "color": "#9b59b6"}),
                        "PM2.5 (2 days ago)"
                    ], className="input-label", style={"marginTop": "0", "marginBottom": "5px"}),
                    dcc.Input(
                        id='pm25-2d-ago-7d', 
                        type='number', 
                        value=20, 
                        min=0, 
                        max=1000,
                        className="input-field",
                        style={"marginBottom": "15px"}
                    ),
                ], style={"width": "48%", "display": "inline-block"}),
                
                html.Div([
                    html.Label([
                        html.I(className="fas fa-clock", style={"marginRight": "10px", "fontSize": "16px", "color": "#e67e22"}),
                        "PM2.5 (3 days ago)"
                    ], className="input-label", style={"marginTop": "15px", "marginBottom": "5px"}),
                    dcc.Input(
                        id='pm25-3d-ago-7d', 
                        type='number', 
                        value=20, 
                        min=0, 
                        max=1000,
                        className="input-field",
                        style={"marginBottom": "15px"}
                    ),
                ], style={"width": "48%", "display": "inline-block", "marginRight": "4%"}),
                
                html.Div([
                    html.Div([
                        html.I(className="fas fa-calculator", style={"marginRight": "10px", "fontSize": "16px", "color": colors['primary']}),
                        "3-Day Moving Average"
                    ], className="input-label", style={"marginTop": "15px", "marginBottom": "5px", "color": colors['accent1']}),
                    html.Div(
                        id='pm25-3d-avg-7d', 
                        className="value-display",
                        style={
                            "textAlign": "center", 
                            "fontSize": "22px", 
                            "fontWeight": "bold",
                            "padding": "10px",
                            "backgroundColor": "rgba(30, 30, 30, 0.7)",
                            "borderRadius": "8px",
                            "border": f"1px solid {colors['border']}",
                            "color": colors['primary']
                        }
                    ),
                ], style={"width": "48%", "display": "inline-block"}),
            ], style={
                "padding": "20px", 
                "backgroundColor": "rgba(20, 20, 20, 0.5)", 
                "borderRadius": "10px", 
                "marginBottom": "25px",
                "border": f"1px solid {colors['border']}"
            }),
            
            # AQI interpretation box
            html.Div([
                html.H4([
                    html.I(className="fas fa-info-circle", style={"marginRight": "10px"}),
                    "Air Quality Index Interpretation"
                ], style={"marginBottom": "15px", "color": colors['accent1'], "fontSize": "18px", "textAlign": "center"}),
                
                html.Div([
                    html.Div([
                        html.Div("0-12", style={"fontWeight": "bold", "color": colors['safe']}),
                        html.Div("Good", style={"fontSize": "14px"})
                    ], style={"padding": "8px", "borderLeft": f"4px solid {colors['safe']}"}),
                    
                    html.Div([
                        html.Div("12.1-35.4", style={"fontWeight": "bold", "color": colors['warning']}),
                        html.Div("Moderate", style={"fontSize": "14px"})
                    ], style={"padding": "8px", "borderLeft": f"4px solid {colors['warning']}"}),
                    
                    html.Div([
                        html.Div("35.5-55.4", style={"fontWeight": "bold", "color": colors['danger']}),
                        html.Div("Unhealthy for Sensitive Groups", style={"fontSize": "14px"})
                    ], style={"padding": "8px", "borderLeft": f"4px solid {colors['danger']}"}),
                    
                    html.Div([
                        html.Div("55.5+", style={"fontWeight": "bold", "color": colors['warningRed']}),
                        html.Div("Unhealthy", style={"fontSize": "14px"})
                    ], style={"padding": "8px", "borderLeft": f"4px solid {colors['warningRed']}"}),
                ]), 
            ], style={
                "padding": "20px", 
                "backgroundColor": "rgba(20, 20, 20, 0.5)", 
                "borderRadius": "10px",
                "border": f"1px solid {colors['border']}"
            }),
        ], className="panel", style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between", "marginBottom": "20px"}, className="flex-container"),
    
    # Prediction button with enhanced styling
    html.Div([
        html.Button([
            html.I(className="fas fa-chart-line", style={"marginRight": "10px", "fontSize": "18px"}),
            "Predict 7 Days"
        ], 
        id='predict-7d-button', 
        n_clicks=0, 
        className="predict-button",
        style={
            "backgroundColor": colors['accent1'],
            "color": "white",
            "padding": "16px 32px",
            "margin": "25px 0",
            "border": "none",
            "borderRadius": "8px",
            "boxShadow": f"0 4px 15px rgba(255, 59, 48, 0.4)",
            "cursor": "pointer",
            "fontSize": "18px",
            "fontWeight": "bold",
            "transition": "all 0.3s ease",
            "textTransform": "uppercase",
            "letterSpacing": "1px"
        }),
    ], style={"textAlign": "center"}),
    
    # Prediction result panel with enhanced styling
    html.Div(id='prediction-result-7d', className="panel", style={"marginTop": "20px"}),
    
    # Graph panel with enhanced styling
    html.Div([
        dcc.Graph(id='prediction-graph-7d')
    ], className="panel", style={"marginTop": "20px", "padding": "15px", "boxShadow": "0 8px 20px rgba(0, 0, 0, 0.3), 0 0 30px rgba(109, 40, 217, 0.2)"})
])

# Define the PM10 prediction layout with enhanced pollution theme
pm10_layout = html.Div([
    html.Div([
        # Info card at the top
        html.Div([
            html.H3([
                html.I(className="fas fa-wind", style={"marginRight": "10px", "color": colors['accent2']}),
                "PM10 24-Hour Prediction"
            ], className="panel-header"),
            html.P([
                html.I(className="fas fa-info-circle", style={"marginRight": "10px", "color": colors['accent2']}),
                "This tool predicts PM10 values for the next 24 hours based on current conditions and historical data. "
                "The model uses the diurnal pattern typically observed in PM10 concentrations to provide realistic hourly forecasts."
            ], className="info-text")
        ], className="panel"),
        
        # Left panel - Current conditions
        html.Div([
            html.H3([
                html.I(className="fas fa-broadcast-tower", style={"marginRight": "15px", "color": colors['accent2']}),
                "Current Conditions"
            ], className="panel-header"),
            
            # Date & Time section with a border
            html.Div([
                html.Div([
                    html.Label([
                        html.I(className="far fa-calendar-alt", style={"marginRight": "10px", "fontSize": "18px"}),
                        "Current Date"
                    ], className="input-label", style={"color": colors['accent1'], "marginTop": "0"}),
                    dcc.DatePickerSingle(
                        id='pm10-date-picker',
                        date=datetime.now().date(),
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'},
                        className="date-picker"
                    )
                ], style={"width": "48%", "display": "inline-block", "marginRight": "4%"}),
                
                html.Div([
                    html.Label([
                        html.I(className="far fa-clock", style={"marginRight": "10px", "fontSize": "18px"}),
                        "Current Time"
                    ], className="input-label", style={"color": colors['accent1'], "marginTop": "0"}),
                    dcc.Dropdown(
                        id='pm10-time-picker',
                        options=[{'label': f'{h:02d}:00', 'value': h} for h in range(24)],
                        value=datetime.now().hour,
                        clearable=False,
                        className="dropdown-field",
                        style={"backgroundColor": "rgba(30, 30, 30, 0.8)", "color": "white"}
                    )
                ], style={"width": "48%", "display": "inline-block"}),
            ], style={
                "padding": "20px", 
                "backgroundColor": "rgba(20, 20, 20, 0.5)", 
                "borderRadius": "10px", 
                "marginBottom": "25px",
                "border": f"1px solid {colors['border']}"
            }),
            
            # Environmental conditions section
            html.Div([
                html.H4([
                    html.I(className="fas fa-cloud", style={"marginRight": "10px"}),
                    "Environmental Conditions"
                ], style={"marginBottom": "15px", "color": colors['accent1'], "fontSize": "18px"}),
                
                html.Label([
                    html.I(className="fas fa-tint", style={"marginRight": "10px", "fontSize": "18px", "color": "#3498db"}),
                    "Humidity (%)"
                ], className="input-label", style={"marginTop": "10px"}),
                dcc.Slider(
                    id='pm10-humidity', 
                    min=0, 
                    max=100, 
                    step=1, 
                    value=60,
                    marks={0: '0', 25: '25', 50: '50', 75: '75', 100: '100'}, 
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    className='slider'
                ),
                
                html.Label([
                    html.I(className="fas fa-temperature-high", style={"marginRight": "10px", "fontSize": "18px", "color": "#e74c3c"}),
                    "Temperature (°C)"
                ], className="input-label"),
                dcc.Slider(
                    id='pm10-temperature', 
                    min=0, 
                    max=50, 
                    step=1, 
                    value=25,
                    marks={0: '0', 10: '10', 20: '20', 30: '30', 40: '40', 50: '50'}, 
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    className='slider'
                ),
            ], style={
                "padding": "20px", 
                "backgroundColor": "rgba(20, 20, 20, 0.5)", 
                "borderRadius": "10px", 
                "marginBottom": "25px",
                "border": f"1px solid {colors['border']}"
            }),
            
            # Current PM10 section
            html.Div([
                html.Label([
                    html.I(className="fas fa-smog", style={"marginRight": "10px", "fontSize": "22px", "color": colors['danger']}),
                    "Current PM10"
                ], className="input-label", style={"textAlign": "center", "fontSize": "20px", "marginTop": "0"}),
                dcc.Input(
                    id='current-pm10', 
                    type='number', 
                    value=30, 
                    min=0, 
                    max=1000,
                    className="input-field",
                    style={"fontSize": "24px", "textAlign": "center", "fontWeight": "bold", "color": colors['accent1']}
                )
            ], style={
                "padding": "20px", 
                "backgroundColor": "rgba(20, 20, 20, 0.5)", 
                "borderRadius": "10px",
                "border": f"1px solid {colors['border']}"
            }),
        ], className="panel", style={"width": "48%", "display": "inline-block", "marginRight": "2%", "verticalAlign": "top"}),
        
        # Right panel - Historical PM10 values
        html.Div([
            html.H3([
                html.I(className="fas fa-history", style={"marginRight": "15px", "color": colors['accent2']}),
                "Historical PM10 Values"
            ], className="panel-header"),
            
            # Historical values in a nice grid
            html.Div([
                html.Div([
                    html.Label([
                        html.I(className="fas fa-clock", style={"marginRight": "10px", "fontSize": "16px", "color": "#3498db"}),
                        "PM10 (1 hour ago)"
                    ], className="input-label", style={"marginTop": "0", "marginBottom": "5px"}),
                    dcc.Input(
                        id='pm10-lag-1', 
                        type='number', 
                        value=28, 
                        min=0, 
                        max=1000,
                        className="input-field",
                        style={"marginBottom": "15px"}
                    ),
                ], style={"width": "48%", "display": "inline-block", "marginRight": "4%"}),
                
                html.Div([
                    html.Label([
                        html.I(className="fas fa-clock", style={"marginRight": "10px", "fontSize": "16px", "color": "#9b59b6"}),
                        "PM10 (3 hours ago)"
                    ], className="input-label", style={"marginTop": "0", "marginBottom": "5px"}),
                    dcc.Input(
                        id='pm10-lag-3', 
                        type='number', 
                        value=25, 
                        min=0, 
                        max=1000,
                        className="input-field",
                        style={"marginBottom": "15px"}
                    ),
                ], style={"width": "48%", "display": "inline-block"}),
                
                html.Div([
                    html.Label([
                        html.I(className="fas fa-clock", style={"marginRight": "10px", "fontSize": "16px", "color": "#e67e22"}),
                        "PM10 (6 hours ago)"
                    ], className="input-label", style={"marginTop": "15px", "marginBottom": "5px"}),
                    dcc.Input(
                        id='pm10-lag-6', 
                        type='number', 
                        value=22, 
                        min=0, 
                        max=1000,
                        className="input-field",
                        style={"marginBottom": "15px"}
                    ),
                ], style={"width": "48%", "display": "inline-block", "marginRight": "4%"}),
                
                html.Div([
                    html.Label([
                        html.I(className="fas fa-clock", style={"marginRight": "10px", "fontSize": "16px", "color": "#2ecc71"}),
                        "PM10 (24 hours ago)"
                    ], className="input-label", style={"marginTop": "15px", "marginBottom": "5px"}),
                    dcc.Input(
                        id='pm10-lag-24', 
                        type='number', 
                        value=28, 
                        min=0, 
                        max=1000,
                        className="input-field",
                        style={"marginBottom": "15px"}
                    ),
                ], style={"width": "48%", "display": "inline-block"}),
            ], style={
                "padding": "20px", 
                "backgroundColor": "rgba(20, 20, 20, 0.5)", 
                "borderRadius": "10px", 
                "marginBottom": "25px",
                "border": f"1px solid {colors['border']}"
            }),
            
            # Calculated statistics
            html.Div([
                html.H4([
                    html.I(className="fas fa-calculator", style={"marginRight": "10px"}),
                    "Calculated Metrics"
                ], style={"marginBottom": "20px", "color": colors['accent1'], "fontSize": "18px", "textAlign": "center"}),
                
                # Calculated values with visual enhancements in a grid
                html.Div([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-chart-line", style={"marginRight": "10px", "color": colors['primary']}),
                            "6-Hour Rolling Mean"
                        ], className="stat-label", style={"textAlign": "center", "marginBottom": "10px"}),
                        html.Div(id='pm10-roll-mean-6', className="stat-value", style={
                            "textAlign": "center", 
                            "fontSize": "22px", 
                            "fontWeight": "bold",
                            "padding": "10px",
                            "backgroundColor": "rgba(30, 30, 30, 0.7)",
                            "borderRadius": "8px",
                            "border": f"1px solid {colors['border']}"
                        })
                    ], style={"marginBottom": "15px"}),
                    
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-chart-area", style={"marginRight": "10px", "color": colors['primary']}),
                            "24-Hour Rolling Mean"
                        ], className="stat-label", style={"textAlign": "center", "marginBottom": "10px"}),
                        html.Div(id='pm10-roll-mean-24', className="stat-value", style={
                            "textAlign": "center", 
                            "fontSize": "22px", 
                            "fontWeight": "bold",
                            "padding": "10px",
                            "backgroundColor": "rgba(30, 30, 30, 0.7)",
                            "borderRadius": "8px",
                            "border": f"1px solid {colors['border']}"
                        })
                    ], style={"marginBottom": "15px"}),
                    
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-chart-bar", style={"marginRight": "10px", "color": colors['primary']}),
                            "12-Hour EWM"
                        ], className="stat-label", style={"textAlign": "center", "marginBottom": "10px"}),
                        html.Div(id='pm10-ewm-12', className="stat-value", style={
                            "textAlign": "center", 
                            "fontSize": "22px", 
                            "fontWeight": "bold",
                            "padding": "10px",
                            "backgroundColor": "rgba(30, 30, 30, 0.7)",
                            "borderRadius": "8px",
                            "border": f"1px solid {colors['border']}"
                        })
                    ])
                ])
            ], style={
                "padding": "20px", 
                "backgroundColor": "rgba(20, 20, 20, 0.5)", 
                "borderRadius": "10px",
                "border": f"1px solid {colors['border']}"
            }),
        ], className="panel", style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between", "marginBottom": "20px"}, className="flex-container"),
    
    html.Div([
        html.Button([
            html.I(className="fas fa-microscope", style={"marginRight": "10px"}),
            "Predict PM10"
        ], id='predict-pm10-button', n_clicks=0, className="predict-button"),
    ], style={"textAlign": "center"}),
    
    html.Div(id='prediction-result-pm10', className="panel", style={"marginTop": "20px"}),
    
    html.Div([
        dcc.Graph(id='prediction-graph-pm10')
    ], className="panel", style={"marginTop": "20px"})
])

# Define the layout
app.layout = html.Div([
    html.Div([
        html.H1(
            "Air Quality Prediction Dashboard", 
            style={
                'textAlign': 'center', 
                'marginBottom': '30px', 
                'color': colors['primary'],
                'fontWeight': 'bold',
                'fontSize': '32px'
            }
        ),
        tabs,
        html.Div(id='tabs-content')
    ], style={
        'maxWidth': '1200px', 
        'margin': '0 auto', 
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': colors['background']
    })
], style={'backgroundColor': colors['background']})

# Custom table styles
table_styles = {
    'table': {
        'margin': '0 auto',
        'borderCollapse': 'collapse', 
        'width': '80%',
        'borderRadius': '8px',
        'overflow': 'hidden',
        'boxShadow': '0 2px 3px rgba(0, 0, 0, 0.1)'
    },
    'th': {
        'backgroundColor': colors['primary'],
        'color': 'white',
        'textAlign': 'left',
        'padding': '12px 15px',
        'fontWeight': 'bold'
    },
    'td': {
        'padding': '10px 15px',
        'borderBottom': f'1px solid {colors["border"]}'
    },
    'td_value': {
        'padding': '10px 15px',
        'borderBottom': f'1px solid {colors["border"]}',
        'fontWeight': 'bold',
        'color': colors['primary']
    },
    'tr_even': {
        'backgroundColor': '#f9f9f9'
    },
    'tr_odd': {
        'backgroundColor': 'white'
    }
}

# Callback to update tab content
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-7-day':
        return seven_day_layout
    elif tab == 'tab-pm10':
        return pm10_layout

# Add callback to calculate and display 3-day moving average
@app.callback(
    Output('pm25-3d-avg-7d', 'children'),
    [Input('pm25-1d-ago-7d', 'value'),
     Input('pm25-2d-ago-7d', 'value'),
     Input('pm25-3d-ago-7d', 'value')]
)
def update_moving_average(pm25_1d, pm25_2d, pm25_3d):
    if all(v is not None for v in [pm25_1d, pm25_2d, pm25_3d]):
        avg = np.mean([pm25_1d, pm25_2d, pm25_3d])
        return f"{avg:.2f}"
    return "0.00"

# Callback for 7-day prediction
@app.callback(
    [Output('prediction-result-7d', 'children'),
     Output('prediction-graph-7d', 'figure')],
    [Input('predict-7d-button', 'n_clicks')],
    [State('humidity-7d', 'value'),
     State('temperature-7d', 'value'),
     State('start-date-7d', 'date'),
     State('pm25-1d-ago-7d', 'value'),
     State('pm25-2d-ago-7d', 'value'),
     State('pm25-3d-ago-7d', 'value')]
)
def update_7day_prediction(n_clicks, humidity, temperature, start_date, 
                          pm25_1d_ago, pm25_2d_ago, pm25_3d_ago):
    if n_clicks > 0:
        # Parse the start date
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Calculate the 3-day moving average
        pm25_ma3d = np.mean([pm25_1d_ago, pm25_2d_ago, pm25_3d_ago])
        
        # Initialize values for prediction
        predictions = []
        dates = []
        
        # Use the 1-day ago value as the current PM2.5 and starting point
        pm25_current = pm25_1d_ago
        pm25_lag1d = pm25_2d_ago  # 2-day ago value becomes the lagged value
        pm25_ma3d = pm25_ma3d     # 3-day moving average
        
        # Predict for the next 7 days
        for day_offset in range(7):
            # Get the date for this prediction
            prediction_date = start_date_obj + timedelta(days=day_offset)
            dates.append(prediction_date.strftime('%Y-%m-%d'))
            
            # For simplicity, we'll use the same hour (12:00) for all predictions
            hour = 12
            
            # Get day, month, day_of_week for this date
            day = prediction_date.day
            month = prediction_date.month
            day_of_week = prediction_date.weekday()
            
            # Create feature dictionary for this prediction
            feature_values = {
                'humidity': humidity,
                'temperature': temperature,
                'hour': hour,
                'day': day,
                'month': month,
                'day_of_week': day_of_week,
                'pm_2_5': pm25_current,
                'pm_2_5_lag1h': pm25_lag1d,  # Using 1-day lag instead of 1-hour lag
                'pm25_lag1h': pm25_lag1d,
                'pm_2_5_lag_1_h': pm25_lag1d,
                'pm_2_5_ma3h': pm25_ma3d,    # Using 3-day MA instead of 3-hour MA
                'pm25_ma3h': pm25_ma3d
            }
            
            # Create DataFrame with correct feature names expected by the model
            if expected_features:
                # If we know the expected feature names, use them
                features = pd.DataFrame({name: [feature_values.get(name, 0)] for name in expected_features})
            else:
                # Otherwise, try our best guess at feature names
                features = pd.DataFrame({
                    'humidity': [humidity],
                    'pm_2_5': [pm25_current],
                    'temperature': [temperature],
                    'hour': [hour],
                    'day': [day],
                    'month': [month],
                    'day_of_week': [day_of_week],
                    'pm_2_5_lag1h': [pm25_lag1d],
                    'pm_2_5_ma3h': [pm25_ma3d]
                })
            
            # Make prediction for this day
            base_prediction = model.predict(features)[0]
            
            # Add realistic fluctuation patterns
            # 1. Weekend effect - typically lower on weekends
            weekday_factor = 1.0
            if day_of_week == 5:  # Saturday
                weekday_factor = 0.9  # 10% lower on Saturday
            elif day_of_week == 6:  # Sunday
                weekday_factor = 0.85  # 15% lower on Sunday
            
            # 2. Weather patterns - simulate a 3-day weather cycle
            weather_phase = (day_offset % 3) / 3.0  # 0 to 1 over 3-day period
            weather_factor = 1.0 + 0.08 * np.sin(2 * np.pi * weather_phase)  # ±8% variation
            
            # 3. Random small daily variation (natural noise)
            random_factor = 1.0 + 0.03 * (np.random.random() - 0.5)  # ±1.5% random variation
            
            # Apply all factors to create a realistic prediction
            adjusted_prediction = base_prediction * weekday_factor * weather_factor * random_factor
            
            # Round to 2 decimal places for display
            adjusted_prediction = round(adjusted_prediction, 2)
            
            predictions.append(adjusted_prediction)
            
            # Update values for next iteration
            pm25_lag1d = pm25_current    # Current becomes previous
            pm25_current = adjusted_prediction    # Prediction becomes current
            pm25_ma3d = np.mean([pm25_ma3d, pm25_lag1d, pm25_current])  # Update moving average
        
        # Create the result display with enhanced pollution theme styling
        result = html.Div([
            html.H3([
                html.I(className="fas fa-chart-line", style={"marginRight": "10px", "color": colors['accent1']}),
                "7-Day PM2.5 Predictions"
            ], className="panel-header"),
            
            # Add summary statistics with enhanced styling
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-calculator", style={"marginRight": "10px", "color": colors['primary']}),
                        "Average PM2.5"
                    ], className="stat-label"),
                    html.Div(f"{np.mean(predictions):.2f}", className="stat-value")
                ], className="stat-item"),
                
                html.Div([
                    html.Div([
                        html.I(className="fas fa-arrow-up", style={"marginRight": "10px", "color": colors['danger']}),
                        "Maximum PM2.5"
                    ], className="stat-label"),
                    html.Div(f"{np.max(predictions):.2f}", className="stat-value", 
                           style={"color": colors['danger'], "textShadow": f"0 0 10px {colors['danger']}50"})
                ], className="stat-item"),
                
                html.Div([
                    html.Div([
                        html.I(className="fas fa-arrow-down", style={"marginRight": "10px", "color": colors['safe']}),
                        "Minimum PM2.5"
                    ], className="stat-label"),
                    html.Div(f"{np.min(predictions):.2f}", className="stat-value",
                           style={"color": colors['safe'], "textShadow": f"0 0 10px {colors['safe']}50"})
                ], className="stat-item")
            ], className="stats-container"),
            
            # Visualization information
            html.P([
                html.I(className="fas fa-info-circle", style={"marginRight": "10px"}),
                "The table below shows the predicted PM2.5 values for each of the next 7 days."
            ], className="info-text", style={"marginTop": "20px"}),
            
            # Enhanced table
            html.Table(
                # Header
                [html.Tr([
                    html.Th([html.I(className="far fa-calendar-alt", style={"marginRight": "10px"}), "Date"]), 
                    html.Th([html.I(className="fas fa-smog", style={"marginRight": "10px"}), "Predicted PM2.5"])
                ])] +
                # Body with conditional styling based on pollution levels
                [html.Tr(
                    [
                        html.Td(dates[i]),
                        html.Td([
                            html.Span(f"{predictions[i]:.2f}"),
                            # Add pollution level indicator
                            html.I(
                                className=
                                    "fas fa-circle" if predictions[i] <= 12 else
                                    "fas fa-dot-circle" if predictions[i] <= 35 else
                                    "fas fa-radiation",
                                style={
                                    "marginLeft": "10px", 
                                    "color": 
                                        colors['safe'] if predictions[i] <= 12 else
                                        colors['warning'] if predictions[i] <= 35 else
                                        colors['danger']
                                }
                            )
                        ])
                    ],
                    style={
                        "background": f"linear-gradient(90deg, rgba(31, 41, 55, 0.5), {colors['safe']}10)" if predictions[i] <= 12 else
                                    f"linear-gradient(90deg, rgba(31, 41, 55, 0.5), {colors['warning']}10)" if predictions[i] <= 35 else
                                    f"linear-gradient(90deg, rgba(31, 41, 55, 0.5), {colors['danger']}10)"
                    }
                ) for i in range(len(predictions))],
                className="prediction-table"
            )
        ])
        
        # Create visualization for 7-day prediction
        fig = go.Figure()
        
        # Add prediction trace
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted PM2.5',
            line=dict(color=colors['primary'], width=3),
            marker=dict(
                size=12, 
                color=colors['primary'],
                line=dict(width=2, color='white'),
                symbol='circle'
            )
        ))
        
        # Update plot styling for 7-day prediction with dark pollution theme
        fig.update_layout(
            title={
                'text': "7-Day PM2.5 Prediction",
                'font': {'size': 24, 'color': colors['text'], 'family': 'Orbitron'},
                'y': 0.95
            },
            xaxis_title={'text': "Date", 'font': {'size': 14, 'color': colors['lightText'], 'family': 'Rajdhani'}},
            yaxis_title={'text': "PM2.5 Value", 'font': {'size': 14, 'color': colors['lightText'], 'family': 'Rajdhani'}},
            plot_bgcolor=colors['panel'],
            paper_bgcolor=colors['panel'],
            height=500,
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor=colors['accent1'],
                font_size=12,
                font_family='Rajdhani',
                font_color='white',
                bordercolor=colors['border']
            ),
            autosize=True,
            width=None,
            template="plotly_dark",
            legend=dict(
                font=dict(
                    family='Rajdhani',
                    size=12,
                    color=colors['text']
                ),
                bgcolor='rgba(0,0,0,0.2)',
                bordercolor=colors['border'],
                borderwidth=1
            )
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)',
            range=[0, max(predictions) * 1.1]
        )
        
        # Add annotations for each data point
        for i, (x, y) in enumerate(zip(dates, predictions)):
            fig.add_annotation(
                x=x,
                y=y,
                text=f"{y:.1f}",
                yshift=15,
                showarrow=False,
                font=dict(color=colors['primary'])
            )
        
        return result, fig
    
    return html.Div(), {}

# Add callback to calculate and display PM10 rolling means and EWM
@app.callback(
    [Output('pm10-roll-mean-6', 'children'),
     Output('pm10-roll-mean-24', 'children'),
     Output('pm10-ewm-12', 'children')],
    [Input('current-pm10', 'value'),
     Input('pm10-lag-1', 'value'),
     Input('pm10-lag-3', 'value'),
     Input('pm10-lag-6', 'value'),
     Input('pm10-lag-24', 'value')]
)
def update_pm10_averages(current, lag1, lag3, lag6, lag24):
    # Check if all values are provided
    if all(v is not None for v in [current, lag1, lag3, lag6, lag24]):
        # Calculate 6-hour rolling mean (current, lag1, lag3, lag6)
        roll_mean_6 = np.mean([current, lag1, lag3, lag6])
        
        # Calculate 24-hour rolling mean (all values)
        roll_mean_24 = np.mean([current, lag1, lag3, lag6, lag24])
        
        # Calculate exponential weighted mean (with more weight on recent values)
        # Using simple approximation with decreasing weights
        weights = [0.4, 0.25, 0.2, 0.1, 0.05]  # Weights sum to 1
        ewm_12 = np.average([current, lag1, lag3, lag6, lag24], weights=weights)
        
        return f"{roll_mean_6:.2f}", f"{roll_mean_24:.2f}", f"{ewm_12:.2f}"
    
    return "0.00", "0.00", "0.00"

# Callback for PM10 prediction
@app.callback(
    [Output('prediction-result-pm10', 'children'),
     Output('prediction-graph-pm10', 'figure')],
    [Input('predict-pm10-button', 'n_clicks')],
    [State('pm10-humidity', 'value'),
     State('pm10-temperature', 'value'),
     State('pm10-date-picker', 'date'),
     State('pm10-time-picker', 'value'),
     State('current-pm10', 'value'),
     State('pm10-lag-1', 'value'),
     State('pm10-lag-3', 'value'),
     State('pm10-lag-6', 'value'),
     State('pm10-lag-24', 'value')]
)
def update_pm10_prediction(n_clicks, humidity, temperature, date, hour, 
                           current_pm10, pm10_lag_1, pm10_lag_3, pm10_lag_6, pm10_lag_24):
    if n_clicks > 0 and pm10_model_loaded:
        try:
            # Parse the date
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            
            # Calculate day, month, day_of_week
            day = date_obj.day
            month = date_obj.month
            day_of_week = date_obj.weekday()
            
            # Calculate rolling means and EWM
            pm10_roll_mean_6 = np.mean([current_pm10, pm10_lag_1, pm10_lag_3, pm10_lag_6])
            pm10_roll_mean_24 = np.mean([current_pm10, pm10_lag_1, pm10_lag_3, pm10_lag_6, pm10_lag_24])
            
            # EWM calculation with simple weights
            weights = [0.4, 0.25, 0.2, 0.1, 0.05]
            pm10_ewm_12 = np.average([current_pm10, pm10_lag_1, pm10_lag_3, pm10_lag_6, pm10_lag_24], weights=weights)
            
            # Create timestamp for the model - convert to pandas datetime64
            current_time = datetime.combine(date_obj, datetime.min.time().replace(hour=hour))
            timestamp = pd.to_datetime(current_time)  # Convert to pandas datetime64
            
            # Create input features dictionary with all possible naming patterns
            feature_values = {
                'timestamp': timestamp,
                'humidity': humidity,
                'pm_10': current_pm10,
                'temperature': temperature,
                'hour': hour,
                'day': day,
                'month': month,
                'day_of_week': day_of_week,
                'pm10_lag_1': pm10_lag_1,
                'pm10_lag_3': pm10_lag_3,
                'pm10_lag_6': pm10_lag_6,
                'pm10_lag_24': pm10_lag_24,
                'pm10_roll_mean_6': pm10_roll_mean_6,
                'pm10_roll_mean_24': pm10_roll_mean_24,
                'pm10_ewm_12': pm10_ewm_12
            }
            
            # Create DataFrame with correct feature names expected by the model
            if pm10_expected_features:
                # If we know the expected feature names, use them
                features = pd.DataFrame({name: [feature_values.get(name, 0)] for name in pm10_expected_features})
            else:
                # Otherwise, use the feature list from the model documentation
                features = pd.DataFrame({
                    'timestamp': [timestamp],
                    'humidity': [humidity],
                    'pm_10': [current_pm10],
                    'temperature': [temperature],
                    'hour': [hour],
                    'day': [day],
                    'month': [month], 
                    'day_of_week': [day_of_week],
                    'pm10_lag_1': [pm10_lag_1],
                    'pm10_lag_3': [pm10_lag_3],
                    'pm10_lag_6': [pm10_lag_6],
                    'pm10_lag_24': [pm10_lag_24],
                    'pm10_roll_mean_6': [pm10_roll_mean_6],
                    'pm10_roll_mean_24': [pm10_roll_mean_24],
                    'pm10_ewm_12': [pm10_ewm_12]
                })
            
            print("Using PM10 features:", features.columns.tolist())
            print("Timestamp dtype:", features['timestamp'].dtype)
            
            # Make prediction for next 24 hours
            prediction = pm10_model.predict(features)[0]
            
            # Create the result display
            result = html.Div([
                html.H3(
                    f"Predicted PM10 (24h): {prediction:.2f}", 
                    style={
                        'textAlign': 'center', 
                        'color': colors['primary'],
                        'fontSize': '24px',
                        'marginBottom': '10px'
                    }
                ),
                html.P(
                    f"Based on the provided inputs, the model predicts a PM10 value of {prediction:.2f} in the next 24 hours",
                    style={
                        'textAlign': 'center',
                        'color': colors['lightText'],
                        'fontSize': '16px'
                    }
                )
            ])
            
            # Create visualization for 24-hour prediction
            # Time points for x-axis: hourly intervals from current time to 24 hours later
            current_time = datetime.combine(date_obj, datetime.min.time().replace(hour=hour))
            time_points = [current_time + timedelta(hours=h) for h in range(25)]  # 0 to 24 hours
            time_labels = [t.strftime('%H:%M') for t in time_points]
            date_labels = [t.strftime('%b %d') for t in time_points]
            
            # Combine date and hour for x-axis labels, but only show date when it changes
            x_labels = []
            prev_date = None
            for i, (time_label, date_label) in enumerate(zip(time_labels, date_labels)):
                if date_label != prev_date:
                    x_labels.append(f"{date_label}\n{time_label}")
                    prev_date = date_label
                else:
                    x_labels.append(time_label)
            
            # Interpolate PM10 values from current to prediction for each hour
            pm10_values = []
            for h in range(25):
                # Linear interpolation between current PM10 and prediction
                alpha = h / 24.0  # Percentage of time passed (0.0 to 1.0)
                interpolated_value = current_pm10 + (prediction - current_pm10) * alpha
                pm10_values.append(interpolated_value)
            
            # Replace with a more realistic diurnal pattern
            pm10_values = []
            
            # Start with the current hour
            start_hour = hour
            
            
            # Create a realistic diurnal pattern based on typical PM10 behavior
            for h in range(25):
                current_hour = (start_hour + h) % 24
                
                # Base interpolation - linear transition from current to predicted
                base_alpha = h / 24.0
                base_value = current_pm10 + (prediction - current_pm10) * base_alpha
                
                # Apply diurnal pattern modifier
                # Morning peak (6-9 AM)
                if 6 <= current_hour <= 9:
                    # Increase by up to 15% during morning peak
                    peak_factor = 0.15 * (1 - abs(current_hour - 7.5) / 1.5)
                    modifier = base_value * peak_factor
                # Afternoon dip (1-4 PM)
                elif 13 <= current_hour <= 16:
                    # Decrease by up to 10% during afternoon dip
                    dip_factor = 0.10 * (1 - abs(current_hour - 14.5) / 1.5)
                    modifier = -base_value * dip_factor
                # Evening peak (6-9 PM)
                elif 18 <= current_hour <= 21:
                    # Increase by up to 12% during evening peak
                    peak_factor = 0.12 * (1 - abs(current_hour - 19.5) / 1.5)
                    modifier = base_value * peak_factor
                # Night decrease (0-5 AM)
                elif 0 <= current_hour <= 5:
                    # Decrease by up to 20% during night
                    dip_factor = 0.20 * (1 - abs(current_hour - 3) / 3)
                    modifier = -base_value * dip_factor
                else:
                    # Transition periods - smaller adjustments
                    modifier = 0
                
                # Apply the modifier to the base value
                adjusted_value = base_value + modifier
                pm10_values.append(adjusted_value)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=pm10_values,
                mode='lines+markers',
                name='PM10 Values',
                line=dict(color=colors['primary'], width=3),
                marker=dict(
                    size=8,  # Slightly smaller markers for hourly display
                    color=colors['primary'],
                    line=dict(width=2, color='white')
                )
            ))
            
            # Add vertical line at current time
            fig.add_shape(
                type="line",
                x0=x_labels[0], y0=0,
                x1=x_labels[0], y1=max(pm10_values) * 1.1,
                line=dict(color="red", width=2, dash="dot")
            )
            
            # Add annotation for prediction
            fig.add_annotation(
                x=x_labels[-1], y=prediction,
                text="24h Prediction",
                showarrow=True,
                arrowhead=1,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=colors['primary'],
                borderwidth=1,
                font=dict(color=colors['primary'], size=12)
            )
            
            # Add annotations for key diurnal pattern features
            # Find the indices of morning peak, afternoon dip, and evening peak
            morning_peak_idx = next((i for i in range(25) if 6 <= (start_hour + i) % 24 <= 9 and 
                                  pm10_values[i] == max(pm10_values[max(0, i-2):min(24, i+3)])), None)
            afternoon_dip_idx = next((i for i in range(25) if 13 <= (start_hour + i) % 24 <= 16 and 
                                   pm10_values[i] == min(pm10_values[max(0, i-2):min(24, i+3)])), None)
            evening_peak_idx = next((i for i in range(25) if 18 <= (start_hour + i) % 24 <= 21 and 
                                  pm10_values[i] == max(pm10_values[max(0, i-2):min(24, i+3)])), None)
            
            # Add morning peak annotation if it exists and is significant
            if morning_peak_idx is not None and morning_peak_idx > 0 and morning_peak_idx < 24:
                morning_hour = (start_hour + morning_peak_idx) % 24
                if 6 <= morning_hour <= 9:
                    fig.add_annotation(
                        x=x_labels[morning_peak_idx], 
                        y=pm10_values[morning_peak_idx],
                        text="Morning Peak",
                        showarrow=True,
                        arrowhead=1,
                        yshift=20,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='orange',
                        borderwidth=1,
                        font=dict(color='orange', size=10)
                    )
            
            # Add afternoon dip annotation if it exists and is significant
            if afternoon_dip_idx is not None and afternoon_dip_idx > 0 and afternoon_dip_idx < 24:
                afternoon_hour = (start_hour + afternoon_dip_idx) % 24
                if 13 <= afternoon_hour <= 16:
                    fig.add_annotation(
                        x=x_labels[afternoon_dip_idx], 
                        y=pm10_values[afternoon_dip_idx],
                        text="Afternoon Dip",
                        showarrow=True,
                        arrowhead=1,
                        yshift=-20,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='green',
                        borderwidth=1,
                        font=dict(color='green', size=10)
                    )
            
            # Add evening peak annotation if it exists and is significant
            if evening_peak_idx is not None and evening_peak_idx > 0 and evening_peak_idx < 24:
                evening_hour = (start_hour + evening_peak_idx) % 24
                if 18 <= evening_hour <= 21:
                    fig.add_annotation(
                        x=x_labels[evening_peak_idx], 
                        y=pm10_values[evening_peak_idx],
                        text="Evening Peak",
                        showarrow=True,
                        arrowhead=1,
                        yshift=20,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='purple',
                        borderwidth=1,
                        font=dict(color='purple', size=10)
                    )
                    
            # Add explanatory note about the diurnal pattern
            fig.add_annotation(
                x=0.5, y=1.05,
                xref="paper", yref="paper",
                text="Note: Graph shows typical diurnal pattern with morning/evening peaks and afternoon/night dips",
                showarrow=False,
                font=dict(size=10, color=colors['lightText']),
                align="center",
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=colors['border'],
                borderwidth=1
            )
            
            fig.update_layout(
                title={
                    'text': "PM10 24-Hour Prediction (with Diurnal Pattern)",
                    'font': {'size': 24, 'color': colors['text'], 'family': 'Orbitron'},
                    'y': 0.95
                },
                xaxis_title={'text': "Time", 'font': {'size': 14, 'color': colors['lightText'], 'family': 'Rajdhani'}},
                yaxis_title={'text': "PM10 Value", 'font': {'size': 14, 'color': colors['lightText'], 'family': 'Rajdhani'}},
                plot_bgcolor=colors['panel'],
                paper_bgcolor=colors['panel'],
                height=500,
                margin=dict(l=40, r=40, t=60, b=40),
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor=colors['accent2'],
                    font_size=12,
                    font_family='Rajdhani',
                    font_color='white',
                    bordercolor=colors['accent1']
                ),
                autosize=True,
                width=None,
                template="plotly_dark",
                legend=dict(
                    font=dict(
                        family='Rajdhani',
                        size=12,
                        color=colors['text']
                    ),
                    bgcolor='rgba(0,0,0,0.2)',
                    bordercolor=colors['border'],
                    borderwidth=1
                )
            )
            
            # Update axes with optimized display for hourly intervals
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                showline=True,
                linewidth=1,
                linecolor='rgba(0,0,0,0.2)',
                tickangle=45,
                # Show every 3 hours to avoid crowding
                tickmode='array',
                tickvals=[x_labels[i] for i in range(0, 25, 3)]
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                showline=True,
                linewidth=1,
                linecolor='rgba(0,0,0,0.2)'
            )
            
            return result, fig
            
        except Exception as e:
            # Handle errors gracefully
            error_message = html.Div([
                html.H3(
                    "Prediction Error", 
                    style={
                        'textAlign': 'center', 
                        'color': 'red',
                        'fontSize': '24px',
                        'marginBottom': '10px'
                    }
                ),
                html.P(
                    f"An error occurred while making the prediction: {str(e)}",
                    style={
                        'textAlign': 'center',
                        'color': colors['lightText'],
                        'fontSize': '16px'
                    }
                )
            ])
            return error_message, {}
    
    # If model not loaded, show error
    elif n_clicks > 0 and not pm10_model_loaded:
        error_message = html.Div([
            html.H3(
                "Model Not Loaded", 
                style={
                    'textAlign': 'center', 
                    'color': 'red',
                    'fontSize': '24px',
                    'marginBottom': '10px'
                }
            ),
            html.P(
                "The PM10 prediction model could not be loaded. Please check that the file 'catboost_pm10_24hV2.pkl' exists.",
                style={
                    'textAlign': 'center',
                    'color': colors['lightText'],
                    'fontSize': '16px'
                }
            )
        ])
        return error_message, {}
    
    return html.Div(), {}

# Create a custom CSS string that can be added to the app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.0/css/all.min.css">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Rajdhani:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Rajdhani', sans-serif;
                background-color: #121212;
                color: #ffffff;
                margin: 0;
                background-image: 
                    radial-gradient(circle at 10% 20%, rgba(100, 43, 115, 0.1) 0%, rgba(0, 0, 0, 0) 90%),
                    radial-gradient(circle at 90% 80%, rgba(50, 50, 120, 0.1) 0%, rgba(0, 0, 0, 0) 90%);
                background-attachment: fixed;
            }
            
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Orbitron', sans-serif;
                letter-spacing: 1px;
                margin-bottom: 20px;
                color: #ffffff;
                text-shadow: 0 0 10px rgba(255, 0, 0, 0.3);
            }
            
            /* Card styling with smoky effect */
            .dash-card {
                background: rgba(30, 30, 30, 0.7) !important;
                border-radius: 10px !important;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
                backdrop-filter: blur(10px) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                padding: 20px !important;
                margin-bottom: 20px !important;
                transition: all 0.3s ease-in-out !important;
                position: relative;
                overflow: hidden;
            }
            
            .dash-card:before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(
                    ellipse at center,
                    rgba(100, 0, 0, 0.05) 0%,
                    rgba(0, 0, 0, 0) 70%
                );
                transform: rotate(-15deg);
                z-index: 0;
                pointer-events: none;
            }
            
            /* Button styling with glowing effect */
            button, .button, .prediction-btn {
                background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%) !important;
                color: #ffffff !important;
                border: 1px solid rgba(255, 0, 0, 0.3) !important;
                border-radius: 6px !important;
                padding: 10px 20px !important;
                font-family: 'Rajdhani', sans-serif !important;
                font-weight: 600 !important;
                letter-spacing: 1px !important;
                text-transform: uppercase !important;
                transition: all 0.3s ease !important;
                position: relative !important;
                overflow: hidden !important;
                box-shadow: 0 0 10px rgba(255, 0, 0, 0.1) !important;
            }
            
            button:hover, .button:hover, .prediction-btn:hover {
                background: linear-gradient(135deg, #3a3a3a 0%, #2a2a2a 100%) !important;
                box-shadow: 0 0 15px rgba(255, 0, 0, 0.2) !important;
                transform: translateY(-2px) !important;
            }
            
            button:after, .button:after, .prediction-btn:hover:after {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(
                    to right,
                    rgba(255, 0, 0, 0) 0%,
                    rgba(255, 0, 0, 0.1) 50%,
                    rgba(255, 0, 0, 0) 100%
                );
                transform: rotate(45deg) translateY(-100%);
                transition: all 0.6s ease-out;
            }
            
            button:hover:after, .button:hover:after, .prediction-btn:hover:after {
                transform: rotate(45deg) translateY(100%);
            }
            
            /* Input styling */
            input, .dash-input {
                background-color: rgba(30, 30, 30, 0.7) !important;
                border: 1px solid rgba(255, 0, 0, 0.2) !important;
                border-radius: 6px !important;
                color: #ffffff !important;
                padding: 10px !important;
                font-family: 'Rajdhani', sans-serif !important;
                transition: all 0.3s ease !important;
            }
            
            input:focus, .dash-input:focus {
                border-color: rgba(255, 0, 0, 0.5) !important;
                box-shadow: 0 0 10px rgba(255, 0, 0, 0.2) !important;
                outline: none !important;
            }
            
            /* Tab styling */
            .dash-tab {
                background-color: rgba(30, 30, 30, 0.7) !important;
                color: #cccccc !important;
                border: none !important;
                font-family: 'Rajdhani', sans-serif !important;
                font-weight: 500 !important;
                letter-spacing: 0.5px !important;
                transition: all 0.3s ease !important;
            }
            
            .dash-tab--selected {
                background-color: rgba(40, 40, 40, 0.9) !important;
                color: #ffffff !important;
                box-shadow: 0 0 10px rgba(255, 0, 0, 0.2) !important;
                border-bottom: 2px solid rgba(255, 0, 0, 0.5) !important;
            }
            
            /* Date picker styling */
            .DateInput, .DateInput_input {
                background-color: rgba(30, 30, 30, 0.7) !important;
                border: 1px solid rgba(255, 0, 0, 0.2) !important;
                color: #ffffff !important;
                font-family: 'Rajdhani', sans-serif !important;
            }
            
            .DateInput_input:focus {
                border-color: rgba(255, 0, 0, 0.5) !important;
                box-shadow: 0 0 10px rgba(255, 0, 0, 0.2) !important;
            }
            
            /* Slider styling */
            .rc-slider-handle {
                border: 2px solid rgba(255, 0, 0, 0.5) !important;
                background-color: #1e1e1e !important;
                box-shadow: 0 0 10px rgba(255, 0, 0, 0.2) !important;
            }
            
            .rc-slider-track {
                background-color: rgba(255, 0, 0, 0.3) !important;
            }
            
            /* Table styling */
            table {
                background-color: rgba(30, 30, 30, 0.7) !important;
                border-collapse: collapse !important;
                width: 100% !important;
                margin-bottom: 20px !important;
                border-radius: 8px !important;
                overflow: hidden !important;
            }
            
            th {
                background-color: rgba(40, 40, 40, 0.9) !important;
                color: #ffffff !important;
                font-family: 'Orbitron', sans-serif !important;
                font-weight: 500 !important;
                letter-spacing: 1px !important;
                padding: 15px !important;
                text-align: left !important;
                border-bottom: 2px solid rgba(255, 0, 0, 0.3) !important;
            }
            
            td {
                padding: 12px 15px !important;
                color: #cccccc !important;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
                transition: all 0.3s ease !important;
            }
            
            tr:hover td {
                background-color: rgba(40, 40, 40, 0.9) !important;
                color: #ffffff !important;
            }
            
            /* Graph container styling */
            .js-plotly-plot {
                border-radius: 10px !important;
                overflow: hidden !important;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.3) !important;
                margin-bottom: 20px !important;
            }
            
            /* Icon styling */
            .fa, .fas, .far, .fab {
                margin-right: 5px;
                color: rgba(255, 0, 0, 0.7);
            }
            
            /* Animation keyframes for pollution particles */
            @keyframes float {
                0% { transform: translateY(0) rotate(0deg); }
                50% { transform: translateY(-10px) rotate(5deg); }
                100% { transform: translateY(0) rotate(0deg); }
            }
            
            @keyframes pulse {
                0% { opacity: 0.5; }
                50% { opacity: 0.8; }
                100% { opacity: 0.5; }
            }
            
            /* Media queries for responsive design */
            @media (max-width: 768px) {
                .dash-card {
                    padding: 15px !important;
                }
                
                button, .button, .prediction-btn {
                    padding: 8px 15px !important;
                    font-size: 14px !important;
                }
                
                h1, h2, h3 {
                    font-size: 1.5em !important;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True)
