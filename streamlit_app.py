import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import hashlib
import base64
import os
import numpy as np

# ============================================================================
# SEGMENT DEFINITIONS & ROUTE MAPPING
# ============================================================================

# Import segment definitions from model
SEGMENTS = {
    'segment_01': 'I0040_Seg26',
    'segment_02': 'I0040_Seg27',
    'segment_03': 'I0040_Seg28',
    'segment_04': 'I55_Seg05',
    'segment_05': 'I240_Seg02',
    'segment_06': 'I240_Seg03',
    'segment_07': 'I240_Seg05',
    'segment_08': 'I240_Seg08',
    'segment_09': 'I240_Seg11',
    'segment_10': 'I240_Seg12',
    'segment_11': 'I240_Seg13',
}

def get_segment_display_name(segment_id):
    """Convert segment_01 to readable format like 'I-40 Seg26'"""
    if segment_id in SEGMENTS:
        route_seg = SEGMENTS[segment_id]  # e.g., 'I0040_Seg26'
        parts = route_seg.split('_')
        route = parts[0]  # 'I0040'
        seg = parts[1]    # 'Seg26'
        
        # Format route name
        if route == "I0040":
            route_name = "I-40"
        elif route == "I55":
            route_name = "I-55"
        elif route == "I240":
            route_name = "I-240"
        else:
            route_name = route
        
        return f"{route_name} {seg}"
    return segment_id

def create_route_segment_mapping():
    """Parse SEGMENTS dictionary to create route-based grouping"""
    mapping = {"All": list(SEGMENTS.keys())}
    
    for seg_id, route_seg in SEGMENTS.items():
        # Parse "I0040_Seg26" → route="I-40"
        route = route_seg.split('_')[0]
        
        # Format: I0040 → I-40, I55 → I-55, I240 → I-240
        if route == "I0040":
            route_name = "I-40"
        elif route == "I55":
            route_name = "I-55"
        elif route == "I240":
            route_name = "I-240"
        else:
            route_name = route
        
        if route_name not in mapping:
            mapping[route_name] = []
        mapping[route_name].append(seg_id)
    
    return mapping

ROUTE_SEGMENTS = create_route_segment_mapping()

# # Debug print (remove in production)
# print("=" * 60)
# print("ROUTE SEGMENTS MAPPING:")
# for route, segments in ROUTE_SEGMENTS.items():
#     print(f"\n{route}:")
#     for seg in segments:
#         print(f"  {seg} → {get_segment_display_name(seg)}")
# print("=" * 60)
# ============================================================================
# AUTHENTICATION
# ============================================================================
def hash_password(p): 
    return hashlib.sha256(p.encode()).hexdigest()

USERS = {"Safe_TN": hash_password("ctiersafety_1")}

def check_login(u, p): 
    return u in USERS and USERS[u] == hash_password(p)

def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def login_page():
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 25%, #e2e8f0 50%, #cbd5e1 75%, #94a3b8 100%);
        background-attachment: fixed;
    }
    .centered-logo img {margin: 0 auto; display: block;}
    div[data-testid="stTextInput"] > div > div > input {
        border: 2px solid #e5e7eb; border-radius: 10px; padding: 12px 16px; font-size: 15px;
    }
    div[data-testid="stTextInput"] > div > div > input:focus {
        border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white; font-weight: 600; font-size: 16px; padding: 12px 24px;
        border-radius: 10px; border: none; box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    }
    .info-box {
        background-color: #dbeafe; border-left: 4px solid #2563eb; padding: 20px;
        border-radius: 8px; margin: 20px 0; text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.9, 1.2, 1.9])
    with col2:
        logo_base64 = get_image_base64("images/Safe_TN_Logo.png")
        if logo_base64:
            st.markdown(
                f'<div class="centered-logo"><img src="data:image/png;base64,{logo_base64}" width="500"></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown("<h1 style='text-align: center;'>SAFE TN</h1>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form(key="login_form_unique", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("Login", use_container_width=True)

            if submit:
                if username and password and check_login(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                elif username and password:
                    st.error("Invalid credentials")
                else:
                    st.warning("Please fill both fields")

        st.markdown("---")
        st.markdown("""
        <div class="info-box" style="text-align: justify;">
            <strong>SAFE TN</strong> developed by the Center for Transportation Innovation, Education, and Research (C-TIER) to support transportation practitioners and enforcement agencies in proactively identifying and understanding roadway safety risks across Tennessee. <br>
            Activities are monitored • Unauthorized access prohibited
        </div>
        """, unsafe_allow_html=True)
        st.caption("Support: ctiermemphis@gmail.com")

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

# ============================================================================
# CONFIG & STYLE
# ============================================================================
st.set_page_config(
    page_title="SAFE TN – Crash Risk Prediction", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    h1 {color:#1f77b4; font-size: 36px !important;}
    h2 {color:#1f77b4; font-size: 28px !important;}
    h3 {color:#1f77b4; font-size: 22px !important;}
    .stMetric {background:#f0f2f6; padding:18px; border-radius:12px; box-shadow:0 3px 10px rgba(0,0,0,0.1);}
    .stMetric label {font-size: 18px !important;}
    .stMetric [data-testid="stMetricValue"] {font-size: 32px !important;}
    .about-box {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6); 
        color:white; 
        padding:20px;
        border-radius:12px; 
        box-shadow:0 4px 15px rgba(0,0,0,0.2); 
        font-size:14.5px; 
        line-height:1.5;
        margin-bottom: 15px;
    }
    .about-box b {color:#fbbf24;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA & ROUTES
# ============================================================================
# ROUTE_SEGMENTS = {
#     "All":  ["segment_01", "segment_02", "segment_03", "segment_04", "segment_05", "segment_06", "segment_07", "segment_08", "segment_09", "segment_10", "segment_11"],
#     "I-40":  ["segment_01", "segment_02", "segment_03"],
#     "I-55":  ["segment_04"],
#     "I-240": ["segment_05", "segment_06", "segment_07", "segment_08", "segment_09", "segment_10", "segment_11"]
# }

@st.cache_data(ttl=3600)
def load_segment_data(segment_id):
    path = f"outputs/risk_score/{segment_id}/data/{segment_id}_future_predictions_with_risk.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['week_start'] = pd.to_datetime(df['week_start'])
        return df
    return None

@st.cache_data(ttl=3600)
def load_historical_data(segment_id):
    hist_path = f"outputs/risk_score/{segment_id}/data/{segment_id}_weekly_crashes.csv"
    print(f"Attempting to load historical data from: {hist_path}")  # Debug
    print(f"File exists: {os.path.exists(hist_path)}")  # Debug
    
    if os.path.exists(hist_path):
        try:
            df = pd.read_csv(hist_path)
            print(f"Loaded {len(df)} rows of historical data")  # Debug
            df['week_start'] = pd.to_datetime(df['week_start'])
            return df
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return pd.DataFrame(columns=['week_start', 'total_crashes'])
    else:
        print("Historical file not found")
        return pd.DataFrame(columns=['week_start', 'total_crashes'])

@st.cache_data(ttl=3600)
def load_segmented_data():
    """Load the segmented interstate data for background analysis"""
    try:
        df = pd.read_csv("data/Segmented_Shelby_Interstates.csv")
        
        # ADD THESE DEBUG LINES HERE:
        print("=" * 50)
        print("SEGMENTED DATA LOADED SUCCESSFULLY")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nUnique Routes: {df['Route'].unique()}")
        print(f"Unique Years: {sorted(df['Year Of Crash'].unique())}")
        print(f"Hit and Run values: {df['Hit and Run'].unique()}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        print("=" * 50)
        
        return df
    except Exception as e:
        print(f"Error loading segmented data: {e}")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_risk_level(val):
    if val < 0.5:  
        return "Very Low Risk",  "#28a745"
    elif val < 1.0: 
        return "Low Risk",      "#ffc107"
    elif val < 1.5: 
        return "Moderate Risk", "#fd7e14"
    elif val < 2.0: 
        return "High Risk",     "#dc3545"
    else:           
        return "Very High Risk", "#c82333"

# ============================================================================
# PLOTS
# ============================================================================
def create_historical_plot(historical_df):
    """Create a clean historical weekly crashes plot with thousand separators."""
    fig = go.Figure()
    
    if not historical_df.empty:
        # Calculate moving average for trend line
        window_size = 4  # 4-week moving average
        historical_df_sorted = historical_df.sort_values('week_start')
        moving_avg = historical_df_sorted['total_crashes'].rolling(window=window_size, center=True).mean()
        
        # Main line plot
        fig.add_trace(go.Scatter(
            x=historical_df_sorted['week_start'], 
            y=historical_df_sorted['total_crashes'],
            mode='lines', 
            name='Weekly Crashes',
            line=dict(color='#1f77b4', width=2.5),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Crashes: %{y:,.0f}<extra></extra>'  # Added comma
        ))
        
        # Add moving average trend line
        fig.add_trace(go.Scatter(
            x=historical_df_sorted['week_start'], 
            y=moving_avg,
            mode='lines', 
            name='4-Week Moving Avg',
            line=dict(color='red', width=2, dash='dot'),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Avg: %{y:,.1f}<extra></extra>'  # Added comma
        ))
    
    fig.update_layout(
        title={
            'text': '<b>Historical Weekly Crashes</b>',
            'font': {'size': 24, 'family': 'Arial', 'color': '#1f77b4'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title={
            'text': '<b>Date</b>',
            'font': {'size': 20, 'family': 'Arial', 'color': 'black'}
        },
        yaxis_title={
            'text': '<b>Total Crashes</b>',
            'font': {'size': 20, 'family': 'Arial', 'color': 'black'}
        },
        hovermode='x unified',
        height=500, 
        template='plotly_white',
        showlegend=True,
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.20, 
            xanchor="center", 
            x=0.5,
            font=dict(size=16, family='Arial', color='black')
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200,200,200,0.3)',
            title_font=dict(size=20, family='Arial', color='black'),
            tickfont=dict(size=16, family='Arial', color='black'),
            linecolor='black',
            linewidth=2,
            mirror=True
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200,200,200,0.3)',
            title_font=dict(size=20, family='Arial', color='black'),
            tickfont=dict(size=16, family='Arial', color='black'),
            linecolor='black',
            linewidth=2,
            mirror=True,
            separatethousands=True  # ADD THIS
        ),
        margin=dict(l=80, r=40, t=100, b=110)
    )
    return fig

def create_forecast_plot(future_df):
    """Create forecast plot with thousand separators."""
    fig = go.Figure()
    
    # Calculate confidence bounds
    se = 1.96 * np.sqrt(future_df['lambda'])
    upper = future_df['lambda'] + se
    lower = (future_df['lambda'] - se).clip(0)
    
    # Lower bound line
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=lower,
        mode='lines', 
        name='Lower Bound (95%)',
        line=dict(color='#28a745', width=2.5, dash='dash'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Lower Bound: %{y:,.2f}<extra></extra>'  # Added comma
    ))
    
    # Upper bound line
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=upper,
        mode='lines', 
        name='Upper Bound (95%)',
        line=dict(color='#dc3545', width=2.5, dash='dash'),
        fill='tonexty', 
        fillcolor='rgba(255,127,14,0.15)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Upper Bound: %{y:,.2f}<extra></extra>'  # Added comma
    ))
    
    # Mean forecast line WITH DATA LABELS
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=future_df['lambda'],
        mode='lines+markers+text',
        name='Mean Forecast (λ)',
        line=dict(color='#ff7f0e', width=3.5),
        marker=dict(size=10, symbol='diamond', color='#ff7f0e', line=dict(color='black', width=1)),
        text=[f"{val:,.2f}" for val in future_df['lambda']],  # Added comma formatting
        textposition='top center',
        textfont=dict(size=12, color='black', family='Arial Black'),
        hovertemplate='<b>Week: %{x|%b %d, %Y}</b><br>Mean Forecast: %{y:,.2f}<extra></extra>'  # Added comma
    ))
    
    fig.update_layout(
        title={
            'text': '<b>Probabilistic Crash Forecast</b>',
            'font': {'size': 24, 'family': 'Arial', 'color': '#1f77b4'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title={
            'text': '<b>Week Start Date</b>',
            'font': {'size': 20, 'family': 'Arial', 'color': 'black'}
        },
        yaxis_title={
            'text': '<b>Expected Crashes per Week</b>',
            'font': {'size': 20, 'family': 'Arial', 'color': 'black'}
        },
        hovermode='x unified',
        height=500, 
        template='plotly_white',
        showlegend=True,
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.45, 
            xanchor="center", 
            x=0.5,
            font=dict(size=16, family='Arial', color='black')
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200,200,200,0.3)',
            title_font=dict(size=20, family='Arial', color='black'),
            tickfont=dict(size=13, family='Arial Black', color='black'),
            linecolor='black',
            linewidth=2,
            mirror=True,
            tickmode='array',
            tickvals=future_df['week_start'],
            ticktext=[date.strftime('%m-%d-%y') for date in future_df['week_start']],
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200,200,200,0.3)',
            title_font=dict(size=20, family='Arial', color='black'),
            tickfont=dict(size=16, family='Arial', color='black'),
            linecolor='black',
            linewidth=2,
            mirror=True,
            separatethousands=True  # ADD THIS
        ),
        margin=dict(l=80, r=40, t=100, b=140)
    )
    return fig

def get_most_likely_from_probabilities(row):
    """
    Calculate most likely outcome from probability distribution
    Returns: (most_likely_count, probability_percent)
    """
    probs = [
        row['prob_0_crash'],
        row['prob_1_crash'],
        row['prob_2_crash'],
        row['prob_3_crash'],
        row['prob_ge4_crash']
    ]
    
    most_likely_idx = probs.index(max(probs))
    crash_counts = [0, 1, 2, 3, "4+"]
    
    return crash_counts[most_likely_idx], probs[most_likely_idx]


def create_dual_gauges(row):
    """Create dual gauges with consistent most likely calculation"""
    # Use probability distribution to find most likely (CONSISTENT)
    most_likely_crashes, prob_percent = get_most_likely_from_probabilities(row)
    
    # For gauge display, convert "4+" to 4
    exp = 4 if most_likely_crashes == "4+" else int(most_likely_crashes)
    prob = prob_percent  # Convert to percentage
    
    mean = row['lambda']
    level, color = get_risk_level(mean)
    
    # Gauge 1: Most Likely Outcome
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=exp,
        title={'text': "<b>Most Likely Outcome</b>", 'font': {'size': 22, 'family': 'Arial', 'color': 'black'}},
        gauge={
            'axis': {'range': [0, 5], 'tickfont': {'size': 16, 'color': 'black'}}, 
            'bar': {'color': color},
            'steps': [
                {'range': [0,1], 'color': '#d4edda'}, 
                {'range': [1,2], 'color': '#fff3cd'},
                {'range': [2,3], 'color': '#f8d7da'}, 
                {'range': [3,5], 'color': '#f5c6cb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': exp
            }
        },
        number={
            'suffix': " crashes" if most_likely_crashes == "4+" else (" crash" if exp == 1 else " crashes"),
            'font': {'size': 28, 'family': 'Arial', 'color': 'black'},
            'prefix': "≥" if most_likely_crashes == "4+" else ""  # Show "≥4 crashes" for 4+
        }
    ))
    fig1.update_layout(
        height=310, 
        margin=dict(t=100, b=10, l=20, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Gauge 2: Probability of Most Likely
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=prob,
        title={'text': "<b>Probability of Most Likely</b>", 'font': {'size': 22, 'family': 'Arial', 'color': 'black'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'size': 16, 'color': 'black'}}, 
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 25], 'color': '#fee2e2'},
                {'range': [25, 50], 'color': '#fef3c7'},
                {'range': [50, 75], 'color': '#d1fae5'},
                {'range': [75, 100], 'color': '#a7f3d0'}
            ]
        },
        number={'suffix': "%", 'font': {'size': 28, 'family': 'Arial', 'color': 'black'}}
    ))
    fig2.update_layout(
        height=310, 
        margin=dict(t=100, b=10, l=20, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig1, fig2, level, color


def create_probability_pie_chart(row):
    """Create probability pie chart with consistent most likely calculation"""
    probs = [
        row['prob_0_crash'],
        row['prob_1_crash'],
        row['prob_2_crash'],
        row['prob_3_crash'],
        row['prob_ge4_crash']
    ]
    
    labels = ["0 Crashes", "1 Crash", "2 Crashes", "3 Crashes", "4+ Crashes"]
    
    # Use consistent calculation
    most_likely_crashes, prob_percent = get_most_likely_from_probabilities(row)
    most_likely_idx = probs.index(prob_percent)
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=[p for p in probs],  # Convert to percentages
        hole=0.4,
        marker=dict(
            colors=['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#991b1b'],
            line=dict(color='#ffffff', width=3)
        ),
        textinfo='label+percent',
        textposition='auto',
        textfont=dict(size=16, family='Arial', color='black'),
        hovertemplate='<b>%{label}</b><br>Probability: %{value:.1f}%<extra></extra>',
        pull=[0.1 if i == most_likely_idx else 0 for i in range(5)],
        sort=False
    )])

    # Center annotation with consistent values
    crash_label = f"{most_likely_crashes} Crash" if most_likely_crashes == 1 else f"{most_likely_crashes} Crashes"
    
    fig.add_annotation(
        text=f"<b>Most Likely:</b><br>{crash_label}<br><b>{prob_percent:.1f}%</b>",
        x=0.5, y=0.5,
        font=dict(size=20, color="white", family="Arial"),
        showarrow=False,
        bgcolor="#1f2937",
        bordercolor="#ffffff",
        borderwidth=2,
        borderpad=10,
        opacity=0.95
    )

    fig.update_layout(
        title={
            'text': '<b>Probability Distribution of Crash Counts</b>',
            'font': {'size': 22, 'family': 'Arial', 'color': '#1f77b4'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=620,
        showlegend=True,
        legend=dict(
            orientation="v", 
            yanchor="middle", 
            y=0.5, 
            xanchor="left", 
            x=1.02,
            font=dict(size=16, family='Arial', color='black')
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family="Arial", size=16, color='black'),
        margin=dict(l=20, r=140, t=80, b=20)
    )
    
    return fig

def create_monthly_crashes_plot(df, selected_year=None, selected_route=None, selected_segment=None):
    """Create monthly crash variation with thousand separators."""
    filtered_df = df.copy()
    if selected_year and selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['Route'] == selected_route]
    if selected_segment and selected_segment != "All Segments":
        filtered_df = filtered_df[filtered_df['Segment ID'] == selected_segment]
    
    # Convert date and extract month
    filtered_df['Crash Date'] = pd.to_datetime(filtered_df['Date of Crash'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['Crash Date'])
    filtered_df['Month_Num'] = filtered_df['Crash Date'].dt.month
    
    # Group by month number
    monthly_crashes = filtered_df.groupby('Month_Num').size().reset_index(name='Crashes')
    
    # Create all 12 months even if some have 0 crashes
    all_months = pd.DataFrame({'Month_Num': range(1, 13)})
    monthly_crashes = all_months.merge(monthly_crashes, on='Month_Num', how='left').fillna(0)
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_crashes['Month_Name'] = monthly_crashes['Month_Num'].apply(lambda x: month_names[x-1])
    
    # Normalize dot sizes
    max_crashes = monthly_crashes['Crashes'].max()
    if max_crashes > 0:
        monthly_crashes['Dot_Size'] = 20 + (monthly_crashes['Crashes'] / max_crashes) * 60
    else:
        monthly_crashes['Dot_Size'] = 20
    
    # Create color scale based on crash count
    monthly_crashes['Color'] = monthly_crashes['Crashes'].apply(
        lambda x: '#034fa0' if x > max_crashes * 0.75 else
                  '#0078d4' if x > max_crashes * 0.5 else
                  '#249ee4' if x > max_crashes * 0.25 else
                  '#54daff'
    )
    
    fig = go.Figure()
    
    # Add scatter plot with varying dot sizes
    fig.add_trace(go.Scatter(
        x=monthly_crashes['Month_Name'],
        y=monthly_crashes['Crashes'],
        mode='markers+text',
        marker=dict(
            size=monthly_crashes['Dot_Size'],
            color=monthly_crashes['Color'],
            line=dict(color='black', width=0.15),
            opacity=0.8
        ),
        text=[f"<b>{int(count):,}</b>" for count in monthly_crashes['Crashes']],  # Added comma formatting
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Total Crashes: %{y:,.0f}<extra></extra>',  # Added comma
        name='Crashes'
    ))
    
    # Add baseline at y=0
    fig.add_shape(
        type="line",
        x0=-0.5, x1=11.5,
        y0=0, y1=0,
        line=dict(color="black", width=2)
    )
    
    fig.update_layout(
        title={
            'text': '<b>Monthly Crash Variation</b>',
            'font': {'size': 24, 'family': 'Arial', 'color': '#1f77b4'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title={'text': '<b>Month</b>', 'font': {'size': 20, 'family': 'Arial', 'color': 'black'}},
        yaxis_title={'text': '<b>Total Crashes</b>', 'font': {'size': 20, 'family': 'Arial', 'color': 'black'}},
        height=650,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=16, family='Arial Black', color='black'),
            linecolor='black',
            linewidth=2,
            mirror=True
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200,200,200,0.3)',
            tickfont=dict(size=16, family='Arial', color='black'),
            linecolor='black',
            linewidth=2,
            mirror=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            separatethousands=True  # ADD THIS
        ),
        margin=dict(l=80, r=40, t=100, b=80)
    )
    
    return fig

def create_day_night_crashes_plot(df, selected_year=None, selected_route=None):
    """Create day vs night crash variations with thousand separators."""
    filtered_df = df.copy()
    if selected_year and selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['Route'] == selected_route]
    
    # Clean up light condition column name
    light_col = None
    for col in filtered_df.columns:
        if 'light' in col.lower():
            light_col = col
            break
    
    if light_col is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Light Condition data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        return fig
    
    # Group by segment and light condition
    day_night_data = filtered_df.groupby(['Segment ID', light_col]).size().reset_index(name='Crashes')
    
    # Pivot to get day and night as separate columns
    pivot_data = day_night_data.pivot(index='Segment ID', columns=light_col, values='Crashes').fillna(0)
    
    # Get day and night columns
    daylight_cols = [col for col in pivot_data.columns if any(term in str(col).lower() for term in ['day', 'light', 'dawn', 'dusk'])]
    dark_cols = [col for col in pivot_data.columns if any(term in str(col).lower() for term in ['dark', 'night'])]
    
    day_crashes = pivot_data[daylight_cols].sum(axis=1) if daylight_cols else pd.Series(0, index=pivot_data.index)
    night_crashes = pivot_data[dark_cols].sum(axis=1) if dark_cols else pd.Series(0, index=pivot_data.index)
    
    segments = [str(seg) for seg in pivot_data.index.tolist()]
    
    fig = go.Figure()
    
    # Day crashes
    fig.add_trace(go.Bar(
        x=segments,
        y=day_crashes,
        name='Daytime',
        marker=dict(color='#ffa500', line=dict(color='black', width=1.5)),
        text=[f"{int(v):,}" for v in day_crashes],  # Added comma formatting
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Daytime Crashes: %{y:,.0f}<extra></extra>'  # Added comma
    ))
    
    # Night crashes
    fig.add_trace(go.Bar(
        x=segments,
        y=night_crashes,
        name='Nighttime',
        marker=dict(color='#191970', line=dict(color='black', width=1.5)),
        text=[f"{int(v):,}" for v in night_crashes],  # Added comma formatting
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Nighttime Crashes: %{y:,.0f}<extra></extra>'  # Added comma
    ))
    
    fig.update_layout(
        title={
            'text': '<b>Day vs Night Crash Variations Across Segments</b>',
            'font': {'size': 24, 'family': 'Arial', 'color': '#1f77b4'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title={'text': '<b>Segment ID</b>', 'font': {'size': 20, 'family': 'Arial', 'color': 'black'}},
        yaxis_title={'text': '<b>Number of Crashes</b>', 'font': {'size': 20, 'family': 'Arial', 'color': 'black'}},
        height=650,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=16, family='Arial', color='black')
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=16, family='Arial Black', color='black'),
            linecolor='black',
            linewidth=2,
            mirror=True
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200,200,200,0.3)',
            tickfont=dict(size=16, family='Arial', color='black'),
            linecolor='black',
            linewidth=2,
            mirror=True,
            separatethousands=True  # ADD THIS
        ),
        margin=dict(l=80, r=40, t=100, b=110)
    )
    return fig

def create_segment_ranking_plots(df, selected_year=None, selected_route=None):
    """Create two ranking plots with thousand separators."""
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_year and selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['Route'] == selected_route]
    
    # Define colors for each route
    route_colors = {
        'I0040': '#1f77b4',
        'I0055': '#a41020',
        'I0240': '#21A366'
    }
    
    # ========== PLOT 1: Top 10 Segments by Total Crashes ==========
    crash_counts = filtered_df.groupby(['Segment ID', 'Route']).size().reset_index(name='Total Crashes')
    crash_counts = crash_counts.sort_values('Total Crashes', ascending=False).head(10)
    crash_counts['Rank'] = range(1, len(crash_counts) + 1)
    crash_counts['Color'] = crash_counts['Route'].map(route_colors)
    
    fig1 = go.Figure()
    
    fig1.add_trace(go.Bar(
        x=crash_counts['Rank'],
        y=crash_counts['Total Crashes'],
        marker=dict(
            color=crash_counts['Color'],
            line=dict(color='black', width=2)
        ),
        width=0.6,
        text=[f"<b>{seg}</b><br>{count:,} crashes"  # Added comma formatting
              for seg, count in zip(crash_counts['Segment ID'], crash_counts['Total Crashes'])],
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>Rank: %{x}</b><br>Segment: %{customdata[0]}<br>Route: %{customdata[1]}<br>Total Crashes: %{y:,.0f}<extra></extra>',  # Added comma
        customdata=crash_counts[['Segment ID', 'Route']].values
    ))
    
    fig1.update_layout(
        title={
            'text': '<b>✨ Top 10 Segments by Total Crashes</b>',
            'font': {'size': 24, 'family': 'Arial', 'color': '#1f77b4'},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.98,
            'yanchor': 'top'
        },
        xaxis_title={
            'text': '<b>Rank (1 = Highest Crashes)</b>',
            'font': {'size': 19, 'family': 'Arial', 'color': 'black'}
        },
        yaxis_title={
            'text': '<b>Number of Crashes</b>',
            'font': {'size': 19, 'family': 'Arial', 'color': 'black'}
        },
        height=650,
        template='plotly_white',
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        xaxis=dict(
            showgrid=False,
            title_font=dict(size=19, family='Arial', color='black'),
            tickfont=dict(size=16, family='Arial Black', color='black'),
            linecolor='black',
            linewidth=2.5,
            mirror=True,
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 10.5]
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1.5,
            gridcolor='rgba(200,200,200,0.4)',
            title_font=dict(size=19, family='Arial', color='black'),
            tickfont=dict(size=16, family='Arial', color='black'),
            linecolor='black',
            linewidth=2.5,
            mirror=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            separatethousands=True  # ADD THIS
        ),
        margin=dict(l=80, r=40, t=90, b=80),
        bargap=0.05,
        bargroupgap=0
    )
    
    # ========== PLOT 2: Top 10 Segments by Hit and Run Cases ==========
    hit_run_df = filtered_df[filtered_df['Hit and Run'] == 'Yes'].copy()
    hit_run_counts = hit_run_df.groupby(['Segment ID', 'Route']).size().reset_index(name='Hit and Run Cases')
    hit_run_counts = hit_run_counts.sort_values('Hit and Run Cases', ascending=False).head(10)
    hit_run_counts['Rank'] = range(1, len(hit_run_counts) + 1)
    hit_run_counts['Color'] = hit_run_counts['Route'].map(route_colors)
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        x=hit_run_counts['Rank'],
        y=hit_run_counts['Hit and Run Cases'],
        marker=dict(
            color=hit_run_counts['Color'],
            line=dict(color='black', width=2)
        ),
        width=0.6,
        text=[f"<b>{seg}</b><br>{count:,} cases"  # Added comma formatting
              for seg, count in zip(hit_run_counts['Segment ID'], hit_run_counts['Hit and Run Cases'])],
        textposition='outside',
        textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>Rank: %{x}</b><br>Segment: %{customdata[0]}<br>Route: %{customdata[1]}<br>Hit & Run Cases: %{y:,.0f}<extra></extra>',  # Added comma
        customdata=hit_run_counts[['Segment ID', 'Route']].values
    ))
    
    fig2.update_layout(
        title={
            'text': '<b>✨ Top 10 Segments by Hit and Run Cases</b>',
            'font': {'size': 24, 'family': 'Arial', 'color': '#dc3545'},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.98,
            'yanchor': 'top'
        },
        xaxis_title={
            'text': '<b>Rank (1 = Highest Hit & Run)</b>',
            'font': {'size': 19, 'family': 'Arial', 'color': 'black'}
        },
        yaxis_title={
            'text': '<b>Number of Hit and Run Cases</b>',
            'font': {'size': 19, 'family': 'Arial', 'color': 'black'}
        },
        height=650,
        template='plotly_white',
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        xaxis=dict(
            showgrid=False,
            title_font=dict(size=19, family='Arial', color='black'),
            tickfont=dict(size=16, family='Arial Black', color='black'),
            linecolor='black',
            linewidth=2.5,
            mirror=True,
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 10.5]
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1.5,
            gridcolor='rgba(200,200,200,0.4)',
            title_font=dict(size=19, family='Arial', color='black'),
            tickfont=dict(size=16, family='Arial', color='black'),
            linecolor='black',
            linewidth=2.5,
            mirror=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            separatethousands=True  # ADD THIS
        ),
        margin=dict(l=80, r=40, t=90, b=80),
        bargap=0.05,
        bargroupgap=0
    )
    
    return fig1, fig2
# ============================================================================
# MAIN APP PAGES
# ============================================================================
def show_forecast_page():
    st.title("🏎️ Traffic Crash Risk Prediction for Shelby County")
    
    # Load segmented data for background analysis
    segmented_df = load_segmented_data()
    
    if segmented_df is not None:
        st.markdown("---")
        st.subheader("🔅 Background: Crash Data Analysis")
        st.markdown("*Historical crash patterns and trends across interstate segments*")
        
        # Filters for background plots
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            years = ["All Years"] + sorted(segmented_df['Year Of Crash'].unique().tolist(), reverse=True)
            selected_year = st.selectbox("Filter by Year", years, key="bg_year")
        with col_filter2:
            routes = ["All Routes"] + sorted(segmented_df['Route'].unique().tolist())
            selected_route = st.selectbox("Filter by Route", routes, key="bg_route")
        with col_filter3:
            # Segment filter for monthly crashes plot only
            segments = ["All Segments"] + sorted(segmented_df['Segment ID'].unique().tolist())
            selected_segment = st.selectbox("Filter by Segment", segments, key="bg_segment", 
                                        help="This filter applies only to the Monthly Crashes Heatmap")
        
        # Monthly crashes heatmap (with segment filter)
        st.markdown("#### 🔅 Monthly Crash Trends")
        st.markdown("""
        <div style="border: 1px solid #1f77b4; border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        fig_monthly = create_monthly_crashes_plot(segmented_df, selected_year, selected_route, selected_segment)
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Day vs Night crashes plot (no segment filter - uses only year and route)
        st.markdown("#### 🔅 Day vs Night Crash Comparison")
        st.markdown("""
        <div style="border: 1px solid #1f77b4; border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        fig_daynight = create_day_night_crashes_plot(segmented_df, selected_year, selected_route)
        st.plotly_chart(fig_daynight, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
            
        # Create and display ranking plots
        st.markdown("#### 🔅 Top 10 Segment Rankings")
        fig_crash, fig_hitrun = create_segment_ranking_plots(segmented_df, selected_year, selected_route)
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.markdown("""
            <div style="border: 1px solid #1f77b4; border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            """, unsafe_allow_html=True)
            st.plotly_chart(fig_crash, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_plot2:
            st.markdown("""
            <div style="border: 1px solid #1f77b4; border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            """, unsafe_allow_html=True)
            st.plotly_chart(fig_hitrun, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Add legend for route colors
        st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <span style="color: #1f77b4; font-size: 16px; margin: 0 15px;">■ I0040</span>
            <span style="color: #a41020; font-size: 16px; margin: 0 15px;">■ I0055</span>
            <span style="color: #21A366; font-size: 16px; margin: 0 15px;">■ I0240</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

# Route & Segment selector
    col1, col2 = st.columns([1,2])
    with col1:
        route = st.selectbox("Select Route", options=list(ROUTE_SEGMENTS.keys()), key="route_sel")
    with col2:
        # Create display names for dropdown
        segment_options = ROUTE_SEGMENTS[route]
        segment_display = [get_segment_display_name(seg) for seg in segment_options]
        
        selected_display = st.selectbox("Select Segment", options=segment_display, key="segment_sel")
        
        # Map back to segment_id
        segment = segment_options[segment_display.index(selected_display)]

    # Load data
    future_df = load_segment_data(segment)
    if future_df is None or future_df.empty:
        st.error(f"No prediction data found for {segment}")
        st.info("Please check that the data files exist in the expected location.")
        return

    historical_df = load_historical_data(segment)

    st.markdown(f"### 🔅 Selected: **{route} ➡️ {get_segment_display_name(segment)}** • {len(future_df)} weeks prediction")

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.metric("✨ Total Expected (12 Weeks)", f"{future_df['lambda'].head(12).sum():.1f}")
    with c2: 
        st.metric("✨ Weekly Average (λ)", f"{future_df['lambda'].mean():.2f}")
    with c3:
        peak = future_df.loc[future_df['lambda'].idxmax()]
        st.metric("✨ Peak Risk Week", f"{peak['lambda']:.2f}", 
                 delta=peak['week_start'].strftime("%b %d"))

    st.markdown("---")
    
    # Historical crashes plot with border
    st.subheader("🔅 Historical Weekly Crashes")
    if not historical_df.empty:
        st.markdown("""
        <div style="border: 1px solid #1f77b4; border-radius: 10px; padding: 20px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        st.plotly_chart(create_historical_plot(historical_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No historical data available for this segment.")
        st.info(f"Expected file location: `outputs/risk_score/{segment}/data/{segment}_weekly_crashes.csv`")
    
    st.markdown("---")
    
    # Prediction plot with border
    st.subheader("🔅 Probabilistic Crash Prediction")
    st.markdown("""
    <div style="border: 1px solid #1f77b4; border-radius: 10px; padding: 20px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    """, unsafe_allow_html=True)
    st.plotly_chart(create_forecast_plot(future_df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # In-depth risk assessment
    st.subheader("🔅 In-Depth Risk Assessment")

    week_labels = [
        f"{r['week_start'].strftime('%b %d')} – {(r['week_start'] + timedelta(days=6)).strftime('%b %d, %Y')}" 
        for _, r in future_df.iterrows()
    ]
    chosen = st.selectbox("Select Week for Detailed Analysis", week_labels, index=0)
    row = future_df.iloc[week_labels.index(chosen)]

    # Risk level badge
    risk_level, risk_color = get_risk_level(row['lambda'])
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <span style="
            display: inline-block;
            padding: 14px 40px;
            font-size: 22px;
            font-weight: bold;
            color: white;
            background: {risk_color};
            border-radius: 50px;
            box-shadow: 0 8px 20px {risk_color}40;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        ">{risk_level}</span>
    </div>
    """, unsafe_allow_html=True)

    # Gauges + Pie Chart with individual borders
    colL, colR = st.columns(2)
    with colL:
        g1, g2, _, _ = create_dual_gauges(row)
        st.markdown("""
        <div style="border: 1px solid #1f77b4; border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        st.plotly_chart(g1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="border: 1px solid #1f77b4; border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        st.plotly_chart(g2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with colR:
        fig_pie = create_probability_pie_chart(row)
        st.markdown("""
        <div style="border: 1px solid #1f77b4; border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

def show_help_page():
    st.title("✍️ Help & User Guide")
    st.markdown("### Understanding Probabilistic Predictions")

    st.markdown("## ✍️ Reading the Predictions")
    st.markdown("""
    **Example Prediction:** Week 3 shows "1-2 crashes (Most likely: 1, 36%)"
    
    **This means:**
    - We expect between 1-2 crashes based on statistical models
    - The single most likely outcome is exactly 1 crash
    - There's a 36% probability of exactly 1 crash occurring
    - We're 95% confident the actual count will fall within the predicted range
    """)

    st.markdown("---")
    st.markdown("## ✍️ Risk Level Interpretation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **🟢 Very Low Risk**
        - Range: 0-1 crashes
        - Low probability of any crashes
        - Standard patrol operations sufficient
        """)
        st.warning("""
        **🟡 Moderate Risk**
        - Range: 1-2 crashes
        - Likely 1-2 crashes expected
        - Increase vigilance and monitoring
        """)
    with col2:
        st.info("""
        **🔵 Low Risk**
        - Range: 0-1 crashes
        - Moderate probability of crashes
        - Maintain readiness protocols
        """)
        st.error("""
        **🔴 High Risk**
        - Range: 2-3+ crashes
        - Multiple crashes highly likely
        - Maximum enforcement recommended
        """)

    st.markdown("---")
    st.markdown("## ❓ Frequently Asked Questions")

    with st.expander("Why show ranges instead of exact numbers?"):
        st.markdown("""
        Crashes are **random events** that cannot be predicted with 100% accuracy. Providing ranges:
        - 📝 Reflects real-world uncertainty
        - 📝 Helps with flexible resource planning
        - 📝 Prevents false precision
        - 📝 Shows probability distribution of different outcomes
        - 📝 Allows for better risk-based decision making
        """)

    with st.expander("How do I use 'Most Likely' values?"):
        st.markdown("""
        The 'Most Likely' value represents the **mode** of the probability distribution (the peak).
        
        **For operational planning:**
        - 🚔 Use the **range** for resource allocation and staffing
        - 🚔 Use **most likely** for base scenario planning
        - 🚔 Check **probability %** to gauge confidence level
        - 🚔 Consider all scenarios in the distribution for contingency planning
        """)

    with st.expander("What does 95% confidence mean?"):
        st.markdown("""
        A 95% confidence interval means:
        - 📈 In 95 out of 100 similar weeks, actual crashes will fall within this range
        - 📈 The range accounts for both model uncertainty and random variation
        - 📈 Wider ranges indicate more uncertainty (often with longer prediction horizons)
        - 📈 Short-term predictions tend to have narrower, more precise ranges
        """)

    with st.expander("How is the prediction generated?"):
        st.markdown("""
        Our prediction system uses:
        - ⚠️ **Machine learning models** trained on historical crash data
        - ⚠️ **Weather patterns** and seasonal factors
        - ⚠️ **Traffic volume** and flow characteristics
        - ⚠️ **Temporal patterns** (day of week, time of year)
        - ⚠️ **Statistical methods** to quantify uncertainty
        """)

    st.markdown("---")
    st.markdown("## 🔦 Technical Support")
    st.info("""
    **Need Help?**
    - 📧 Email: ctiermemphis@gmail.com
    - 🕐 Hours: Monday-Friday, 9AM-5PM CST
    - 🏢 C-TIER, The University of Memphis
    """)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Show login or main app
    if not st.session_state.authenticated:
        login_page()
    else:
        # Sidebar
        with st.sidebar:
            logo_base64 = get_image_base64("images/Safe_TN_Logo.png")
            if logo_base64:
                st.markdown(
                    f'<img src="data:image/png;base64,{logo_base64}" width="200">',
                    unsafe_allow_html=True
                )
            else:
                st.markdown("### SAFE TN")
            
            st.markdown("### Safety Analytics & Forecasting Environment")
            st.title("Let's Navigate")
            
            page = st.radio(
                "Go to", 
                ["Probabilistic Crash Prediction", "Help & Guide"], 
                label_visibility="collapsed", 
                key="nav_radio"
            )
            
            st.markdown("---")
            # ABOUT BOX – BEAUTIFUL BLUE
            st.markdown("""
            <div class="about-box">
                <b>About SAFE TN</b><br><br>
                SAFE TN (<i>Safety Analytics & Forecasting Environment for Tennessee</i>) is a probabilistic crash-risk prediction tool developed by the 
                <b>Center for Transportation Innovation, Education and Research (C-TIER)</b> at The University of Memphis for the 
                Tennessee Highway Safety Office and the Enforcement agencies.<br><br>
                Using advanced machine-learning techniques, it delivers weekly crash predictions with certain confidence intervals for Shelby County.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="about-box">
                <b>About C-TIER</b><br><br>
                The <i>Center for Transportation Innovation, Education and Research (C-TIER)</i> at The University of Memphis 
                has developed this predictive tool with the motive of enhancing traffic safety across Tennessee, by integrating 
                real-time traffic and crash data.<br><br>
                Our transportation safety research emphasizes probabilistic 
                prediction, behavioral analysis, and engineering solutions to support the <b>Tennessee Highway Safety Office</b>
                and local agencies in deploying precise, high-impact enforcement and infrastructure improvements.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.info("**Support**\nctiermemphis@gmail.com")

            st.write(f"**👤 Logged in as:** {st.session_state.username}")
            
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                logout()

        # Main content area
        if page == "Probabilistic Crash Prediction":
            show_forecast_page()
        elif page == "Help & Guide":
            show_help_page()
