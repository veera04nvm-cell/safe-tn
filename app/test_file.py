import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import hashlib
import base64
# ============================================================================
# AUTHENTICATION
# ============================================================================
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()
USERS = {"Safe_TN": hash_password("ctiersafety_1")}

def center_image(image_path, width):
    """Center an image using HTML"""
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="{image_path}" width="{width}" style="margin: 0 auto;">
        </div>
        """,
        unsafe_allow_html=True
    )
def check_login(u, p): return u in USERS and USERS[u] == hash_password(p)

def get_image_base64(image_path):
    """Convert image to base64 for HTML embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def login_page():
    """Enhanced login page with professional styling"""
    st.markdown("""
    <style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Full page background gradient - Professional Neutral Theme */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 25%, #e2e8f0 50%, #cbd5e1 75%, #94a3b8 100%);
        background-attachment: fixed;
    }
    
    /* Alternative Option 1: Warm Professional */
    /* .stApp {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 25%, #fde68a 50%, #fcd34d 75%, #fbbf24 100%);
        background-attachment: fixed;
    } */
    
    /* Alternative Option 2: Cool Gray Modern */
    /* .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 30%, #f3f4f6 60%, #e5e7eb 100%);
        background-attachment: fixed;
    } */
    
    /* Alternative Option 3: Soft Purple Tech */
    /* .stApp {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 30%, #e9d5ff 60%, #d8b4fe 100%);
        background-attachment: fixed;
    } */
    
    /* Alternative Option 4: Teal Professional */
    /* .stApp {
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 30%, #99f6e4 60%, #5eead4 100%);
        background-attachment: fixed;
    } */
    
    /* Alternative Option 5: Clean White to Light Gray */
    /* .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 50%, #f5f5f5 100%);
        background-attachment: fixed;
    } */
    
    /* Main container styling */
    .login-container {
        max-width: 480px;
        margin: 60px auto;
        padding: 0;
        text-align: center;
    }
    
    /* Logo container with gradient background */
    .logo-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #0f172a 100%);
        padding: 50px 40px;
        border-radius: 20px 20px 0 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    /* Form container */
    .form-box {
        background: #ffffff;
        padding: 40px;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        border-top: 3px solid #3b82f6;
    }
    
    /* Typography */
    .main-title {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #1e40af;
        margin: 25px 0 8px 0 !important;
        line-height: 1.3 !important;
    }
    
    .subtitle {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #2563eb;
        margin-bottom: 0 !important;
        letter-spacing: 1px;
    }
    
    /* Center image - HTML method */
    .centered-logo {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    .centered-logo img {
        margin: 0 auto !important;
        display: block !important;
    }
    
    /* Input field styling */
    div[data-testid="stTextInput"] > div > div > input {
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stTextInput"] > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
        margin-top: 10px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
    }
    
    /* Info box styling */
    .info-box {
        background-color: #dbeafe;
        border-left: 4px solid #2563eb;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        text-align: left;
    }
    
    .info-box strong {
        color: #1e40af;
    }
    
    .warning-item {
        margin: 8px 0;
        font-size: 14px;
        color: #1f2937;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main container
    col1, col2, col3 = st.columns([1.9, 1.2, 1.9])
    
    with col2:
        
        # Center logo using HTML
        logo_base64 = get_image_base64("images/Safe_TN_Logo.png")
        if logo_base64:
            st.markdown(
                f'''
                <div class="centered-logo">
                    <img src="data:image/png;base64,{logo_base64}" width="700">
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:
            st.warning("Logo image not found. Please ensure 'images/Safe_TN_Logo.png' exists.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # st.markdown('<p class="main-title">Safety Analytics & Forecasting Environment</p>', unsafe_allow_html=True)
        # st.markdown('<p class="subtitle">SAFE TN</p>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username and password:
                    if check_login(username, password):
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")
        
        # Information box
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            <strong>The SAFE TN services are provided by Center for Transportation Innovations Education and Research for stakeholders to visualize crash forecasts.</strong>
            <div class="warning-item">⚠️ End user activities are monitored and logged. Unauthorized access is prohibited.</div>
            <div class="warning-item">⚠️ By logging in, you agree to comply with all applicable policies and guidelines.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("For access issues, contact ctiermemphis@gmail.com for technical support")
        
        st.markdown('</div>', unsafe_allow_html=True)


def logout():
    """Logout user and clear session"""
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.rerun()

# ============================================================================
# CONFIG + CUSTOM STYLES
# ============================================================================
st.set_page_config(page_title="SAFE TN – Crash Risk Forecast", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    h1, h2, h3 {color:#1f77b4;}
    .stMetric {background:#f0f2f6; padding:18px; border-radius:12px; box-shadow:0 3px 10px rgba(0,0,0,0.1);}
    .about-box {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-size: 14.5px;
        line-height: 1.5;
        margin: 15px 0;
    }
    .about-box b {color: #fbbf24;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA
# ============================================================================
@st.cache_data
def load_data():
    try:
        weekly_df = pd.read_csv('data/weekly_crashes_enhanced.csv')
        weekly_df['week_start'] = pd.to_datetime(weekly_df['week_start'])
        future_df = pd.read_csv('data/future_predictions_with_intervals.csv')
        future_df['week_start'] = pd.to_datetime(future_df['week_start'])
        return weekly_df, future_df, None
    except Exception as e:
        return None, None, str(e)

# ============================================================================
# PLOTS (unchanged – your original preserved)
# ============================================================================
def create_forecast_plot_with_intervals(weekly_df, future_df):
    """Enhanced forecast plot with better styling and interactivity"""
    fig = go.Figure()
    
    # Historical data with trend line
    fig.add_trace(go.Scatter(
        x=weekly_df['week_start'], 
        y=weekly_df['total_crashes'],
        mode='lines+markers', 
        name='Historical Crashes',
        line=dict(color='#2563eb', width=3), 
        marker=dict(size=6, color='#2563eb', line=dict(width=1, color='white')),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Crashes: %{y:.0f}<extra></extra>',
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.1)'
    ))
    
    # Confidence interval fill
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=future_df['predicted_upper'],
        mode='lines', 
        name='Upper Bound (95% CI)',
        line=dict(color='rgba(100,149,237,0)', width=0), 
        showlegend=False, 
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=future_df['predicted_lower'],
        mode='lines', 
        name='95% Confidence Interval',
        line=dict(color='rgba(100,149,237,0)', width=0),
        fill='tonexty', 
        fillcolor='rgba(100,149,237,0.25)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Lower: %{y:.2f}<extra></extra>'
    ))
    
    # Prediction bounds
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=future_df['predicted_lower'],
        mode='lines', 
        name='Lower Bound (95% CI)',
        line=dict(color='#3b82f6', width=2, dash='dot'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Lower: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=future_df['predicted_upper'],
        mode='lines', 
        name='Upper Bound (95% CI)',
        line=dict(color='#3b82f6', width=2, dash='dot'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Upper: %{y:.2f}<extra></extra>'
    ))
    
    # Mean prediction with enhanced styling
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=future_df['predicted_mean'],
        mode='lines+markers', 
        name='Mean Prediction',
        line=dict(color='#f59e0b', width=4, dash='dash'),
        marker=dict(size=10, symbol='diamond', color='#f59e0b', 
                   line=dict(width=2, color='white')),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Predicted: %{y:.2f} crashes<extra></extra>'
    ))
    
    # Forecast start line
    last_date = weekly_df['week_start'].max()
    fig.add_vline(
        x=last_date, 
        line_dash="dash", 
        line_color="red", 
        line_width=3,
        annotation_text="Forecast Start",
        annotation_position="top left",
        annotation_font_size=14,
        annotation_font_color="red"
    )
    
    # Enhanced layout
    fig.update_layout(
        title=dict(
            text='<b>Weekly Crash History & Probabilistic Forecast</b>',
            font=dict(size=24, color='#1e40af'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Date',
            titlefont=dict(size=14, color='#64748b'),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            title='Expected Crashes per Week',
            titlefont=dict(size=14, color='#64748b'),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        hovermode='x unified',
        height=600,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.8)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=80, b=50, l=50, r=50)
    )
    return fig

def get_risk_level(val):
    if val < 0.5:   return "Very Low Risk", "#28a745"
    elif val < 1.0: return "Low Risk", "#ffc107"
    elif val < 1.5: return "Moderate Risk", "#fd7e14"
    elif val < 2.0: return "High Risk", "#dc3545"
    else:           return "Very High Risk", "#c82333"

def create_12week_bar_chart(future_df):
    """Enhanced bar chart with better styling and annotations"""
    colors = [get_risk_level(v)[1] for v in future_df['predicted_mean']]
    lower_err = future_df['predicted_mean'] - future_df['predicted_lower']
    upper_err = future_df['predicted_upper'] - future_df['predicted_mean']
    
    # Format week labels
    week_labels = [f"{row['week_start'].strftime('%b %d')}" for _, row in future_df.iterrows()]
    
    fig = go.Figure(go.Bar(
        x=week_labels,
        y=future_df['predicted_mean'],
        marker_color=colors,
        marker_line=dict(color='white', width=2),
        error_y=dict(
            type='data', 
            symmetric=False, 
            array=upper_err, 
            arrayminus=lower_err,
            thickness=3, 
            width=10,
            color='rgba(0,0,0,0.6)'
        ),
        text=[f"{val:.2f}" for val in future_df['predicted_mean']],
        textposition='outside',
        textfont=dict(size=11, color='#1e293b'),
        hovertemplate='<b>%{x}</b><br>Mean: %{y:.2f}<br>Range: %{customdata}<extra></extra>',
        customdata=future_df['crash_range']
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Next 12 Weeks – Expected Crashes (95% Confidence Intervals)</b>",
            font=dict(size=20, color='#1e40af'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Week',
            titlefont=dict(size=14, color='#64748b'),
            tickangle=-45,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title='Expected Crashes',
            titlefont=dict(size=14, color='#64748b'),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        height=550,
        template="plotly_white",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=80, b=100, l=50, r=50)
    )
    return fig

def create_trend_analysis(weekly_df, future_df):
    """Create trend analysis comparing historical vs forecast"""
    # Calculate historical average
    hist_avg = weekly_df['total_crashes'].mean()
    
    # Combine historical and forecast
    combined_dates = pd.concat([weekly_df['week_start'], future_df['week_start']])
    combined_values = pd.concat([weekly_df['total_crashes'], future_df['predicted_mean']])
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Historical Trend vs Forecast', 'Deviation from Historical Average'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Historical trend
    fig.add_trace(
        go.Scatter(
            x=weekly_df['week_start'],
            y=weekly_df['total_crashes'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2563eb', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Forecast trend
    fig.add_trace(
        go.Scatter(
            x=future_df['week_start'],
            y=future_df['predicted_mean'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#f59e0b', width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ),
        row=1, col=1
    )
    
    # Average line
    fig.add_hline(
        y=hist_avg,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Historical Avg: {hist_avg:.2f}",
        row=1, col=1
    )
    
    # Deviation chart
    hist_dev = weekly_df['total_crashes'] - hist_avg
    forecast_dev = future_df['predicted_mean'] - hist_avg
    
    fig.add_trace(
        go.Bar(
            x=weekly_df['week_start'],
            y=hist_dev,
            name='Historical Deviation',
            marker_color='#2563eb',
            opacity=0.6
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=future_df['week_start'],
            y=forecast_dev,
            name='Forecast Deviation',
            marker_color='#f59e0b',
            opacity=0.8
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
    fig.update_layout(
        height=700,
        title_text="<b>Trend Analysis: Historical vs Forecast</b>",
        title_font=dict(size=22, color='#1e40af'),
        showlegend=True,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Crashes", row=1, col=1)
    fig.update_yaxes(title_text="Deviation", row=2, col=1)
    
    return fig

def create_risk_distribution(future_df):
    """Create risk level distribution chart"""
    risk_levels = [get_risk_level(v)[0] for v in future_df['predicted_mean']]
    risk_df = pd.DataFrame({'Risk Level': risk_levels})
    risk_counts = risk_df['Risk Level'].value_counts()
    
    # Color mapping
    color_map = {
        'Very Low Risk': '#28a745',
        'Low Risk': '#ffc107',
        'Moderate Risk': '#fd7e14',
        'High Risk': '#dc3545',
        'Very High Risk': '#c82333'
    }
    
    colors = [color_map.get(level, '#gray') for level in risk_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=colors,
            marker_line=dict(color='white', width=2),
            text=risk_counts.values,
            textposition='outside',
            textfont=dict(size=14, color='#1e293b', weight='bold')
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="<b>Risk Level Distribution (Next 12 Weeks)</b>",
            font=dict(size=20, color='#1e40af'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Risk Level',
            titlefont=dict(size=14, color='#64748b')
        ),
        yaxis=dict(
            title='Number of Weeks',
            titlefont=dict(size=14, color='#64748b'),
            gridcolor='rgba(0,0,0,0.1)'
        ),
        height=450,
        template='plotly_white',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_monthly_comparison(future_df):
    """Create monthly comparison chart"""
    monthly_data = future_df.groupby('month').agg({
        'predicted_mean': 'mean',
        'predicted_lower': 'mean',
        'predicted_upper': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_data['month'],
        y=monthly_data['predicted_mean'],
        name='Average Expected Crashes',
        marker_color='#3b82f6',
        marker_line=dict(color='white', width=2),
        text=[f"{val:.2f}" for val in monthly_data['predicted_mean']],
        textposition='outside',
        error_y=dict(
            type='data',
            symmetric=False,
            array=monthly_data['predicted_upper'] - monthly_data['predicted_mean'],
            arrayminus=monthly_data['predicted_mean'] - monthly_data['predicted_lower']
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Monthly Average Forecast Comparison</b>",
            font=dict(size=20, color='#1e40af'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Month',
            titlefont=dict(size=14, color='#64748b')
        ),
        yaxis=dict(
            title='Average Expected Crashes',
            titlefont=dict(size=14, color='#64748b'),
            gridcolor='rgba(0,0,0,0.1)'
        ),
        height=450,
        template='plotly_white',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_dual_gauges(row):
    """Enhanced gauge charts with better styling"""
    exp = int(row['most_likely_crashes'])
    prob = row['likelihood_percent']
    mean = row['predicted_mean']
    level, color = get_risk_level(mean)
    
    # Expected crashes gauge
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=exp,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Expected Crash<br><sub>Most Likely Outcome</sub>", 'font': {'size': 18}},
        delta={'reference': mean, 'position': "top", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1], 'color': '#d4edda'},
                {'range': [1, 2], 'color': '#fff3cd'},
                {'range': [2, 3], 'color': '#f8d7da'},
                {'range': [3, 5], 'color': '#f5c6cb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 4
            }
        },
        number={'suffix': " crash", 'font': {'size': 48, 'color': color}}
    ))
    fig1.update_layout(
        height=350,
        margin=dict(t=80, b=20, l=20, r=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    # Confidence level gauge
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level<br><sub>Probability of Most Likely</sub>", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2563eb", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "#fee2e2"},
                {'range': [30, 50], 'color': "#fef3c7"},
                {'range': [50, 70], 'color': "#dbeafe"},
                {'range': [70, 100], 'color': "#dcfce7"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        },
        number={'suffix': "%", 'font': {'size': 48, 'color': "#2563eb"}}
    ))
    fig2.update_layout(
        height=350,
        margin=dict(t=80, b=20, l=20, r=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig1, fig2, level, color

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    weekly_df, future_df, error = load_data()
    if error:
        st.error(f"Data loading failed: {error}")
        st.stop()

    week_options = [f"{row['week_start'].strftime('%b %d')} – {(row['week_start'] + timedelta(days=6)).strftime('%b %d, %Y')}"
                    for _, row in future_df.iterrows()]

    # =========================== SIDEBAR ===========================
    with st.sidebar:
        st.image("images/Safe_TN_Logo.png", width=200)
        
        st.markdown("### Safety Analytics & Forecasting Environment")
        
        st.title("Let's Navigate")

        page = st.radio("Navigation", ["Probablistic Crash Forecast", "Help & Guide"], label_visibility="collapsed")

        st.markdown("---")

        # ABOUT BOX – BEAUTIFUL BLUE
        st.markdown("""
        <div class="about-box">
            <b>About SAFE TN</b><br><br>
            SAFE TN (<i>Safety Analytics & Forecasting Environment for Tennessee</i>) is a probabilistic crash-risk forecasting tool developed by the 
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
            forecasting, behavioral analysis, and engineering solutions to support the <b>Tennessee Highway Safety Office</b>
            and local agencies in deploying precise, high-impact enforcement and infrastructure improvements.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("**Support**\nctiermemphis@gmail.com")

    # =========================== MAIN DASHBOARD ===========================
    if page == "Probablistic Crash Forecast":
        st.title("Traffic Crash Risk Forecast – Shelby County")
        st.markdown(f"### Probabilistic weekly prediction with uncertainty • <span style='color:#1f77b4; font-weight:bold;'>Forecast Period: {len(future_df)} weeks ahead</span>", 
                    unsafe_allow_html=True)

        # Enhanced metrics with better styling
        hist_avg = weekly_df['total_crashes'].mean()
        forecast_avg = future_df['predicted_mean'].mean()
        change_pct = ((forecast_avg - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: 
            st.metric(
                "Total Expected (Next 12 Weeks)", 
                f"{future_df['predicted_mean'].sum():.1f}",
                delta=f"vs {weekly_df['total_crashes'].sum():.0f} (last 12 weeks)"
            )
        with c2: 
            st.metric(
                "Weekly Average Forecast", 
                f"{forecast_avg:.2f}",
                delta=f"{change_pct:+.1f}% vs historical"
            )
        with c3:
            peak = future_df.loc[future_df['predicted_mean'].idxmax()]
            st.metric(
                "Peak Risk Week", 
                f"{peak['predicted_mean']:.2f}",
                delta=peak['week_start'].strftime("%b %d")
            )
        with c4:
            high_risk_weeks = sum([1 for v in future_df['predicted_mean'] if v >= 1.5])
            st.metric(
                "High Risk Weeks", 
                f"{high_risk_weeks}",
                delta=f"out of {len(future_df)} weeks"
            )

        st.markdown("---")
        
        # Main forecast plot
        st.plotly_chart(create_forecast_plot_with_intervals(weekly_df, future_df), use_container_width=True)
        
        st.markdown("---")
        
        # Additional visualizations in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Detailed Forecast", 
            "📈 Trend Analysis", 
            "🎯 Risk Distribution", 
            "📅 Monthly Comparison"
        ])
        
        with tab1:
            st.subheader("Next 12 Weeks – Detailed Forecast")
            st.plotly_chart(create_12week_bar_chart(future_df), use_container_width=True)
            
            # Summary statistics table
            st.markdown("### Forecast Summary Statistics")
            summary_data = {
                'Metric': [
                    'Mean Expected Crashes',
                    'Minimum Expected',
                    'Maximum Expected',
                    'Standard Deviation',
                    '95% CI Width (Avg)',
                    'Weeks with Risk ≥ 1.5'
                ],
                'Value': [
                    f"{future_df['predicted_mean'].mean():.2f}",
                    f"{future_df['predicted_mean'].min():.2f}",
                    f"{future_df['predicted_mean'].max():.2f}",
                    f"{future_df['predicted_mean'].std():.2f}",
                    f"{(future_df['predicted_upper'] - future_df['predicted_lower']).mean():.2f}",
                    f"{high_risk_weeks}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with tab2:
            st.subheader("Trend Analysis: Historical vs Forecast")
            st.plotly_chart(create_trend_analysis(weekly_df, future_df), use_container_width=True)
            
            # Historical comparison
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Historical Statistics")
                st.metric("Average (Historical)", f"{hist_avg:.2f}")
                st.metric("Std Dev (Historical)", f"{weekly_df['total_crashes'].std():.2f}")
                st.metric("Max (Historical)", f"{weekly_df['total_crashes'].max():.0f}")
            with col2:
                st.markdown("### Forecast Statistics")
                st.metric("Average (Forecast)", f"{forecast_avg:.2f}")
                st.metric("Std Dev (Forecast)", f"{future_df['predicted_mean'].std():.2f}")
                st.metric("Max (Forecast)", f"{future_df['predicted_mean'].max():.2f}")
        
        with tab3:
            st.subheader("Risk Level Distribution")
            st.plotly_chart(create_risk_distribution(future_df), use_container_width=True)
            
            # Risk breakdown
            risk_levels = [get_risk_level(v)[0] for v in future_df['predicted_mean']]
            risk_df = pd.DataFrame({
                'Week': [f"Week {i+1}" for i in range(len(future_df))],
                'Date': future_df['week_start'].dt.strftime('%b %d, %Y'),
                'Risk Level': risk_levels,
                'Expected Crashes': future_df['predicted_mean'].round(2)
            })
            st.markdown("### Week-by-Week Risk Breakdown")
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        with tab4:
            st.subheader("Monthly Forecast Comparison")
            st.plotly_chart(create_monthly_comparison(future_df), use_container_width=True)
            
            # Monthly details
            monthly_details = future_df.groupby('month').agg({
                'predicted_mean': ['mean', 'min', 'max', 'sum'],
                'week_start': 'count'
            }).round(2)
            monthly_details.columns = ['Avg Crashes', 'Min', 'Max', 'Total', 'Weeks']
            monthly_details = monthly_details.reset_index()
            st.markdown("### Monthly Forecast Details")
            st.dataframe(monthly_details, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # In-Depth Risk Assessment
        st.subheader("🔍 In-Depth Risk Assessment")
        selected_label = st.selectbox("Select Week for Detailed Analysis", week_options, index=0)
        sel_row = future_df.iloc[week_options.index(selected_label)]
        gauge1, gauge2, risk_level, risk_color = create_dual_gauges(sel_row)
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(gauge1, use_container_width=True)
            st.plotly_chart(gauge2, use_container_width=True)
        with col_right:
            start = sel_row['week_start'].strftime("%B %d, %Y")
            end = (sel_row['week_start'] + timedelta(days=6)).strftime("%B %d, %Y")
            
            # Calculate additional metrics
            ci_width = sel_row['predicted_upper'] - sel_row['predicted_lower']
            uncertainty_pct = (ci_width / sel_row['predicted_mean'] * 100) if sel_row['predicted_mean'] > 0 else 0
            
            st.markdown(f"""
            <div style="background-color:{risk_color}15; border-left:8px solid {risk_color}; 
                        border-radius:14px; padding:35px; height:100%; display:flex; flex-direction:column; 
                        justify-content:center; box-shadow:0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="color:{risk_color}; margin:0;">Risk Summary</h2>
                <h4 style="margin:15px 0 25px 0; color:#2c3e50; font-weight:normal;">{start} to {end}</h4>
                
                <div style="background-color:white; padding:15px; border-radius:8px; margin:10px 0;">
                    <p style="font-size:18px; margin:8px 0;"><strong>Expected Range:</strong> {sel_row['crash_range']} crashes</p>
                    <p style="font-size:16px; margin:8px 0; color:#64748b;">95% Confidence Interval: [{sel_row['predicted_lower']:.2f}, {sel_row['predicted_upper']:.2f}]</p>
                </div>
                
                <div style="background-color:white; padding:15px; border-radius:8px; margin:10px 0;">
                    <p style="font-size:18px; margin:8px 0;">
                        <strong>Most Likely:</strong> {int(sel_row['most_likely_crashes'])} crash 
                        <strong>({sel_row['likelihood_percent']:.1f}% probability)</strong>
                    </p>
                    <p style="font-size:16px; margin:8px 0; color:#64748b;">Mean Prediction: {sel_row['predicted_mean']:.2f} crashes</p>
                </div>
                
                <div style="background-color:white; padding:15px; border-radius:8px; margin:10px 0;">
                    <p style="font-size:14px; margin:8px 0; color:#64748b;">Uncertainty: {uncertainty_pct:.1f}% (CI Width: {ci_width:.2f})</p>
                </div>
                
                <h2 style="color:{risk_color}; margin:30px 0 0 0; font-weight:bold; font-size:28px;">{risk_level}</h2>
            </div>
            """, unsafe_allow_html=True)


    # ========================================================================
    # HELP & GUIDE – YOUR FINAL VERSION
    # ========================================================================
    elif page == "Help & Guide":
        st.title("Help & User Guide")
        st.markdown("### Understanding probabilistic forecasts")

        st.markdown("## Reading the Predictions")
        st.markdown("""
        **Example Forecast:** Week 3 shows "1-2 crashes (Most likely: 1, 36%)"
        
        **This means:**
        - We expect between 1-2 crashes
        - The single most likely outcome is 1 crash
        - There's a 36% probability of exactly 1 crash
        - 95% confident the actual will be within the range
        """)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **Range: 0-1 crashes**
            Very Low Risk
            - Low probability of crashes
            - Standard patrol sufficient
            """)
            st.warning("""
            **Range: 1-2 crashes**
            Moderate Risk
            - Likely 1-2 crashes
            - Increase vigilance
            """)
        with col2:
            st.info("""
            **Range: 1-2 crashes**
            Low Risk
            - Moderate probability
            - Maintain readiness
            """)
            st.error("""
            **Range: 2-3+ crashes**
            High Risk
            - Multiple crashes likely
            - Maximum enforcement
            """)

        st.markdown("---")
        st.markdown("## FAQ")

        with st.expander("Why show ranges instead of exact numbers?"):
            st.markdown("""
            Crashes are **random events**. Providing ranges:
            - Reflects real uncertainty
            - Helps with resource planning
            - Prevents false precision
            - Shows probability of different outcomes
            """)

        with st.expander("How do I use 'Most Likely' values?"):
            st.markdown("""
            The 'Most Likely' value is the **mode** of the probability distribution.
            
            **For planning:**
            - Use the **range** for resource allocation
            - Use **most likely** for scenario planning
            - Check **probability %** to gauge confidence
            """)

        with st.expander("What does 95% confidence mean?"):
            st.markdown("""
            95% confidence interval means:
            - 95 out of 100 times, the actual crashes will fall in this range
            - The range accounts for model + random uncertainty
            - Wider ranges = more uncertainty (longer forecast horizon)
            """)

        st.markdown("---")
        st.info("""
        **Technical Support:**
        - Email: ctiermemphis@gmail.com
        - Hours: Monday-Friday, 9AM-5PM
        """)

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        login_page()
    else:
        with st.sidebar:
            st.write(f"**Logged in: {st.session_state.username}**")
            if st.button("Logout", use_container_width=True):
                logout()
        main()