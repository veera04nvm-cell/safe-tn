import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import hashlib
import base64
import os

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
    page_title="SAFE TN – Crash Risk Forecast", 
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
ROUTE_SEGMENTS = {
    "I-40":  ["segment_01", "segment_02", "segment_03"],
    "I-55":  ["segment_04"],
    "I-240": ["segment_05", "segment_06", "segment_07", "segment_08", "segment_09", "segment_10", "segment_11"]
}

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
    """Create a clean historical weekly crashes plot with large fonts and borders."""
    import numpy as np
    
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
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Crashes: %{y:.0f}<extra></extra>'
        ))
        
        # Add moving average trend line
        fig.add_trace(go.Scatter(
            x=historical_df_sorted['week_start'], 
            y=moving_avg,
            mode='lines', 
            name='4-Week Moving Avg',
            line=dict(color='red', width=2, dash='dot'),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Avg: %{y:.1f}<extra></extra>'
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
            mirror=True
        ),
        margin=dict(l=80, r=40, t=100, b=110)
    )
    return fig

def create_forecast_plot(future_df):
    """Create forecast plot with upper, lower, mean - large fonts and borders."""
    import numpy as np
    
    fig = go.Figure()
    
    # Calculate confidence bounds
    se = 1.96 * np.sqrt(future_df['lambda'])
    upper = future_df['lambda'] + se
    lower = (future_df['lambda'] - se).clip(0)
    
    # Upper bound line
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=upper,
        mode='lines', 
        name='Upper Bound (95%)',
        line=dict(color='#dc3545', width=2.5, dash='dash'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Upper Bound: %{y:.2f}<extra></extra>'
    ))
    
    # Mean forecast line
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=future_df['lambda'],
        mode='lines+markers', 
        name='Mean Forecast (λ)',
        line=dict(color='#ff7f0e', width=3.5),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Mean: %{y:.2f}<extra></extra>'
    ))
    
    # Lower bound line
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], 
        y=lower,
        mode='lines', 
        name='Lower Bound (95%)',
        line=dict(color='#28a745', width=2.5, dash='dash'),
        fill='tonexty', 
        fillcolor='rgba(255,127,14,0.15)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Lower Bound: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '<b>Probabilistic Crash Forecast</b>',
            'font': {'size': 24, 'family': 'Arial', 'color': '#1f77b4'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title={
            'text': '<b>Date</b>',
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
            yanchor="bottom", 
            y=-0.20, 
            xanchor="right", 
            x=1,
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
            mirror=True
        ),
        margin=dict(l=80, r=40, t=100, b=80)
    )
    return fig

def create_dual_gauges(row):
    exp = int(row['most_likely_crashes'])
    prob = row['probability_%']
    mean = row['lambda']
    level, color = get_risk_level(mean)
    
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
        number={'suffix': " crash" if exp == 1 else " crashes", 'font': {'size': 48, 'family': 'Arial', 'color': 'black'}}
    ))
    fig1.update_layout(
        height=310, 
        margin=dict(t=100, b=10, l=20, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
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
        number={'suffix': "%", 'font': {'size': 48, 'family': 'Arial', 'color': 'black'}}
    ))
    fig2.update_layout(
        height=310, 
        margin=dict(t=100, b=10, l=20, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig1, fig2, level, color

def create_probability_pie_chart(row):
    probs = [
        row['prob_0_crash'],
        row['prob_1_crash'],
        row['prob_2_crash'],
        row['prob_3_crash'],
        row['prob_ge4_crash']
    ]
    labels = ["0 Crashes", "1 Crash", "2 Crashes", "3 Crashes", "4+ Crashes"]
    most_likely_idx = probs.index(max(probs))
    most_likely_count = [0,1,2,3,"4+"][most_likely_idx]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=probs,
        hole=0.4,
        marker=dict(
            colors=['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#991b1b'],
            line=dict(color='#ffffff', width=3)
        ),
        textinfo='label+percent',
        textposition='auto',
        textfont=dict(size=16, family='Arial', color='black'),
        hovertemplate='<b>%{label}</b><br>Probability: %{percent}<extra></extra>',
        pull=[0.1 if i == most_likely_idx else 0 for i in range(5)],
        sort=False
    )])

    fig.add_annotation(
        text=f"<b>Most Likely:</b><br>{most_likely_count} Crash{'es' if most_likely_count != 1 else ''}<br><b>{max(probs):.1f}%</b>",
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

# ============================================================================
# MAIN APP PAGES
# ============================================================================
def show_forecast_page():
    st.title("🚗 Traffic Crash Risk Forecast – Shelby County")

    # Route & Segment selector
    col1, col2 = st.columns([1,2])
    with col1:
        route = st.selectbox("Select Route", options=list(ROUTE_SEGMENTS.keys()), key="route_sel")
    with col2:
        segment = st.selectbox("Select Segment", options=ROUTE_SEGMENTS[route], key="segment_sel")

    # Load data
    future_df = load_segment_data(segment)
    if future_df is None or future_df.empty:
        st.error(f"No forecast data found for {segment}")
        st.info("Please check that the data files exist in the expected location.")
        return

    historical_df = load_historical_data(segment)

    st.markdown(f"### 🔅 Selected: **{route} ➡️ {segment.upper()}** • {len(future_df)} weeks forecast")

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
        <div style="border: 3px solid #1f77b4; border-radius: 10px; padding: 20px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        st.plotly_chart(create_historical_plot(historical_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No historical data available for this segment.")
        st.info(f"Expected file location: `outputs/risk_score/{segment}/data/{segment}_weekly_crashes.csv`")
    
    st.markdown("---")
    
    # Forecast plot with border
    st.subheader("🔅 Probabilistic Crash Forecast")
    st.markdown("""
    <div style="border: 3px solid #1f77b4; border-radius: 10px; padding: 20px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
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
        <div style="border: 3px solid #1f77b4; border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        st.plotly_chart(g1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="border: 3px solid #1f77b4; border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        st.plotly_chart(g2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with colR:
        fig_pie = create_probability_pie_chart(row)
        st.markdown("""
        <div style="border: 3px solid #1f77b4; border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # st.markdown(f"""
    # <p style="text-align:center; color:#4b5563; font-size:16px; margin-top:10px;">
    #     📅 Week: <strong>{row['week_start'].strftime('%B %d')} – {(row['week_start'] + timedelta(days=6)).strftime('%B %d, %Y')}</strong>
    # </p>
    # """, unsafe_allow_html=True)

def show_help_page():
    st.title("📘 Help & User Guide")
    st.markdown("### Understanding Probabilistic Forecasts")

    st.markdown("## 📖 Reading the Predictions")
    st.markdown("""
    **Example Forecast:** Week 3 shows "1-2 crashes (Most likely: 1, 36%)"
    
    **This means:**
    - We expect between 1-2 crashes based on statistical models
    - The single most likely outcome is exactly 1 crash
    - There's a 36% probability of exactly 1 crash occurring
    - We're 95% confident the actual count will fall within the predicted range
    """)

    st.markdown("---")
    st.markdown("## 🎯 Risk Level Interpretation")
    
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
        - 📈 Wider ranges indicate more uncertainty (often with longer forecast horizons)
        - 📈 Short-term forecasts tend to have narrower, more precise ranges
        """)

    with st.expander("How is the forecast generated?"):
        st.markdown("""
        Our forecasting system uses:
        - ⚠️ **Machine learning models** trained on historical crash data
        - ⚠️ **Weather patterns** and seasonal factors
        - ⚠️ **Traffic volume** and flow characteristics
        - ⚠️ **Temporal patterns** (day of week, time of year)
        - ⚠️ **Statistical methods** to quantify uncertainty
        """)

    st.markdown("---")
    st.markdown("## 🛠️ Technical Support")
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
            st.title("Navigation")
            
            page = st.radio(
                "Go to", 
                ["Probabilistic Crash Forecast", "Help & Guide"], 
                label_visibility="collapsed", 
                key="nav_radio"
            )
            
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

            st.write(f"**👤 Logged in as:** {st.session_state.username}")
            
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                logout()

        # Main content area
        if page == "Probabilistic Crash Forecast":
            show_forecast_page()
        elif page == "Help & Guide":
            show_help_page()