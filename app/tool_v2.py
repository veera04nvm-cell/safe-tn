"""
SAFE TN — Daily Crash Risk Prediction Dashboard
================================================
Updated from weekly to daily prediction pipeline.
- Segments identified by MSLINK (from daily_crash_prediction_pipeline.py)
- Data loaded from outputs/daily_risk_score/MSLINK_<id>/data/
- All temporal references updated from week → day
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import hashlib
import base64
import os
import numpy as np

# ============================================================================
# CONFIGURATION — match pipeline output paths
# ============================================================================
BASE_OUTPUT_DIR = 'outputs/daily_risk_score'   # pipeline writes here

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
    except Exception:
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
        logo_b64 = get_image_base64("images/Safe_TN_Logo.png")
        if logo_b64:
            st.markdown(
                f'<div class="centered-logo"><img src="data:image/png;base64,{logo_b64}" width="500"></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown("<h1 style='text-align: center;'>SAFE TN</h1>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form(key="login_form_unique", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit  = st.form_submit_button("Login", use_container_width=True)

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
            <strong>SAFE TN</strong> developed by the Center for Transportation Innovation, Education, and Research
            (C-TIER) to support transportation practitioners and enforcement agencies in proactively identifying
            and understanding roadway safety risks across Tennessee.<br>
            Activities are monitored • Unauthorized access prohibited
        </div>
        """, unsafe_allow_html=True)
        st.caption("Support: ctiermemphis@gmail.com")


def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()


# ============================================================================
# SEGMENT / MSLINK DISCOVERY
# Reads the actual output folders so no hard-coded segment list is needed.
# ============================================================================
@st.cache_data(ttl=3600)
def discover_mslinks():
    """
    Scan BASE_OUTPUT_DIR for folders named MSLINK_<id> and return a sorted
    list of MSLINK values (as strings).
    """
    mslinks = []
    if not os.path.isdir(BASE_OUTPUT_DIR):
        return mslinks
    for name in os.listdir(BASE_OUTPUT_DIR):
        if name.startswith("MSLINK_") and os.path.isdir(os.path.join(BASE_OUTPUT_DIR, name)):
            mslink_val = name[len("MSLINK_"):]
            mslinks.append(mslink_val)
    return sorted(mslinks, key=lambda x: (len(x), x))


def segment_folder(mslink):
    return os.path.join(BASE_OUTPUT_DIR, f"MSLINK_{mslink}")


# ============================================================================
# CONFIG & STYLE
# ============================================================================
st.set_page_config(
    page_title="SAFE TN – Daily Crash Risk Prediction",
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
        color:white; padding:20px; border-radius:12px;
        box-shadow:0 4px 15px rgba(0,0,0,0.2); font-size:14.5px;
        line-height:1.5; margin-bottom: 15px;
    }
    .about-box b {color:#fbbf24;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADERS
# ============================================================================
@st.cache_data(ttl=3600)
def load_segment_data(mslink):
    """Load future daily predictions for a given MSLINK."""
    folder = segment_folder(mslink)
    path = os.path.join(folder, "data", f"MSLINK_{mslink}_future_predictions_with_risk.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None


@st.cache_data(ttl=3600)
def load_historical_data(mslink):
    """Load historical daily crashes for a given MSLINK."""
    folder = segment_folder(mslink)
    path = os.path.join(folder, "data", f"MSLINK_{mslink}_daily_crashes.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame(columns=['date', 'crash_count'])


@st.cache_data(ttl=3600)
def load_crash_shapefile_data():
    """Load raw crash-level data for hotspot maps (2021 onwards)."""
    try:
        file_path = "data/Unlocked_Segmented_2022_2025_Crashes_Final.csv"
        usecols = [
            'GPS Coordi', 'GPS Coor_1', 'MSLINK', 'Type of Cr',
            'Year Of Cr', 'Total Kill', 'Total Inj',
            'Hit and Ru', 'RTE_NME', 'CNTY_SEAT', 'Date of Cr', 'Light Cond'
        ]
        df = pd.read_csv(
            file_path, usecols=usecols,
            dtype={
                'MSLINK': 'category', 'Type of Cr': 'category',
                'RTE_NME': 'category', 'CNTY_SEAT': 'category',
                'Hit and Ru': 'category', 'Light Cond': 'category'
            },
            low_memory=False
        )
        df = df.rename(columns={
            'GPS Coordi': 'latitude', 'GPS Coor_1': 'longitude',
            'MSLINK': 'segment_id', 'Type of Cr': 'severity',
            'Year Of Cr': 'year', 'Total Kill': 'fatalities',
            'Total Inj': 'injuries', 'Hit and Ru': 'hit_and_run',
            'RTE_NME': 'route', 'CNTY_SEAT': 'city',
            'Date of Cr': 'Date of Crash', 'Light Cond': 'Light Condition'
        })
        df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce').astype('float32')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').astype('float32')
        df['year']       = pd.to_numeric(df['year'],      errors='coerce').astype('int16')
        df['fatalities'] = pd.to_numeric(df['fatalities'],errors='coerce').fillna(0).astype('int8')
        df['injuries']   = pd.to_numeric(df['injuries'],  errors='coerce').fillna(0).astype('int8')
        df = df[df['year'] >= 2021]
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude']  >= 34.9) & (df['latitude']  <= 36.7)]
        df = df[(df['longitude'] >= -90.5) & (df['longitude'] <= -88.0)]
        return df
    except Exception as e:
        st.error(f"Error loading crash data: {e}")
        return None


@st.cache_data(ttl=3600)
def load_segmented_data():
    """Load segmented interstate data for background analysis."""
    try:
        df = pd.read_csv("data/Segmented_Shelby_Interstates.csv")
        return df
    except Exception as e:
        print(f"Error loading segmented data: {e}")
        return None


# ============================================================================
# HELPER — RISK LEVEL (daily thresholds from pipeline)
# ============================================================================
def get_risk_level(lam):
    """Match risk thresholds from forecast_future() in the pipeline."""
    if lam >= 1.0:
        return "High Risk",      "#dc3545"
    elif lam >= 0.5:
        return "Medium Risk",    "#fd7e14"
    elif lam >= 0.2:
        return "Low Risk",       "#ffc107"
    else:
        return "Very Low Risk",  "#28a745"


def get_most_likely_from_probabilities(row):
    probs = [
        row['prob_0_crash'], row['prob_1_crash'], row['prob_2_crash'],
        row['prob_3_crash'], row['prob_ge4_crash']
    ]
    most_likely_idx = probs.index(max(probs))
    crash_counts    = [0, 1, 2, 3, "4+"]
    return crash_counts[most_likely_idx], probs[most_likely_idx]


# ============================================================================
# PLOTS — FORECAST PAGE
# ============================================================================
def create_historical_plot(historical_df):
    """Historical daily crashes with 7-day moving average."""
    fig = go.Figure()
    if not historical_df.empty:
        hist = historical_df.sort_values('date')
        ma   = hist['crash_count'].rolling(window=7, center=True).mean()

        fig.add_trace(go.Scatter(
            x=hist['date'], y=hist['crash_count'],
            mode='lines', name='Daily Crashes',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Crashes: %{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=hist['date'], y=ma,
            mode='lines', name='7-Day Moving Avg',
            line=dict(color='red', width=2, dash='dot'),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Avg: %{y:,.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text='<b>Historical Daily Crashes</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'),
                   x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Date</b>',           font=dict(size=20, family='Arial', color='black')),
        yaxis_title=dict(text='<b>Daily Crash Count</b>', font=dict(size=20, family='Arial', color='black')),
        hovermode='x unified', height=500, template='plotly_white',
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5,
                    font=dict(size=16, family='Arial', color='black')),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)',
                   tickfont=dict(size=14, family='Arial', color='black'),
                   linecolor='black', linewidth=2, mirror=True),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)',
                   tickfont=dict(size=16, family='Arial', color='black'),
                   linecolor='black', linewidth=2, mirror=True, separatethousands=True),
        margin=dict(l=80, r=40, t=100, b=110), paper_bgcolor='white', plot_bgcolor='white'
    )
    return fig


def create_forecast_plot(future_df):
    """Daily probabilistic crash forecast with 95 % CI."""
    fig = go.Figure()

    # Use pre-computed CI columns when available; fall back to Poisson approx
    if 'predicted_lower' in future_df.columns and 'predicted_upper' in future_df.columns:
        lower = future_df['predicted_lower']
        upper = future_df['predicted_upper']
    else:
        se    = 1.96 * np.sqrt(future_df['lambda'].clip(lower=0))
        lower = (future_df['lambda'] - se).clip(lower=0)
        upper =  future_df['lambda'] + se

    fig.add_trace(go.Scatter(
        x=future_df['date'], y=lower, mode='lines', name='Lower Bound (95%)',
        line=dict(color='#28a745', width=2, dash='dash'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Lower: %{y:,.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=future_df['date'], y=upper, mode='lines', name='Upper Bound (95%)',
        line=dict(color='#dc3545', width=2, dash='dash'),
        fill='tonexty', fillcolor='rgba(255,127,14,0.15)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Upper: %{y:,.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=future_df['date'], y=future_df['lambda'],
        mode='lines+markers', name='Mean Prediction (λ)',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=5, color='#ff7f0e', line=dict(color='black', width=0.8)),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>λ: %{y:,.3f}<extra></extra>'
    ))

    # Colour-code risk level on x-axis via shapes
    risk_colors = {'Very Low': '#28a745', 'Low': '#ffc107', 'Medium': '#fd7e14', 'High': '#dc3545'}
    if 'risk_level' in future_df.columns:
        for _, row in future_df.iterrows():
            rc = risk_colors.get(row['risk_level'], '#1f77b4')
            fig.add_shape(type='rect',
                x0=row['date'] - pd.Timedelta(hours=12),
                x1=row['date'] + pd.Timedelta(hours=12),
                y0=0, y1=0.04, yref='paper',
                fillcolor=rc, opacity=0.6, line_width=0)

    fig.update_layout(
        title=dict(text='<b>Daily Probabilistic Crash Forecast</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'),
                   x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Date</b>',                     font=dict(size=20, family='Arial', color='black')),
        yaxis_title=dict(text='<b>Expected Daily Crashes (λ)</b>', font=dict(size=20, family='Arial', color='black')),
        hovermode='x unified', height=500, template='plotly_white',
        legend=dict(orientation="h", yanchor="top", y=-0.30, xanchor="center", x=0.5,
                    font=dict(size=14, family='Arial', color='black')),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)',
                   tickfont=dict(size=13, family='Arial', color='black'),
                   linecolor='black', linewidth=2, mirror=True),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)',
                   tickfont=dict(size=16, family='Arial', color='black'),
                   linecolor='black', linewidth=2, mirror=True, separatethousands=True),
        margin=dict(l=80, r=40, t=100, b=140), paper_bgcolor='white', plot_bgcolor='white'
    )
    return fig


def create_dual_gauges(row):
    most_likely_crashes, prob_percent = get_most_likely_from_probabilities(row)
    exp    = 4 if most_likely_crashes == "4+" else int(most_likely_crashes)
    lam    = row['lambda']
    level, color = get_risk_level(lam)

    # Gauge 1 — Most Likely Outcome
    suffix = (" crash" if exp == 1 else " crashes")
    prefix = "≥" if most_likely_crashes == "4+" else ""
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number", value=exp,
        title={'text': "<b>Most Likely Outcome</b>", 'font': {'size': 22, 'family': 'Arial', 'color': 'black'}},
        gauge={
            'axis': {'range': [0, 5], 'tickfont': {'size': 16, 'color': 'black'}},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 1], 'color': '#d4edda'}, {'range': [1, 2], 'color': '#fff3cd'},
                {'range': [2, 3], 'color': '#f8d7da'}, {'range': [3, 5], 'color': '#f5c6cb'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': exp}
        },
        number={'suffix': suffix, 'prefix': prefix, 'font': {'size': 28, 'family': 'Arial', 'color': 'black'}}
    ))
    fig1.update_layout(height=310, margin=dict(t=100, b=10, l=20, r=20),
                       paper_bgcolor='white', plot_bgcolor='white')

    # Gauge 2 — Probability of Most Likely
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number", value=prob_percent,
        title={'text': "<b>Probability of Most Likely</b>", 'font': {'size': 22, 'family': 'Arial', 'color': 'black'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'size': 16, 'color': 'black'}},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 25],  'color': '#fee2e2'}, {'range': [25, 50], 'color': '#fef3c7'},
                {'range': [50, 75], 'color': '#d1fae5'}, {'range': [75, 100],'color': '#a7f3d0'}
            ]
        },
        number={'suffix': "%", 'font': {'size': 28, 'family': 'Arial', 'color': 'black'}}
    ))
    fig2.update_layout(height=310, margin=dict(t=100, b=10, l=20, r=20),
                       paper_bgcolor='white', plot_bgcolor='white')

    return fig1, fig2, level, color


def create_probability_pie_chart(row):
    probs = [row['prob_0_crash'], row['prob_1_crash'], row['prob_2_crash'],
             row['prob_3_crash'], row['prob_ge4_crash']]
    labels = ["0 Crashes", "1 Crash", "2 Crashes", "3 Crashes", "4+ Crashes"]
    most_likely_crashes, prob_percent = get_most_likely_from_probabilities(row)
    most_likely_idx = probs.index(prob_percent)

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=probs, hole=0.4,
        marker=dict(colors=['#10b981','#3b82f6','#f59e0b','#ef4444','#991b1b'],
                    line=dict(color='#ffffff', width=3)),
        textinfo='label+percent', textposition='auto',
        textfont=dict(size=16, family='Arial', color='black'),
        hovertemplate='<b>%{label}</b><br>Probability: %{value:.1f}%<extra></extra>',
        pull=[0.1 if i == most_likely_idx else 0 for i in range(5)],
        sort=False
    )])

    crash_label = (f"{most_likely_crashes} Crash"
                   if most_likely_crashes == 1
                   else f"{most_likely_crashes} Crashes")
    fig.add_annotation(
        text=f"<b>Most Likely:</b><br>{crash_label}<br><b>{prob_percent:.1f}%</b>",
        x=0.5, y=0.5, font=dict(size=20, color="white", family="Arial"),
        showarrow=False, bgcolor="#1f2937", bordercolor="#ffffff",
        borderwidth=2, borderpad=10, opacity=0.95
    )
    fig.update_layout(
        title=dict(text='<b>Probability Distribution of Daily Crash Counts</b>',
                   font=dict(size=22, family='Arial', color='#1f77b4'),
                   x=0.5, xanchor='center'),
        height=620, showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
                    font=dict(size=16, family='Arial', color='black')),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family="Arial", size=16, color='black'),
        margin=dict(l=20, r=140, t=80, b=20)
    )
    return fig


def create_risk_calendar_heatmap(future_df):
    """
    Calendar-style heatmap: x = day-of-week, y = week number,
    coloured by λ (expected crashes).
    """
    df = future_df.copy()
    df['dow']      = df['date'].dt.dayofweek          # 0=Mon … 6=Sun
    df['week_num'] = ((df['date'] - df['date'].min()).dt.days // 7)

    pivot = df.pivot(index='week_num', columns='dow', values='lambda')

    day_names  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    week_labels = []
    for wn in pivot.index:
        # first date of that week
        first = df[df['week_num'] == wn]['date'].min()
        week_labels.append(first.strftime('Wk %b %d'))

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[day_names[i] for i in pivot.columns],
        y=week_labels,
        colorscale='RdYlGn_r',
        zmin=0,
        colorbar=dict(title='<b>λ</b>', thickness=15),
        hovertemplate='<b>%{y} — %{x}</b><br>λ: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='<b>Daily Risk Calendar</b>',
                   font=dict(size=22, family='Arial', color='#1f77b4'),
                   x=0.5, xanchor='center'),
        xaxis=dict(title='<b>Day of Week</b>', tickfont=dict(size=14),
                   linecolor='black', linewidth=2, mirror=True),
        yaxis=dict(title='<b>Week</b>', tickfont=dict(size=13), autorange='reversed',
                   linecolor='black', linewidth=2, mirror=True),
        height=max(300, 60 * len(week_labels)),
        margin=dict(l=100, r=40, t=80, b=60),
        paper_bgcolor='white', plot_bgcolor='white'
    )
    return fig


# ============================================================================
# PLOTS — HOTSPOT PAGE  (unchanged from original, works on raw crash data)
# ============================================================================
def create_monthly_crashes_plot(df, selected_year=None, selected_route=None, selected_segment=None):
    filtered_df = df.copy()
    if selected_year and selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['Route'] == selected_route]
    if selected_segment and selected_segment != "All Segments":
        filtered_df = filtered_df[filtered_df['Segment ID'] == selected_segment]

    filtered_df['Crash Date'] = pd.to_datetime(filtered_df['Date of Crash'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['Crash Date'])
    filtered_df['Month_Num'] = filtered_df['Crash Date'].dt.month

    monthly_crashes = filtered_df.groupby('Month_Num').size().reset_index(name='Crashes')
    all_months = pd.DataFrame({'Month_Num': range(1, 13)})
    monthly_crashes = all_months.merge(monthly_crashes, on='Month_Num', how='left').fillna(0)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_crashes['Month_Name'] = monthly_crashes['Month_Num'].apply(lambda x: month_names[x - 1])

    max_crashes = monthly_crashes['Crashes'].max()
    monthly_crashes['Dot_Size'] = 20 + (monthly_crashes['Crashes'] / max(max_crashes, 1)) * 60
    monthly_crashes['Color'] = monthly_crashes['Crashes'].apply(
        lambda x: '#034fa0' if x > max_crashes * 0.75 else
                  '#0078d4' if x > max_crashes * 0.50 else
                  '#249ee4' if x > max_crashes * 0.25 else '#54daff'
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_crashes['Month_Name'], y=monthly_crashes['Crashes'],
        mode='markers+text',
        marker=dict(size=monthly_crashes['Dot_Size'], color=monthly_crashes['Color'],
                    line=dict(color='black', width=0.15), opacity=0.8),
        text=[f"<b>{int(c):,}</b>" for c in monthly_crashes['Crashes']],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Total Crashes: %{y:,.0f}<extra></extra>',
        name='Crashes'
    ))
    fig.add_shape(type="line", x0=-0.5, x1=11.5, y0=0, y1=0,
                  line=dict(color="black", width=2))
    fig.update_layout(
        title=dict(text='<b>✨ Monthly Crash Variation</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Month</b>', font=dict(size=20, family='Arial', color='black')),
        yaxis_title=dict(text='<b>Total Crashes</b>', font=dict(size=20, family='Arial', color='black')),
        height=650, template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=16, family='Arial Black', color='black'),
                   linecolor='black', linewidth=2, mirror=True),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)',
                   tickfont=dict(size=16, family='Arial', color='black'),
                   linecolor='black', linewidth=2, mirror=True,
                   zeroline=True, zerolinewidth=2, zerolinecolor='black', separatethousands=True),
        margin=dict(l=80, r=40, t=100, b=80)
    )
    return fig


def create_day_night_crashes_plot(df, selected_year=None, selected_route=None):
    filtered_df = df.copy()
    if selected_year and selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['Route'] == selected_route]

    light_col = next((c for c in filtered_df.columns if 'light' in c.lower()), None)
    if light_col is None:
        fig = go.Figure()
        fig.add_annotation(text="Light Condition data not available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=20, color="red"))
        return fig

    day_night_data = filtered_df.groupby(['Segment ID', light_col]).size().reset_index(name='Crashes')
    pivot_data     = day_night_data.pivot(index='Segment ID', columns=light_col,
                                          values='Crashes').fillna(0)
    dark_cols     = [c for c in pivot_data.columns if any(t in str(c).lower() for t in ['dark', 'night'])]
    daylight_cols = [c for c in pivot_data.columns if c not in dark_cols]

    day_crashes   = pivot_data[daylight_cols].sum(axis=1) if daylight_cols else pd.Series(0, index=pivot_data.index)
    night_crashes = pivot_data[dark_cols].sum(axis=1)     if dark_cols     else pd.Series(0, index=pivot_data.index)

    total_crashes = day_crashes + night_crashes
    sort_idx      = total_crashes.sort_values(ascending=False).index
    day_crashes   = day_crashes.loc[sort_idx].iloc[:10]
    night_crashes = night_crashes.loc[sort_idx].iloc[:10]
    segments      = [str(s) for s in sort_idx[:10]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=segments, x=day_crashes.values, name='Daytime', orientation='h',
        marker=dict(color='#ffa500', line=dict(color='black', width=1.2)),
        text=[f"{int(v):,}" for v in day_crashes.values], textposition='outside',
        textfont=dict(size=13, color='black', family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Daytime Crashes: %{x:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        y=segments, x=night_crashes.values, name='Nighttime', orientation='h',
        marker=dict(color='#191970', line=dict(color='black', width=1.2)),
        text=[f"{int(v):,}" for v in night_crashes.values], textposition='outside',
        textfont=dict(size=13, color='black', family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Nighttime Crashes: %{x:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='<b>✨ Top 10 Segments: Day vs Night Crash Comparison</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Number of Crashes</b>', font=dict(size=20, family='Arial', color='black')),
        yaxis_title=dict(text='<b>MSLINK (Ranked by Total Crashes)</b>', font=dict(size=20, family='Arial', color='black')),
        height=650, template='plotly_white', paper_bgcolor='white', plot_bgcolor='white',
        barmode='group', bargap=0.15, bargroupgap=0.05,
        legend=dict(orientation="h", yanchor="top", y=0.98, xanchor="center", x=1.15,
                    font=dict(size=16, family='Arial', color='black')),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)',
                   tickfont=dict(size=15, family='Arial', color='black'),
                   linecolor='black', linewidth=2, mirror=True, separatethousands=True),
        yaxis=dict(showgrid=False, tickfont=dict(size=14, family='Arial Black', color='black'),
                   linecolor='black', linewidth=2, mirror=True, autorange='reversed'),
        margin=dict(l=100, r=120, t=80, b=60)
    )
    return fig


def create_segment_ranking_plots(df, selected_year=None, selected_route=None):
    filtered_df = df.copy()
    if selected_year and selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['Route'] == selected_route]

    # — Plot 1: Top 10 by Total Crashes —
    crash_counts = filtered_df.groupby(['Segment ID', 'Route']).size().reset_index(name='Total Crashes')
    crash_counts = crash_counts.sort_values('Total Crashes', ascending=False).head(10)
    crash_counts['Rank'] = range(1, len(crash_counts) + 1)
    n = len(crash_counts)
    blue_gradient = [f'rgb({int(0+(173-0)*i/max(n-1,1))},{int(51+(216-51)*i/max(n-1,1))},{int(153+(230-153)*i/max(n-1,1))})' for i in range(n)]
    crash_counts['Color'] = blue_gradient

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=crash_counts['Rank'], y=crash_counts['Total Crashes'],
        marker=dict(color=crash_counts['Color'], line=dict(color='black', width=2)),
        width=0.6,
        text=[f"<b>{seg}</b><br>{count:,} crashes" for seg, count in zip(crash_counts['Segment ID'], crash_counts['Total Crashes'])],
        textposition='outside', textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>Rank %{x}</b><br>MSLINK: %{customdata[0]}<br>Route: %{customdata[1]}<br>Crashes: %{y:,.0f}<extra></extra>',
        customdata=crash_counts[['Segment ID', 'Route']].values
    ))
    fig1.update_layout(
        title=dict(text='<b>✨ Top 10 Segments by Total Crashes</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Rank</b>',               font=dict(size=19, family='Arial', color='black')),
        yaxis_title=dict(text='<b>Number of Crashes</b>', font=dict(size=19, family='Arial', color='black')),
        height=650, template='plotly_white', showlegend=False,
        paper_bgcolor='white', plot_bgcolor='#f8f9fa',
        xaxis=dict(showgrid=False, tickfont=dict(size=16, family='Arial Black', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   tickmode='linear', tick0=1, dtick=1, range=[0.5, 10.5]),
        yaxis=dict(showgrid=True, gridwidth=1.5, gridcolor='rgba(200,200,200,0.4)',
                   tickfont=dict(size=16, family='Arial', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   zeroline=True, zerolinewidth=2, zerolinecolor='black', separatethousands=True),
        margin=dict(l=80, r=40, t=90, b=80)
    )

    # — Plot 2: Top 10 by Hit & Run —
    hit_run_df    = filtered_df[filtered_df['Hit and Run'] == 'Yes'].copy()
    hit_run_counts = hit_run_df.groupby(['Segment ID', 'Route']).size().reset_index(name='Hit and Run Cases')
    hit_run_counts = hit_run_counts.sort_values('Hit and Run Cases', ascending=False).head(10)
    hit_run_counts['Rank'] = range(1, len(hit_run_counts) + 1)
    nr = len(hit_run_counts)
    red_gradient = [f'rgb({int(139+(255-139)*i/max(nr-1,1))},{int(0+(182-0)*i/max(nr-1,1))},{int(0+(193-0)*i/max(nr-1,1))})' for i in range(nr)]
    hit_run_counts['Color'] = red_gradient

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=hit_run_counts['Rank'], y=hit_run_counts['Hit and Run Cases'],
        marker=dict(color=hit_run_counts['Color'], line=dict(color='black', width=2)),
        width=0.6,
        text=[f"<b>{seg}</b><br>{count:,} cases" for seg, count in zip(hit_run_counts['Segment ID'], hit_run_counts['Hit and Run Cases'])],
        textposition='outside', textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>Rank %{x}</b><br>MSLINK: %{customdata[0]}<br>Route: %{customdata[1]}<br>Hit & Run: %{y:,.0f}<extra></extra>',
        customdata=hit_run_counts[['Segment ID', 'Route']].values
    ))
    fig2.update_layout(
        title=dict(text='<b>✨ Top 10 Segments by Hit and Run Cases</b>',
                   font=dict(size=24, family='Arial', color='#dc3545'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Rank</b>',                       font=dict(size=19, family='Arial', color='black')),
        yaxis_title=dict(text='<b>Number of Hit and Run Cases</b>', font=dict(size=19, family='Arial', color='black')),
        height=650, template='plotly_white', showlegend=False,
        paper_bgcolor='white', plot_bgcolor='#f8f9fa',
        xaxis=dict(showgrid=False, tickfont=dict(size=16, family='Arial Black', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   tickmode='linear', tick0=1, dtick=1, range=[0.5, 10.5]),
        yaxis=dict(showgrid=True, gridwidth=1.5, gridcolor='rgba(200,200,200,0.4)',
                   tickfont=dict(size=16, family='Arial', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   zeroline=True, zerolinewidth=2, zerolinecolor='black', separatethousands=True),
        margin=dict(l=80, r=40, t=90, b=80)
    )
    return fig1, fig2


def create_fatality_ranking_plot(df, selected_year=None, selected_route=None):
    filtered_df = df.copy()
    if selected_year and selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['Route'] == selected_route]

    fat_col = next((c for c in ['fatalities', 'Total Kill'] if c in filtered_df.columns), None)
    if fat_col is None:
        return go.Figure()
    fatality_counts = filtered_df.groupby(['Segment ID', 'Route'])[fat_col].sum().reset_index()
    fatality_counts = fatality_counts.rename(columns={fat_col: 'Total_Fatalities'})
    fatality_counts = fatality_counts.sort_values('Total_Fatalities', ascending=False).head(10)
    fatality_counts = fatality_counts[fatality_counts['Total_Fatalities'] > 0]
    fatality_counts['Rank'] = range(1, len(fatality_counts) + 1)
    nf = len(fatality_counts)
    red_gradient = [f'rgb({int(139+(255-139)*i/max(nf-1,1))},{int(0+(182-0)*i/max(nf-1,1))},{int(0+(193-0)*i/max(nf-1,1))})' for i in range(nf)]
    fatality_counts['Color'] = red_gradient

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fatality_counts['Rank'], y=fatality_counts['Total_Fatalities'],
        marker=dict(color=fatality_counts['Color'], line=dict(color='black', width=2)),
        width=0.6,
        text=[f"<b>{seg}</b><br>{int(c):,} fatalities" for seg, c in zip(fatality_counts['Segment ID'], fatality_counts['Total_Fatalities'])],
        textposition='outside', textfont=dict(size=14, color='black', family='Arial Black'),
        hovertemplate='<b>Rank %{x}</b><br>MSLINK: %{customdata[0]}<br>Route: %{customdata[1]}<br>Fatalities: %{y:,.0f}<extra></extra>',
        customdata=fatality_counts[['Segment ID', 'Route']].values
    ))
    fig.update_layout(
        title=dict(text='<b>✨ Top 10 Segments by Fatalities</b>',
                   font=dict(size=24, family='Arial', color='#dc3545'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Rank</b>',                   font=dict(size=19, family='Arial', color='black')),
        yaxis_title=dict(text='<b>Number of Fatalities</b>', font=dict(size=19, family='Arial', color='black')),
        height=650, template='plotly_white', showlegend=False,
        paper_bgcolor='white', plot_bgcolor='#f8f9fa',
        xaxis=dict(showgrid=False, tickfont=dict(size=16, family='Arial Black', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   tickmode='linear', tick0=1, dtick=1, range=[0.5, 10.5]),
        yaxis=dict(showgrid=True, gridwidth=1.5, gridcolor='rgba(200,200,200,0.4)',
                   tickfont=dict(size=16, family='Arial', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   zeroline=True, zerolinewidth=2, zerolinecolor='black', separatethousands=True),
        margin=dict(l=80, r=40, t=90, b=80)
    )
    return fig


# ============================================================================
# HOTSPOT MAP PLOTS (unchanged logic, same as original)
# ============================================================================
def create_crash_frequency_heatmap(df, selected_year=None, selected_route=None):
    filtered_df = df.copy()
    if selected_year and selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
    if selected_route and selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['route'] == selected_route]

    st.caption(f"📊 Displaying {len(filtered_df):,} crashes")
    filtered_df['hover_text'] = (
        '<b>MSLINK:</b> ' + filtered_df['segment_id'].astype(str) + '<br>' +
        '<b>City:</b> ' + filtered_df['city'].astype(str) + '<br>' +
        '<b>Route:</b> ' + filtered_df['route'].astype(str) + '<br>' +
        '<b>Injuries:</b> ' + filtered_df['injuries'].astype(str) + '<br>' +
        '<b>Fatalities:</b> ' + filtered_df['fatalities'].astype(str) + '<br>' +
        '<b>Year:</b> ' + filtered_df['year'].astype(str)
    )
    fig = go.Figure()
    fig.add_trace(go.Densitymapbox(
        lat=filtered_df['latitude'], lon=filtered_df['longitude'],
        radius=15, colorscale='Reds', showscale=True,
        hoverinfo='skip', opacity=0.6,
        colorbar=dict(title="<b>Density</b>", thickness=15, len=0.7)
    ))
    fig.add_trace(go.Scattermapbox(
        lat=filtered_df['latitude'], lon=filtered_df['longitude'],
        mode='markers',
        marker=dict(size=4, color='rgba(255, 0, 0, 0.2)', opacity=0.3),
        text=filtered_df['hover_text'],
        hovertemplate='%{text}<extra></extra>', showlegend=False
    ))
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=35.15, lon=-90.05), zoom=9),
        title=dict(text='<b>✨ Crash Frequency Heatmap</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        height=700, margin=dict(l=0, r=0, t=60, b=0), paper_bgcolor='white', hovermode='closest'
    )
    return fig


def create_severity_scatter_map(df, selected_year=None, selected_route=None):
    filtered_df = df.copy()
    if selected_year and selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
    if selected_route and selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['route'] == selected_route]

    severity_colors = {
        'Fatal Injury': '#96092B', 'Incapacitating Injury': '#FF4500',
        'Non-Incapacitating Injury': '#FFD700', 'Possible Injury': '#32CD32',
        'Property Damage Only': '#1E90FF'
    }
    filtered_df['marker_size'] = 10
    fig = px.scatter_mapbox(
        filtered_df, lat='latitude', lon='longitude',
        color='severity', color_discrete_map=severity_colors,
        size='marker_size', size_max=10,
        hover_name='segment_id',
        hover_data={'latitude': False, 'longitude': False, 'route': True, 'city': True,
                    'severity': True, 'fatalities': True, 'injuries': True,
                    'year': True, 'marker_size': False},
        center=dict(lat=35.15, lon=-90.05), zoom=9,
        mapbox_style="open-street-map", height=700, opacity=0.85
    )
    fig.update_layout(
        title=dict(text='<b>✨ Crash Severity Map</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        margin=dict(l=0, r=0, t=60, b=0), paper_bgcolor='white',
        legend=dict(title=dict(text='<b>Severity Type</b>', font=dict(size=14)),
                    orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=1)
    )
    return fig


def create_segment_hotspot_map(df, selected_year=None, selected_route=None, top_n=10):
    filtered_df = df.copy()
    if selected_year and selected_year != "All Years":
        filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
    if selected_route and selected_route != "All Routes":
        filtered_df = filtered_df[filtered_df['route'] == selected_route]

    segment_stats = filtered_df.groupby('segment_id').agg(
        latitude=('latitude', 'mean'), longitude=('longitude', 'mean'),
        route=('route', 'first'), city=('city', 'first'),
        total_crashes=('severity', 'count'),
        fatalities=('fatalities', 'sum'), injuries=('injuries', 'sum')
    ).reset_index()
    top_segments = segment_stats.nlargest(top_n, 'total_crashes').copy()
    top_segments['marker_size'] = 15 + (top_segments['total_crashes'] / top_segments['total_crashes'].max()) * 35
    top_segments['rank'] = range(1, len(top_segments) + 1)

    fig = px.scatter_mapbox(
        top_segments, lat='latitude', lon='longitude',
        size='marker_size', color='total_crashes', color_continuous_scale='Reds',
        hover_name='segment_id',
        hover_data={'latitude': False, 'longitude': False, 'route': True, 'city': True,
                    'total_crashes': True, 'fatalities': True, 'injuries': True,
                    'marker_size': False, 'rank': True},
        center=dict(lat=35.15, lon=-90.05), zoom=9,
        mapbox_style="open-street-map", height=700
    )
    for _, row in top_segments.head(5).iterrows():
        fig.add_trace(go.Scattermapbox(
            lat=[row['latitude']], lon=[row['longitude']], mode='text',
            text=[f"#{row['rank']}"],
            textfont=dict(size=14, color='white', family='Arial Black'),
            showlegend=False, hoverinfo='skip'
        ))
    fig.update_layout(
        title=dict(text=f'<b>✨ Top {top_n} Crash Hotspot Segments</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        margin=dict(l=0, r=0, t=60, b=0), paper_bgcolor='white',
        coloraxis_colorbar=dict(title="<b>Total<br>Crashes</b>", thickness=15, len=0.7)
    )
    return fig


# ============================================================================
# PAGE 1 — INTERACTIVE CRASH MAPS
# ============================================================================
def show_hotspot_maps_page():
    st.title("🔅 Interactive Crash Hotspot Maps")
    st.markdown("### Visualizing High-Risk Segments Across Shelby County")

    if 'crash_data_loaded' not in st.session_state:
        st.session_state.crash_data_loaded = False

    if not st.session_state.crash_data_loaded:
        st.info("🔅 Click the button below to load crash data and begin analysis")
        if st.button("🔅 Load Crash Data", use_container_width=True):
            with st.spinner("Loading crash data…"):
                st.session_state.crash_df = load_crash_shapefile_data()
                st.session_state.crash_data_loaded = True
                st.rerun()
        return

    crash_df = st.session_state.crash_df
    if crash_df is None or crash_df.empty:
        st.error("Unable to load crash data.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🔅 Total Crashes",   f"{len(crash_df):,}")
    with c2: st.metric("🔅 Total Fatalities", f"{int(crash_df['fatalities'].sum()):,}")
    with c3: st.metric("🔅 Total Injuries",   f"{int(crash_df['injuries'].sum()):,}")
    with c4: st.metric("🔅 Unique Segments",   f"{crash_df['segment_id'].nunique():,}")

    st.markdown("---")
    st.markdown("### 🔎 Filter Options")
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        years = ["All Years"] + sorted(crash_df['year'].dropna().unique().tolist(), reverse=True)
        selected_year  = st.selectbox("Select Year",  years, key="hotspot_year")
    with cf2:
        routes = ["All Routes"] + sorted(crash_df['route'].dropna().unique().tolist())
        selected_route = st.selectbox("Select Route", routes, key="hotspot_route")
    with cf3:
        top_n = st.slider("Top N Segments to Display", 5, 20, 10, 1)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔅 Frequency Heatmap", "🔅 Severity Map", "🔅 Top Hotspots"])

    with tab1:
        st.markdown("#### 🔅 Crash Density Heatmap")
        st.markdown("*Red zones indicate areas with the highest concentration of crashes.*")
        st.plotly_chart(create_crash_frequency_heatmap(crash_df, selected_year, selected_route),
                        use_container_width=True)

    with tab2:
        st.markdown("#### 🔅 Crash Severity Distribution")
        st.markdown("*Each point represents a crash, coloured by severity level.*")
        st.plotly_chart(create_severity_scatter_map(crash_df, selected_year, selected_route),
                        use_container_width=True)

    with tab3:
        st.markdown(f"#### 🔅 Top {top_n} Highest-Risk Segments")
        st.markdown("*Segments ranked by total crash frequency (larger circles = more crashes)*")
        st.plotly_chart(create_segment_hotspot_map(crash_df, selected_year, selected_route, top_n),
                        use_container_width=True)

        filtered_df = crash_df.copy()
        if selected_year  != "All Years":  filtered_df = filtered_df[filtered_df['year']  == int(selected_year)]
        if selected_route != "All Routes": filtered_df = filtered_df[filtered_df['route'] == selected_route]
        seg_stats = filtered_df.groupby('segment_id').agg(
            total_crashes=('severity', 'count'),
            fatalities=('fatalities', 'sum'),
            injuries=('injuries', 'sum')
        ).reset_index().nlargest(top_n, 'total_crashes')
        seg_stats['rank'] = range(1, len(seg_stats) + 1)
        seg_stats = seg_stats.rename(columns={'segment_id': 'MSLINK'})[
            ['rank', 'MSLINK', 'total_crashes', 'fatalities', 'injuries']]
        st.markdown("##### 📋 Top Segments Summary Table")
        st.dataframe(seg_stats.style.format(
            {'total_crashes': '{:,.0f}', 'fatalities': '{:,.0f}', 'injuries': '{:,.0f}'}),
            use_container_width=True, height=400)

    st.markdown("---")
    st.markdown("### 🔅 Historical Background of Crash Data Analysis")

    analysis_df = crash_df.rename(columns={
        'year': 'Year Of Crash', 'route': 'Route',
        'segment_id': 'Segment ID', 'hit_and_run': 'Hit and Run',
        'fatalities': 'fatalities'
    })

    st.markdown("#### 🔅 Monthly Crash Variation")
    st.plotly_chart(create_monthly_crashes_plot(analysis_df, selected_year, selected_route),
                    use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔅 Day vs Night Crash Comparison")
    st.plotly_chart(create_day_night_crashes_plot(analysis_df, selected_year, selected_route),
                    use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔅 Top 10 MSLINK Rankings")
    fig_crash, _ = create_segment_ranking_plots(analysis_df, selected_year, selected_route)
    fig_fat      = create_fatality_ranking_plot(analysis_df,  selected_year, selected_route)
    col_p1, col_p2 = st.columns(2)
    with col_p1: st.plotly_chart(fig_crash, use_container_width=True)
    with col_p2: st.plotly_chart(fig_fat,   use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔅 Key Insights")
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("""
        <div style="background: linear-gradient(135deg,#667eea,#764ba2); color:white;
                    padding:20px; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.2);">
            <h4>🔅 Hotspot Identification</h4>
            <ul>
                <li>Geographic clusters of crashes</li>
                <li>Segments requiring immediate attention</li>
                <li>Patterns in crash severity distribution</li>
                <li>Areas for targeted enforcement</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with ci2:
        st.markdown("""
        <div style="background: linear-gradient(135deg,#f093fb,#f5576c); color:white;
                    padding:20px; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.2);">
            <h4>🔅 Safety Recommendations</h4>
            <ul>
                <li>Deploy resources to high-density zones</li>
                <li>Increase patrols in top-ranked Segments</li>
                <li>Investigate infrastructure improvements</li>
                <li>Monitor seasonal variation patterns</li>
            </ul>
        </div>""", unsafe_allow_html=True)


# ============================================================================
# PAGE 2 — DAILY PROBABILISTIC CRASH PREDICTION
# ============================================================================
def show_forecast_page():
    st.title("🏎️ Daily Traffic Crash Risk Prediction — Shelby County")

    # ---- Discover available Segments from output folders ----
    mslinks = discover_mslinks()
    if not mslinks:
        st.warning(f"No prediction outputs found in `{BASE_OUTPUT_DIR}`. "
                   "Run the pipeline first.")
        return

    # ---- Load raw data to build route → MSLINK mapping ----
    # We peek at the raw CSV to know which route each MSLINK belongs to,
    # so the Route filter mirrors the hotspot-map page.
    @st.cache_data(ttl=3600)
    def build_mslink_route_map():
        try:
            df = pd.read_csv(
                "data/Unlocked_Segmented_2022_2025_Crashes_Final.csv",
                usecols=['MSLINK', 'RTE_NME'], low_memory=False
            )
            df['MSLINK']   = df['MSLINK'].astype(str)
            df['RTE_NME']  = df['RTE_NME'].astype(str)
            mapping = df.drop_duplicates('MSLINK').set_index('MSLINK')['RTE_NME'].to_dict()
            return mapping
        except Exception:
            return {}

    mslink_route_map = build_mslink_route_map()

    # Build route → [mslink, …] groups (only for mslinks that have output data)
    route_groups: dict[str, list[str]] = {"All Routes": mslinks}
    for ml in mslinks:
        route = mslink_route_map.get(str(ml), "Unknown")
        route_groups.setdefault(route, []).append(ml)

    # ---- Sidebar-style filters rendered inline ----
    st.markdown("### 🔎 Filter Options")
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        route_options   = list(route_groups.keys())
        selected_route  = st.selectbox("Select Route", route_options, key="fc_route")

    with col_f2:
        available_mslinks = route_groups[selected_route]
        selected_mslink   = st.selectbox(
            "Select MSLINK",
            available_mslinks,
            format_func=lambda m: f"MSLINK {m}  ({mslink_route_map.get(str(m), '—')})",
            key="fc_mslink"
        )

    # ---- Load data ----
    future_df   = load_segment_data(selected_mslink)
    historical_df = load_historical_data(selected_mslink)

    if future_df is None or future_df.empty:
        st.error(f"No prediction data found for MSLINK {selected_mslink}.")
        st.info(f"Expected: `{segment_folder(selected_mslink)}/data/MSLINK_{selected_mslink}_future_predictions_with_risk.csv`")
        return

    route_label = mslink_route_map.get(str(selected_mslink), "")
    st.markdown(f"### 🔅 MSLINK **{selected_mslink}** &nbsp;|&nbsp; {route_label} &nbsp;|&nbsp; {len(future_df)}-day forecast")

    # ---- Date range filter for forecast ----
    min_date = future_df['date'].min().date()
    max_date = future_df['date'].max().date()
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Forecast Start", value=min_date,
                                   min_value=min_date, max_value=max_date, key="fc_start")
    with col_d2:
        end_date   = st.date_input("Forecast End",   value=max_date,
                                   min_value=min_date, max_value=max_date, key="fc_end")

    mask       = (future_df['date'].dt.date >= start_date) & (future_df['date'].dt.date <= end_date)
    future_filt = future_df[mask].copy()

    if future_filt.empty:
        st.warning("No forecast data in the selected date range.")
        return

    # ---- Summary metrics ----
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("✨ Total Expected λ",   f"{future_filt['lambda'].sum():.2f}")
    with c2: st.metric("✨ Daily Average λ",     f"{future_filt['lambda'].mean():.3f}")
    with c3:
        peak = future_filt.loc[future_filt['lambda'].idxmax()]
        st.metric("✨ Peak Risk Day", peak['date'].strftime("%b %d, %Y"),
                  delta=f"λ = {peak['lambda']:.3f}")
    with c4:
        high_days = int((future_filt['risk_level'] == 'High').sum()) if 'risk_level' in future_filt.columns else 0
        st.metric("✨ High-Risk Days", str(high_days))

    st.markdown("---")

    # ---- Historical daily crashes ----
    st.subheader("🔅 Historical Daily Crashes")
    if not historical_df.empty:
        # Optional year filter for historical view
        hist_years = sorted(historical_df['date'].dt.year.unique().tolist())
        sel_years  = st.multiselect("Filter Historical Years", hist_years, default=hist_years,
                                    key="hist_year_filter")
        hist_filt  = historical_df[historical_df['date'].dt.year.isin(sel_years)] if sel_years else historical_df

        st.markdown("""
        <div style="border:1px solid #1f77b4; border-radius:10px; padding:20px;
                    background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        st.plotly_chart(create_historical_plot(hist_filt), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No historical data available for this MSLINK.")
        st.info(f"Expected: `{segment_folder(selected_mslink)}/data/MSLINK_{selected_mslink}_daily_crashes.csv`")

    st.markdown("---")

    # ---- Probabilistic forecast ----
    st.subheader("🔅 Daily Probabilistic Crash Forecast")
    st.markdown("""
    <div style="border:1px solid #1f77b4; border-radius:10px; padding:20px;
                background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
    """, unsafe_allow_html=True)
    st.plotly_chart(create_forecast_plot(future_filt), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Risk calendar heatmap ----
    st.markdown("---")
    st.subheader("🔅 Daily Risk Calendar")
    st.markdown("""
    <div style="border:1px solid #1f77b4; border-radius:10px; padding:20px;
                background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
    """, unsafe_allow_html=True)
    st.plotly_chart(create_risk_calendar_heatmap(future_filt), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---- In-depth risk assessment ----
    st.subheader("🔅 In-Depth Daily Risk Assessment")

    day_labels = [r['date'].strftime('%A, %b %d, %Y') for _, r in future_filt.iterrows()]
    chosen     = st.selectbox("Select Day for Detailed Analysis", day_labels, index=0,
                              key="day_detail_sel")
    row        = future_filt.iloc[day_labels.index(chosen)]

    # Risk badge
    risk_level, risk_color = get_risk_level(row['lambda'])
    st.markdown(f"""
    <div style="text-align:center; margin:20px 0;">
        <span style="display:inline-block; padding:14px 40px; font-size:22px;
                     font-weight:bold; color:white; background:{risk_color};
                     border-radius:50px; box-shadow:0 8px 20px {risk_color}40;
                     text-transform:uppercase; letter-spacing:1.5px;">{risk_level}</span>
    </div>""", unsafe_allow_html=True)

    # λ details row
    md1, md2, md3, md4 = st.columns(4)
    with md1: st.metric("λ (Expected Crashes)",  f"{row['lambda']:.4f}")
    with md2: st.metric("Lower Bound (95%)",      str(int(row.get('predicted_lower', 0))))
    with md3: st.metric("Upper Bound (95%)",      str(int(row.get('predicted_upper', row['lambda'] * 2))))
    with md4: st.metric("Forecast Method",        str(row.get('method', '—')))

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauges + Pie
    colL, colR = st.columns(2)
    with colL:
        g1, g2, _, _ = create_dual_gauges(row)
        st.markdown("""<div style="border:1px solid #1f77b4; border-radius:10px; padding:15px;
                        background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1); margin-bottom:20px;">
                    """, unsafe_allow_html=True)
        st.plotly_chart(g1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div style="border:1px solid #1f77b4; border-radius:10px; padding:15px;
                        background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
                    """, unsafe_allow_html=True)
        st.plotly_chart(g2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with colR:
        st.markdown("""<div style="border:1px solid #1f77b4; border-radius:10px; padding:15px;
                        background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
                    """, unsafe_allow_html=True)
        st.plotly_chart(create_probability_pie_chart(row), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Uncertainty breakdown
    st.markdown("---")
    st.subheader("🔅 Uncertainty Breakdown")
    ucols = ['model_uncertainty', 'residual_uncertainty', 'total_uncertainty']
    if all(c in row.index for c in ucols):
        uc1, uc2, uc3 = st.columns(3)
        with uc1: st.metric("Model Uncertainty (σ_model)",    f"{row['model_uncertainty']:.4f}")
        with uc2: st.metric("Residual Uncertainty (σ_resid)", f"{row['residual_uncertainty']:.4f}")
        with uc3: st.metric("Total Uncertainty (σ_total)",    f"{row['total_uncertainty']:.4f}")

    # Crash probability table
    st.markdown("---")
    st.subheader("🔅 Crash Probability Table")
    prob_cols = {
        'P(0 crashes)':  'prob_0_crash',
        'P(1 crash)':    'prob_1_crash',
        'P(2 crashes)':  'prob_2_crash',
        'P(3 crashes)':  'prob_3_crash',
        'P(≥4 crashes)': 'prob_ge4_crash',
    }
    prob_data = {k: f"{row.get(v, 0):.1f}%" for k, v in prob_cols.items()}
    st.dataframe(pd.DataFrame(prob_data, index=[chosen]).T.rename(columns={chosen: 'Probability'}),
                 use_container_width=True)

    # Full forecast table
    st.markdown("---")
    st.subheader("📋 Full Forecast Table")
    display_cols = ['date', 'lambda', 'predicted_lower', 'predicted_upper',
                    'risk_level', 'method', 'most_likely_crashes', 'probability_%',
                    'prob_0_crash', 'prob_1_crash', 'prob_2_crash',
                    'prob_3_crash', 'prob_ge4_crash']
    show_cols = [c for c in display_cols if c in future_filt.columns]
    tbl = future_filt[show_cols].copy()
    tbl['date'] = tbl['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(tbl.style.format({
        'lambda': '{:.4f}',
        'prob_0_crash': '{:.1f}%', 'prob_1_crash': '{:.1f}%',
        'prob_2_crash': '{:.1f}%', 'prob_3_crash': '{:.1f}%',
        'prob_ge4_crash': '{:.1f}%', 'probability_%': '{:.1f}%'
    }), use_container_width=True, height=400)


# ============================================================================
# PAGE 3 — HELP
# ============================================================================
def show_help_page():
    st.title("✍️ Help & User Guide")
    st.markdown("### Understanding Daily Probabilistic Predictions")

    st.markdown("## ✍️ Reading the Predictions")
    st.markdown("""
    **Example Prediction:** Wednesday May 14 shows "1 crash (Most likely: 1, 36%)"

    **This means:**
    - We expect 1 crash on that day based on statistical models
    - The single most likely outcome is exactly 1 crash
    - There's a 36% probability of exactly 1 crash occurring
    - We're 95% confident the actual count will fall within the predicted range
    """)

    st.markdown("---")
    st.markdown("## ✍️ Risk Level Interpretation (Daily Thresholds)")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**🟢 Very Low Risk** — λ < 0.2\nLow probability of any crash. Standard patrol.")
        st.warning("**🟡 Low Risk** — λ 0.2–0.5\nModerate probability. Maintain readiness.")
    with col2:
        st.info("**🟠 Medium Risk** — λ 0.5–1.0\nLikely at least one crash. Increase vigilance.")
        st.error("**🔴 High Risk** — λ ≥ 1.0\nOne or more crashes highly likely. Max enforcement.")

    st.markdown("---")
    st.markdown("## ✍️ Key Features on the Forecast Page")
    with st.expander("Daily Risk Calendar"):
        st.markdown("""
        The calendar heatmap shows the forecast period arranged by day-of-week and week.
        Greener = lower risk, Redder = higher risk. Use it to spot dangerous days at a glance.
        """)
    with st.expander("Probability Distribution Pie Chart"):
        st.markdown("""
        Shows the Poisson probability of 0, 1, 2, 3, or 4+ crashes on the selected day.
        The most-likely outcome is highlighted (pulled slice). The centre annotation shows
        the most likely count and its probability.
        """)
    with st.expander("Uncertainty Breakdown"):
        st.markdown("""
        - **σ_model** — disagreement between the 3 ensemble models (RF, GBR, Ridge)
        - **σ_resid** — historical residual variance from the training fit
        - **σ_total** — combined uncertainty used to draw the 95% CI; grows with forecast horizon
        """)
    with st.expander("Forecast Methods"):
        st.markdown("""
        | Days ahead | Method      | Description |
        |-----------|-------------|-------------|
        | 1–14      | ML          | Pure ensemble prediction |
        | 15–30     | ML+Season   | Light seasonal blending |
        | 31–60     | Hybrid      | Heavier seasonal blend |
        | 61+       | Seasonal    | Historical seasonal mean |
        """)

    st.markdown("---")
    st.markdown("## ❓ FAQs")
    with st.expander("Why show ranges instead of exact numbers?"):
        st.markdown("Crashes are random events — ranges reflect real uncertainty and help with flexible resource planning.")
    with st.expander("What does 95% confidence mean?"):
        st.markdown("In 95 out of 100 similar days, actual crashes will fall inside the predicted range.")
    with st.expander("How is MSLINK used?"):
        st.markdown("MSLINK is the unique road-segment identifier from the crash database. "
                    "Each MSLINK gets its own model trained on its own crash history.")

    st.markdown("---")
    st.info("**Support:** ctiermemphis@gmail.com | Mon–Fri 9 AM–5 PM CST | C-TIER, The University of Memphis")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        login_page()
    else:
        with st.sidebar:
            logo_b64 = get_image_base64("images/Safe_TN_Logo.png")
            if logo_b64:
                st.markdown(f'<img src="data:image/png;base64,{logo_b64}" width="200">',
                            unsafe_allow_html=True)
            else:
                st.markdown("### SAFE TN")

            st.markdown("### Safety Analytics & Forecasting Environment")
            st.title("Let's Navigate")

            page = st.radio(
                "Go to",
                ["Interactive Crash Maps", "Daily Crash Prediction", "Help & Guide"],
                label_visibility="collapsed",
                key="nav_radio"
            )

            st.markdown("---")
            st.markdown("""
            <div class="about-box">
                <b>About SAFE TN</b><br><br>
                SAFE TN (<i>Safety Analytics & Forecasting Environment for Tennessee</i>) is a
                probabilistic crash-risk prediction tool developed by the
                <b>Center for Transportation Innovation, Education and Research (C-TIER)</b>
                at The University of Memphis for the Tennessee Highway Safety Office and
                Enforcement agencies.<br><br>
                Using advanced machine-learning techniques, it delivers <b>daily</b> crash
                predictions with 95% confidence intervals for Shelby County.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="about-box">
                <b>About C-TIER</b><br><br>
                The <i>Center for Transportation Innovation, Education and Research (C-TIER)</i>
                at The University of Memphis has developed this predictive tool to enhance traffic
                safety across Tennessee by integrating real-time traffic and crash data.<br><br>
                Our research emphasises probabilistic prediction, behavioural analysis, and
                engineering solutions to support the <b>Tennessee Highway Safety Office</b>
                and local agencies.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.info("**Support**\nctiermemphis@gmail.com")
            st.write(f"**👤 Logged in as:** {st.session_state.username}")
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                logout()

        if page == "Interactive Crash Maps":
            show_hotspot_maps_page()
        elif page == "Daily Crash Prediction":
            show_forecast_page()
        elif page == "Help & Guide":
            show_help_page()
