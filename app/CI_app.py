import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import hashlib

# ============================================================================
# AUTHENTICATION SYSTEM
# ============================================================================

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "Safe_TN": hash_password("ctiersafety_1")
}

def check_login(username, password):
    """Verify login credentials"""
    if username in USERS:
        return USERS[username] == hash_password(password)
    return False

def login_page():
    """Display login page"""
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            background-color: #f0f2f6;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image(r"D:\OneDrive - The University of Memphis\2024_THSO_DUI\Dashboard\images\C-TIER logo.PNG", width=300)
        st.title("Safety Analytics and Forecasting Environment for Tennessee")
        st.markdown("### SAFE TN")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("🔐 Login", use_container_width=True)
            
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
        
        st.markdown("---")
        st.info("""
        **The SAFE TN services are provided by Center for Transportation Innovations Education and Research for stakeholders to visualize crash prediction analyses easily.**
        - ⚠️ End user activities are monitored and logged. Unauthorized access is prohibited.
        - ⚠️ By logging in, you agree to comply with all applicable policies and guidelines.
        """)
        
        st.markdown("---")
        st.caption("For access issues, contact IT support")

def logout():
    """Logout user"""
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.rerun()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Traffic Crash Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING WITH CACHING
# ============================================================================

@st.cache_data
def load_data():
    """Load all data files with caching for performance"""
    try:
        weekly_df = pd.read_csv('data/weekly_crashes_enhanced.csv')
        weekly_df['week_start'] = pd.to_datetime(weekly_df['week_start'])
        
        # Load predictions with intervals
        future_df = pd.read_csv('data/future_predictions_with_intervals.csv')
        future_df['week_start'] = pd.to_datetime(future_df['week_start'])
        
        metrics_df = pd.read_csv('data/model_performance_comparison.csv')
        
        return weekly_df, future_df, metrics_df, None
    except Exception as e:
        return None, None, None, str(e)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_risk_level(crash_value):
    """Determine risk level and color based on mean prediction"""
    if crash_value < 0.5:
        return "Very Low Risk", "🟢", "#28a745"
    elif crash_value < 1.0:
        return "Low Risk", "🟡", "#ffc107"
    elif crash_value < 1.5:
        return "Moderate Risk", "🟠", "#fd7e14"
    elif crash_value < 2.0:
        return "High Risk", "🔴", "#dc3545"
    else:
        return "Very High Risk", "🔴", "#c82333"

# ============================================================================
# VISUALIZATION FUNCTIONS - UPDATED FOR INTERVALS
# ============================================================================

def create_forecast_plot_with_intervals(weekly_df, future_df, show_risk_zones=True):
    """Create forecast with confidence bands"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=weekly_df['week_start'],
        y=weekly_df['total_crashes'],
        mode='lines+markers',
        name='Historical Crashes',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=5),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Crashes: %{y:.0f}<extra></extra>'
    ))
    
    # Confidence band (shaded area)
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
        fillcolor='rgba(100,149,237,0.2)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Range: %{y:.1f}<extra></extra>'
    ))
    
    # Lower bound line
    fig.add_trace(go.Scatter(
        x=future_df['week_start'],
        y=future_df['predicted_lower'],
        mode='lines',
        name='Lower Bound',
        line=dict(color='#6495ED', width=1.5, dash='dot'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Min: %{y:.1f}<extra></extra>'
    ))
    
    # Mean prediction
    fig.add_trace(go.Scatter(
        x=future_df['week_start'],
        y=future_df['predicted_mean'],
        mode='lines+markers',
        name='Mean Prediction',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Expected: %{y:.1f}<extra></extra>'
    ))
    
    # Upper bound line
    fig.add_trace(go.Scatter(
        x=future_df['week_start'],
        y=future_df['predicted_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(color='#6495ED', width=1.5, dash='dot'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Max: %{y:.1f}<extra></extra>'
    ))
    
    # Vertical line
    last_date = weekly_df['week_start'].max()
    fig.add_shape(
        type="line",
        x0=last_date, x1=last_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dot")
    )
    fig.add_annotation(
        x=last_date, y=1, yref="paper",
        text="Forecast Start",
        showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(size=12, color="red")
    )
    
    fig.update_layout(
        title='Weekly Crash History & Probabilistic Forecast',
        xaxis_title='Date',
        yaxis_title='Expected Crashes per Week',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_risk_gauge_with_range(mean_value, lower_value, upper_value):
    """Create gauge with range indicator"""
    risk_level, emoji, color = get_risk_level(mean_value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=mean_value,
        title={'text': f"{risk_level}", 'font': {'size': 20}},
        delta={'reference': lower_value, 'suffix': f' (Range: {lower_value:.1f}-{upper_value:.1f})'},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 3], 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.5], 'color': '#d4edda'},
                {'range': [0.5, 1.0], 'color': '#fff3cd'},
                {'range': [1.0, 2.0], 'color': '#f8d7da'},
                {'range': [2.0, 3.0], 'color': '#f5c6cb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 2.0
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=80, b=20))
    return fig

def create_weekly_bars_with_intervals(future_df):
    """Create bar chart with error bars"""
    colors = [get_risk_level(val)[2] for val in future_df['predicted_mean']]
    
    fig = go.Figure()
    
    # Calculate error bar sizes
    lower_error = future_df['predicted_mean'] - future_df['predicted_lower']
    upper_error = future_df['predicted_upper'] - future_df['predicted_mean']
    
    fig.add_trace(go.Bar(
        x=future_df['week_start'],
        y=future_df['predicted_mean'],
        marker_color=colors,
        error_y=dict(
            type='data',
            symmetric=False,
            array=upper_error,
            arrayminus=lower_error,
            color='rgba(0,0,0,0.3)',
            thickness=1.5,
            width=5
        ),
        text=future_df['crash_range'],
        textposition='outside',
        hovertemplate='<b>Week of %{x|%b %d}</b><br>Range: %{text}<br>Mean: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Expected Crashes by Week (with 95% Confidence Intervals)',
        xaxis_title='Week Starting',
        yaxis_title='Expected Crashes',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load data
    weekly_df, future_df, metrics_df, error = load_data()
    
    if error:
        st.error(f"⚠️ Error loading data: {error}")
        st.info("""
        Please ensure the following files exist in the `data/` folder:
        - data/weekly_crashes_enhanced.csv
        - data/future_predictions_with_intervals.csv
        - data/model_performance_comparison.csv
        
        Run the training script first:
        ```
        python scripts/train_model_CI.py
        ```
        """)
        return
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.image("images/Speeding_Crashes.jpg", width=100)
        st.title("Traffic Crash Prediction")
        st.markdown("---")
        
        st.markdown("### Navigation")
        page = st.radio(
            "Select View:",
            ["🏠 Dashboard Overview", "📅 Weekly Forecast", "📈 Model Performance", "❓ Help & Guide"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        total_pred = future_df['predicted_mean'].sum()
        avg_pred = future_df['predicted_mean'].mean()
        num_weeks = len(future_df)
        st.metric("Forecast Period", f"{num_weeks} weeks")
        st.metric("Total Expected", f"{total_pred:.1f} crashes")
        st.metric("Weekly Average", f"{avg_pred:.1f} crashes")
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.info("For technical support, contact:\nTechnical Team\nctiermemphis@gmail.com")
    
    # ========================================================================
    # DASHBOARD OVERVIEW
    # ========================================================================
    
    if page == "🏠 Dashboard Overview":
        st.title("Traffic Crash Prediction for Shelby County")
        st.markdown("### Probabilistic crash risk forecasting with confidence intervals")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        num_weeks = len(future_df)
        total_predicted = future_df['predicted_mean'].sum()
        avg_predicted = future_df['predicted_mean'].mean()
        max_predicted = future_df['predicted_mean'].max()
        max_week = future_df.loc[future_df['predicted_mean'].idxmax(), 'week_start']
        
        with col1:
            st.metric(
                label=f"📊 Total Expected ({num_weeks} weeks)",
                value=f"{total_predicted:.1f}",
                delta=f"{((total_predicted/num_weeks - weekly_df['total_crashes'].mean()) / weekly_df['total_crashes'].mean() * 100):.1f}% vs avg"
            )
        
        with col2:
            st.metric(
                label="📈 Weekly Average",
                value=f"{avg_predicted:.1f}",
                help="Average expected crashes per week"
            )
        
        with col3:
            st.metric(
                label="⚠️ Peak Week Risk",
                value=f"{max_predicted:.1f}",
                delta=f"{max_week.strftime('%b %d')}"
            )
        
        with col4:
            st.metric(
                label="✅ Model Accuracy",
                value=f"{metrics_df.iloc[1]['R²']:.1%}",
                help="R² Score - How well the model fits the data"
            )
        
        st.markdown("---")
        
        # Current Week Alert with Range
        current_week = future_df.iloc[0]
        risk_level, emoji, color = get_risk_level(current_week['predicted_mean'])
        
        st.markdown(f"""
        <div style='background-color: {color}22; padding: 20px; border-radius: 10px; border-left: 5px solid {color}'>
            <h2>{emoji} This Week's Forecast</h2>
            <p style='font-size: 18px;'><strong>Week of {current_week['week_start'].strftime('%B %d, %Y')}</strong></p>
            <p style='font-size: 24px; font-weight: bold; color: {color};'>Expected Range: {current_week['crash_range']} crashes</p>
            <p style='font-size: 18px;'>Mean: {current_week['predicted_mean']:.1f} | Most Likely: {current_week['most_likely_crashes']} ({current_week['likelihood_percent']:.1f}% probability)</p>
            <p style='font-size: 16px;'>{risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main Forecast Plot with Intervals
        st.subheader(f"📊 Historical Data & {num_weeks}-Week Probabilistic Forecast")
        fig_forecast = create_forecast_plot_with_intervals(weekly_df, future_df)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Risk Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Current Week Risk Gauge")
            fig_gauge = create_risk_gauge_with_range(
                current_week['predicted_mean'],
                current_week['predicted_lower'],
                current_week['predicted_upper']
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.subheader("📊 Forecast Uncertainty")
            st.markdown("""
            **Understanding Prediction Intervals:**
            - **Shaded band** = 95% confidence region
            - **Mean line** = Most likely outcome
            - **Range** = Minimum to maximum expected
            
            **Current Week Details:**
            """)
            st.info(f"""
            - 95% Confidence: [{current_week['predicted_lower']:.1f}, {current_week['predicted_upper']:.1f}]
            - Crash Range: {current_week['crash_range']}
            - Most Likely: {current_week['most_likely_crashes']} crashes ({current_week['likelihood_percent']:.1f}%)
            - Method: {current_week['method']}
            """)
    
    # ========================================================================
    # WEEKLY FORECAST
    # ========================================================================
    
    elif page == "📅 Weekly Forecast":
        num_weeks = len(future_df)
        st.title("📅 Detailed Weekly Probabilistic Forecast")
        st.markdown(f"### Next {num_weeks} weeks with confidence intervals")
        
        # Weekly bar chart with error bars
        fig_bars = create_weekly_bars_with_intervals(future_df)
        st.plotly_chart(fig_bars, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Week-by-Week Breakdown with Probabilities")
        
        # Create detailed table
        forecast_table = future_df.copy()
        forecast_table['Week Starting'] = forecast_table['week_start'].dt.strftime('%b %d, %Y')
        forecast_table['Week #'] = forecast_table['week_of_year']
        forecast_table['Expected Range'] = forecast_table['crash_range']
        forecast_table['Mean'] = forecast_table['predicted_mean'].round(2)
        forecast_table['Most Likely'] = forecast_table['most_likely_crashes'].astype(str) + ' (' + forecast_table['likelihood_percent'].astype(str) + '%)'
        forecast_table['Risk Level'] = forecast_table['predicted_mean'].apply(lambda x: get_risk_level(x)[0])
        
        display_df = forecast_table[['Week Starting', 'Week #', 'Expected Range', 'Mean', 'Most Likely', 'Risk Level']]
        
        # Color code
        def highlight_risk(row):
            if row['Risk Level'] == 'Very Low Risk':
                return ['background-color: #d4edda']*len(row)
            elif row['Risk Level'] == 'Low Risk':
                return ['background-color: #d1ecf1']*len(row)
            elif row['Risk Level'] == 'Moderate Risk':
                return ['background-color: #fff3cd']*len(row)
            elif row['Risk Level'] == 'High Risk':
                return ['background-color: #f8d7da']*len(row)
            else:
                return ['background-color: #f5c6cb']*len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_risk, axis=1),
            use_container_width=True,
            height=min(500, len(display_df) * 35 + 50)
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast (CSV)",
            data=csv,
            file_name=f"crash_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # ========================================================================
    # MODEL PERFORMANCE
    # ========================================================================
    
    elif page == "📈 Model Performance":
        st.title("📈 Model Performance & Accuracy")
        st.markdown("### Understanding predictions and uncertainty")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy (R²)", f"{metrics_df.iloc[1]['R²']:.1%}")
            st.metric("Average Error (MAE)", f"{metrics_df.iloc[1]['MAE']:.2f} crashes")
        
        with col2:
            st.metric("RMSE", f"{metrics_df.iloc[1]['RMSE']:.2f}")
            improvement = ((metrics_df.iloc[0]['MAE'] - metrics_df.iloc[1]['MAE']) / metrics_df.iloc[0]['MAE'] * 100)
            st.metric("Improvement vs Baseline", f"{improvement:.1f}%")
        
        with col3:
            avg_interval = (future_df['predicted_upper'] - future_df['predicted_lower']).mean()
            st.metric("Avg Interval Width", f"{avg_interval:.2f} crashes")
            st.metric("Forecast Method", "Hybrid ML+Seasonal")
        
        st.markdown("---")
        
        st.subheader("📚 Understanding Prediction Intervals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **What are Prediction Intervals?**
            
            Instead of saying "exactly 1.2 crashes," we provide:
            - **A range** (e.g., 0-2 crashes)
            - **A most likely value** (e.g., 1 crash, 36% probability)
            - **Confidence bounds** (95% sure actual will be in this range)
            
            This reflects the **random nature** of crash events.
            """)
        
        with col2:
            st.markdown("""
            **Why Ranges Instead of Point Estimates?**
            
            Crashes are random events influenced by:
            - Weather conditions
            - Driver behavior
            - Traffic patterns
            - Unpredictable factors
            
            Providing ranges is more **honest and useful** for planning.
            """)
        
        st.info("💡 **Bottom Line:** The model provides realistic, probabilistic forecasts that acknowledge uncertainty while maintaining high accuracy.")
    
    # ========================================================================
    # HELP & GUIDE
    # ========================================================================
    
    elif page == "❓ Help & Guide":
        st.title("❓ Help & User Guide")
        st.markdown("### Understanding probabilistic forecasts")
        
        st.markdown("## 🔢 Reading the Predictions")
        
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
            🟢 Very Low Risk
            - Low probability of crashes
            - Standard patrol sufficient
            """)
            
            st.warning("""
            **Range: 1-2 crashes**
            🟠 Moderate Risk
            - Likely 1-2 crashes
            - Increase vigilance
            """)
        
        with col2:
            st.info("""
            **Range: 1-2 crashes**
            🟡 Low Risk
            - Moderate probability
            - Maintain readiness
            """)
            
            st.error("""
            **Range: 2-3+ crashes**
            🔴 High Risk
            - Multiple crashes likely
            - Maximum enforcement
            """)
        
        st.markdown("---")
        
        st.markdown("## ❓ FAQ")
        
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
# RUN APP
# ============================================================================

if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['username'] = None
    
    if not st.session_state['authenticated']:
        login_page()
    else:
        with st.sidebar:
            st.markdown("---")
            st.write(f"👤 Logged in as: **{st.session_state['username']}**")
            if st.button("🚪 Logout", use_container_width=True):
                logout()
        
        main()