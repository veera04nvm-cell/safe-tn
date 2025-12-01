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

# Store users (In production, use a database!)
# Format: username: hashed_password
USERS = {
    "Safe_TN": hash_password("ctiersafety_1")  # Change these!
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
        - ⚠️ By logging in, you agree to comply with all applicable policies and guidelines.`
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
    # page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .risk-very-low {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .risk-low {
        background-color: #d1ecf1;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
    }
    .risk-moderate {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    h1 {
        color: #1f77b4;
    }
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
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
        
        future_df = pd.read_csv('data/future_predictions_ml.csv')
        future_df['week_start'] = pd.to_datetime(future_df['week_start'])
        
        metrics_df = pd.read_csv('data/model_performance_comparison.csv')
        
        return weekly_df, future_df, metrics_df, None
    except Exception as e:
        return None, None, None, str(e)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_risk_level(crash_value):
    """Determine risk level and color"""
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

def format_crash_interpretation(value):
    """Convert crash expectation to probability interpretation"""
    if value < 0.5:
        return f"{int(value * 100)}% chance of a crash"
    elif value < 1.5:
        return f"Likely {int(value)} crash(es)"
    else:
        return f"Likely {int(value)}-{int(value)+1} crashes"

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_forecast_plot(weekly_df, future_df, show_risk_zones=True):
    """Create main forecast visualization - FIXED VERSION"""
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
    
    # Future predictions
    fig.add_trace(go.Scatter(
        x=future_df['week_start'],
        y=future_df['predicted_crashes'],
        mode='lines+markers',
        name='Predicted Crashes',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Expected: %{y:.1f}<extra></extra>'
    ))
    
    # Add vertical line separating past and future (FIXED)
    last_date = weekly_df['week_start'].max()
    fig.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dot")
    )
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=12, color="red")
    )
    
    # Risk zones
    if show_risk_zones:
        fig.add_hrect(y0=0, y1=0.5, fillcolor="green", opacity=0.1, layer="below", line_width=0)
        fig.add_hrect(y0=0.5, y1=1.0, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
        fig.add_hrect(y0=1.0, y1=2.0, fillcolor="orange", opacity=0.1, layer="below", line_width=0)
        fig.add_hrect(y0=2.0, y1=5.0, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    
    fig.update_layout(
        title='Weekly Crash History & Forecast',
        xaxis_title='Date',
        yaxis_title='Expected Crashes per Week',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_risk_gauge(crash_value):
    """Create gauge chart for risk level"""
    risk_level, emoji, color = get_risk_level(crash_value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=crash_value,
        title={'text': f"{risk_level}", 'font': {'size': 24}},
        number={'font': {'size': 48}},
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

def create_weekly_bars(future_df):
    """Create bar chart of weekly predictions"""
    colors = [get_risk_level(val)[2] for val in future_df['predicted_crashes']]
    
    fig = go.Figure(go.Bar(
        x=future_df['week_start'],
        y=future_df['predicted_crashes'],
        marker_color=colors,
        text=future_df['predicted_crashes'].round(1),
        textposition='outside',
        hovertemplate='<b>Week of %{x|%b %d}</b><br>Expected: %{y:.1f} crashes<extra></extra>'
    ))
    
    fig.update_layout(
        title='Expected Crashes by Week',
        xaxis_title='Week Starting',
        yaxis_title='Expected Crashes',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_risk_distribution(future_df):
    """Create pie chart of risk levels"""
    risk_categories = pd.cut(
        future_df['predicted_crashes'],
        bins=[0, 0.5, 1.0, 1.5, 2.0, 10],
        labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    )
    risk_counts = risk_categories.value_counts()
    
    colors = ['#28a745', '#17a2b8', '#ffc107', '#fd7e14', '#dc3545']
    
    fig = go.Figure(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>%{value} weeks<br>%{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Risk Level Distribution (Next {len(future_df)} Weeks)',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_model_performance(metrics_df):
    """Create model performance comparison"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['MAE', 'RMSE', 'R²'],
        y=[metrics_df.iloc[0]['MAE'], metrics_df.iloc[0]['RMSE'], metrics_df.iloc[0]['R²']],
        name='Baseline Model',
        marker_color='#3498db',
        text=[f"{metrics_df.iloc[0]['MAE']:.2f}", 
              f"{metrics_df.iloc[0]['RMSE']:.2f}", 
              f"{metrics_df.iloc[0]['R²']:.3f}"],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=['MAE', 'RMSE', 'R²'],
        y=[metrics_df.iloc[1]['MAE'], metrics_df.iloc[1]['RMSE'], metrics_df.iloc[1]['R²']],
        name='ML Ensemble',
        marker_color='#e74c3c',
        text=[f"{metrics_df.iloc[1]['MAE']:.2f}", 
              f"{metrics_df.iloc[1]['RMSE']:.2f}", 
              f"{metrics_df.iloc[1]['R²']:.3f}"],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Model Performance Metrics',
        xaxis_title='Metric',
        yaxis_title='Value',
        barmode='group',
        height=350,
        template='plotly_white'
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
        - data/future_predictions_ml.csv
        - data/model_performance_comparison.csv
        
        Run the training script first:
        ```
        python scripts/train_model.py
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
        st.markdown("### ⚙️ Settings")
        show_risk_zones = st.checkbox("Show Risk Zones", value=True)
        show_historical = st.checkbox("Show Historical Data", value=True)
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        total_pred = future_df['predicted_crashes'].sum()
        avg_pred = future_df['predicted_crashes'].mean()
        num_weeks = len(future_df)
        st.metric("Forecast Period", f"{num_weeks} weeks")
        st.metric("Total Expected", f"{total_pred:.1f} crashes")
        st.metric("Weekly Average", f"{avg_pred:.1f} crashes")
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.info("For technical support, contact:\nTechnical Team\n ctiermemphis@gmail.com")
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    if page == "🏠 Dashboard Overview":
        # Header
        st.title("Traffic Crash Prediction for Shelby County")
        st.markdown("### Real-time crash risk forecasting for enforcement planning")
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        num_weeks = len(future_df)
        total_predicted = future_df['predicted_crashes'].sum()
        avg_predicted = future_df['predicted_crashes'].mean()
        max_predicted = future_df['predicted_crashes'].max()
        max_week = future_df.loc[future_df['predicted_crashes'].idxmax(), 'week_start']
        
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
        
        # Current Week Alert
        current_week_pred = future_df.iloc[0]['predicted_crashes']
        current_week_date = future_df.iloc[0]['week_start']
        risk_level, emoji, color = get_risk_level(current_week_pred)
        
        st.markdown(f"""
        <div style='background-color: {color}22; padding: 20px; border-radius: 10px; border-left: 5px solid {color}'>
            <h2>{emoji} This Week's Forecast</h2>
            <p style='font-size: 18px;'><strong>Week of {current_week_date.strftime('%B %d, %Y')}</strong></p>
            <p style='font-size: 24px; font-weight: bold; color: {color};'>Expected: {current_week_pred:.1f} crashes</p>
            <p style='font-size: 16px;'>{risk_level} - {format_crash_interpretation(current_week_pred)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main Forecast Plot
        st.subheader(f" Historical Data & {num_weeks}-Week Forecast")
        fig_forecast = create_forecast_plot(weekly_df, future_df, show_risk_zones)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Risk Analysis Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Current Week Risk Gauge")
            fig_gauge = create_risk_gauge(current_week_pred)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.subheader(" Risk Distribution")
            fig_pie = create_risk_distribution(future_df)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # ========================================================================
    # WEEKLY FORECAST PAGE
    # ========================================================================
    
    elif page == "📅 Weekly Forecast":
        num_weeks = len(future_df)
        st.title("📅 Detailed Weekly Forecast")
        st.markdown(f"### Next {num_weeks} weeks of crash predictions")
        
        # Weekly bar chart
        fig_bars = create_weekly_bars(future_df)
        st.plotly_chart(fig_bars, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Week-by-Week Breakdown")
        
        # Create detailed table with risk levels
        forecast_table = future_df.copy()
        forecast_table['Week Starting'] = forecast_table['week_start'].dt.strftime('%b %d, %Y')
        forecast_table['Week #'] = forecast_table['week_of_year']
        forecast_table['Expected Crashes'] = forecast_table['predicted_crashes'].round(1)
        forecast_table['Risk Level'] = forecast_table['predicted_crashes'].apply(lambda x: get_risk_level(x)[0])
        forecast_table['Interpretation'] = forecast_table['predicted_crashes'].apply(format_crash_interpretation)
        
        display_df = forecast_table[['Week Starting', 'Week #', 'Expected Crashes', 'Risk Level', 'Interpretation']]
        
        # Color code the table
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
    # MODEL PERFORMANCE PAGE
    # ========================================================================
    
    elif page == "📈 Model Performance":
        st.title("📈 Model Performance & Accuracy")
        st.markdown("### Understanding how well the predictions work")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Accuracy (R²)", f"{metrics_df.iloc[1]['R²']:.1%}")
            st.metric("Average Error (MAE)", f"{metrics_df.iloc[1]['MAE']:.2f} crashes")
        
        with col2:
            st.metric("Root Mean Square Error", f"{metrics_df.iloc[1]['RMSE']:.2f}")
            improvement = ((metrics_df.iloc[0]['MAE'] - metrics_df.iloc[1]['MAE']) / metrics_df.iloc[0]['MAE'] * 100)
            st.metric("Improvement vs Baseline", f"{improvement:.1f}%")
        
        st.markdown("---")
        
        # Performance comparison chart
        fig_perf = create_model_performance(metrics_df)
        st.plotly_chart(fig_perf, use_container_width=True)
        
        st.markdown("---")
        
        # Explanation
        st.subheader("📚 What These Metrics Mean")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **R² (R-Squared): Model Fit**
            - Ranges from 0% to 100%
            - Our model: **{:.1%}**
            - Interpretation: The model explains {:.1%} of crash variations
            - ✅ Excellent performance!
            """.format(metrics_df.iloc[1]['R²'], metrics_df.iloc[1]['R²']))
        
        with col2:
            st.markdown("""
            **MAE (Mean Absolute Error)**
            - Average prediction error
            - Our model: **{:.2f} crashes**
            - Interpretation: Predictions are typically off by ±{:.2f} crashes
            - ✅ Very accurate!
            """.format(metrics_df.iloc[1]['MAE'], metrics_df.iloc[1]['MAE']))
        
        st.info("💡 **Bottom Line:** The model is highly accurate and reliable for enforcement planning.")
    
    # ========================================================================
    # HELP & GUIDE PAGE
    # ========================================================================
    
    elif page == "❓ Help & Guide":
        st.title("❓ Help & User Guide")
        st.markdown("### How to use this dashboard effectively")
        
        # Quick Start
        st.markdown("## Quick Start Guide")
        st.markdown("""
        1. **Check the Dashboard Overview** - See current week's risk and overall forecast
        2. **Review Weekly Forecast** - Plan enforcement activities for high-risk weeks
        3. **Monitor Risk Levels** - Use color-coded alerts for quick decision making
        4. **Download Reports** - Export forecasts for team meetings and planning
        """)
        
        st.markdown("---")
        
        # Understanding the Numbers
        st.markdown("## 🔢 Understanding the Predictions")
        
        st.markdown("""
        The numbers you see (like 0.3, 1.2, 2.1) represent **expected crash risk**, not exact counts.
        
        Think of them like weather probabilities:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **0.1 - 0.5 crashes**
            🟢 Very Low Risk
            - 10-50% chance of a crash
            - Normal patrol activities
            - Standard monitoring
            """)
            
            st.warning("""
            **1.0 - 2.0 crashes**
            🟠 Moderate Risk
            - High likelihood of 1-2 crashes
            - Increase patrols
            - Enhanced visibility
            """)
        
        with col2:
            st.info("""
            **0.5 - 1.0 crashes**
            🟡 Low Risk
            - 50-100% chance of a crash
            - Standard enforcement
            - Monitor conditions
            """)
            
            st.error("""
            **2.0+ crashes**
            🔴 High Risk
            - Very likely 2-3 crashes
            - Maximum enforcement
            - Special attention needed
            """)
        
        st.markdown("---")
        
        # Frequently Asked Questions
        st.markdown("## ❓ Frequently Asked Questions")
        
        with st.expander("Why are the numbers decimals instead of whole numbers?"):
            st.markdown("""
            These are **statistical expectations** - like saying "average household has 2.3 people."
            
            You won't see exactly 0.3 crashes, but it tells you the risk level:
            - 0.3 = 30% probability
            - 1.5 = Most likely 1 or 2 crashes
            - 2.2 = Most likely 2 or 3 crashes
            """)
        
        with st.expander("How accurate are these predictions?"):
            st.markdown(f"""
            Our model has **{metrics_df.iloc[1]['R²']:.1%} accuracy** (R² score), which is excellent!
            
            - Average error: ±{metrics_df.iloc[1]['MAE']:.2f} crashes
            - The model learns from historical patterns, traffic conditions, and time factors
            - Predictions become more reliable when used for weekly planning rather than exact daily counts
            """)
        
        with st.expander("What should I do with high-risk week predictions?"):
            st.markdown("""
            When a week shows high risk (🔴 1.5+ expected crashes):
            
            1. **Increase patrol presence** during peak hours
            2. **Deploy resources** to historically problematic areas
            3. **Conduct targeted enforcement** (speed, DUI checkpoints)
            4. **Coordinate with team** for coverage planning
            5. **Monitor conditions** in real-time during that week
            """)
        
        with st.expander("Can I trust predictions far into the future?"):
            st.markdown("""
            - **Near-term (1-4 weeks)**: Most reliable - use for tactical planning
            - **Medium-term (5-12 weeks)**: Good for operational planning
            - **Longer-term (13-24 weeks)**: General trends for strategic planning
            - **Very long-term (25+ weeks)**: Seasonal patterns only
            
            The model uses a hybrid approach:
            - Short-term: Primarily machine learning (recent patterns)
            - Long-term: Blend of ML + seasonal patterns
            
            Unexpected events (construction, weather, special events) can change outcomes.
            """)
        
        with st.expander("Why do predictions vary more in the long-term?"):
            st.markdown("""
            This is **intentional and realistic**! Here's why:
            
            - Real crash patterns aren't perfectly smooth
            - Seasonal factors cause natural variation (weather, holidays, etc.)
            - The hybrid model blends ML predictions with historical seasonal patterns
            - Small random variations represent unpredictable factors
            
            This variation helps with realistic planning rather than giving false precision.
            """)
        
        st.markdown("---")
        
        # Contact Support
        st.markdown("## 📞 Need Help?")
        st.info("""
        **Technical Support:**
        - Email: ctiermemphis@gmail.com
        - Hours: Monday-Friday, 9AM-5PM
        
        **For urgent issues during operations:**
        - Contact your supervisor
        - Call dispatch center
        """)

# ============================================================================
# RUN APP WITH AUTHENTICATION
# ============================================================================

if __name__ == "__main__":
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['username'] = None
    
    # Check authentication
    if not st.session_state['authenticated']:
        login_page()
    else:
        # Show logout button in sidebar
        with st.sidebar:
            st.markdown("---")
            st.write(f"👤 Logged in as: **{st.session_state['username']}**")
            if st.button("🚪 Logout", use_container_width=True):
                logout()
        
        # Run main app
        main()