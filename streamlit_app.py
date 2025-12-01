import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
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
    col1, col2, col3 = st.columns([1, 3, 1])
    
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
    fig = go.Figure()
    # [Your full original function – 100% unchanged]
    fig.add_trace(go.Scatter(x=weekly_df['week_start'], y=weekly_df['total_crashes'],
                             mode='lines+markers', name='Historical Crashes',
                             line=dict(color='#1f77b4', width=2), marker=dict(size=5),
                             hovertemplate='<b>%{x|%b %d, %Y}</b><br>Crashes: %{y:.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=future_df['week_start'], y=future_df['predicted_upper'],
                             mode='lines', name='Upper Bound (95% CI)',
                             line=dict(color='rgba(100,149,237,0)', width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=future_df['week_start'], y=future_df['predicted_lower'],
                             mode='lines', name='95% Confidence Interval',
                             line=dict(color='rgba(100,149,237,0)', width=0),
                             fill='tonexty', fillcolor='rgba(100,149,237,0.2)'))
    fig.add_trace(go.Scatter(x=future_df['week_start'], y=future_df['predicted_lower'],
                             mode='lines', name='Lower Bound', line=dict(color='#6495ED', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=future_df['week_start'], y=future_df['predicted_mean'],
                             mode='lines+markers', name='Mean Prediction',
                             line=dict(color='#ff7f0e', width=3, dash='dash'),
                             marker=dict(size=8, symbol='diamond')))
    fig.add_trace(go.Scatter(x=future_df['week_start'], y=future_df['predicted_upper'],
                             mode='lines', name='Upper Bound', line=dict(color='#6495ED', width=1.5, dash='dot')))
    last_date = weekly_df['week_start'].max()
    fig.add_shape(type="line", x0=last_date, x1=last_date, y0=0, y1=1, yref="paper",
                  line=dict(color="red", width=2, dash="dot"))
    fig.add_annotation(x=last_date, y=1, yref="paper", text="Forecast Start",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=12, color="red"))
    fig.update_layout(title='Weekly Crash History & Probabilistic Forecast',
                      xaxis_title='Date', yaxis_title='Expected Crashes per Week',
                      hovermode='x unified', height=520, template='plotly_white',
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def get_risk_level(val):
    if val < 0.5:   return "Very Low Risk", "#28a745"
    elif val < 1.0: return "Low Risk", "#ffc107"
    elif val < 1.5: return "Moderate Risk", "#fd7e14"
    elif val < 2.0: return "High Risk", "#dc3545"
    else:           return "Very High Risk", "#c82333"

def create_12week_bar_chart(future_df):
    colors = [get_risk_level(v)[1] for v in future_df['predicted_mean']]
    lower_err = future_df['predicted_mean'] - future_df['predicted_lower']
    upper_err = future_df['predicted_upper'] - future_df['predicted_mean']
    fig = go.Figure(go.Bar(x=future_df['week_start'], y=future_df['predicted_mean'],
                           marker_color=colors,
                           error_y=dict(type='data', symmetric=False, array=upper_err, arrayminus=lower_err,
                                        thickness=2, width=8),
                           text=future_df['crash_range'], textposition='outside'))
    fig.update_layout(title="Next 12 Weeks – Expected Crashes (95% Confidence Intervals)",
                      height=500, template="plotly_white", showlegend=False)
    return fig

def create_dual_gauges(row):
    exp = int(row['most_likely_crashes'])
    prob = row['likelihood_percent']
    mean = row['predicted_mean']
    level, color = get_risk_level(mean)
    fig1 = go.Figure(go.Indicator(mode="gauge+number", value=exp,
        title={'text': "Expected Crash<br><sub>Most Likely Outcome</sub>"},
        gauge={'axis': {'range': [0, 5]}, 'bar': {'color': color},
               'steps': [{'range': [0,1], 'color': '#d4edda'}, {'range': [1,2], 'color': '#fff3cd'},
                         {'range': [2,3], 'color': '#f8d7da'}, {'range': [3,5], 'color': '#f5c6cb'}]},
        number={'suffix': " crash", 'font': {'size': 52}}))
    fig1.update_layout(height=310, margin=dict(t=100, b=10))
    fig2 = go.Figure(go.Indicator(mode="gauge+number", value=prob,
        title={'text': "Confidence Level<br><sub>Probability of Most Likely</sub>"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"}},
        number={'suffix': "%", 'font': {'size': 52}}))
    fig2.update_layout(height=310, margin=dict(t=100, b=10))
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

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Expected (Next 12 Weeks)", f"{future_df['predicted_mean'].sum():.1f}")
        with c2: st.metric("Weekly Average", f"{future_df['predicted_mean'].mean():.2f}")
        with c3:
            peak = future_df.loc[future_df['predicted_mean'].idxmax()]
            st.metric("Peak Risk Week", f"{peak['predicted_mean']:.2f}", delta=peak['week_start'].strftime("%b %d"))

        st.markdown("---")
        st.plotly_chart(create_forecast_plot_with_intervals(weekly_df, future_df), use_container_width=True)
        st.markdown("---")
        st.subheader("Next 12 Weeks – Detailed Forecast")
        st.plotly_chart(create_12week_bar_chart(future_df), use_container_width=True)
        st.markdown("---")
        st.subheader("In-Depth Risk Assessment")
        selected_label = st.selectbox("Select Week", week_options, index=0)
        sel_row = future_df.iloc[week_options.index(selected_label)]
        gauge1, gauge2, risk_level, risk_color = create_dual_gauges(sel_row)
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(gauge1, use_container_width=True)
            st.plotly_chart(gauge2, use_container_width=True)
        with col_right:
            start = sel_row['week_start'].strftime("%B %d, %Y")
            end = (sel_row['week_start'] + timedelta(days=6)).strftime("%B %d, %Y")
            st.markdown(f"""
            <div style="background-color:{risk_color}15; border-left:8px solid {risk_color}; 
                        border-radius:14px; padding:35px; height:100%; display:flex; flex-direction:column; 
                        justify-content:center; box-shadow:0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="color:{risk_color}; margin:0;">Risk Summary</h2>
                <h4 style="margin:15px 0 25px 0; color:#2c3e50; font-weight:normal;">{start} to {end}</h4>
                <p style="font-size:18px; margin:12px 0;"><strong>Expected Range:</strong> {sel_row['crash_range']} crashes</p>
                <p style="font-size:18px; margin:12px 0;">
                    <strong>Most Likely:</strong> {int(sel_row['most_likely_crashes'])} crash 
                    <strong>({sel_row['likelihood_percent']:.1f}% probability)</strong>
                </p>
                <h2 style="color:{risk_color}; margin:30px 0 0 0; font-weight:bold;">{risk_level}</h2>
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