import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import hashlib

# ============================================================================
# AUTHENTICATION
# ============================================================================
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()
USERS = {"Safe_TN": hash_password("ctiersafety_1")}

def check_login(u, p): return u in USERS and USERS[u] == hash_password(p)

def login_page():
    st.markdown("""
    <style>
    .login-box {max-width:420px; margin:100px auto; padding:40px; background:#f8f9fa; 
                border-radius:12px; box-shadow:0 8px 30px rgba(0,0,0,0.18); text-align:center;}
    </style>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(r"D:\OneDrive - The University of Memphis\2024_THSO_DUI\Dashboard\images\C-TIER logo.PNG", width=280)
        st.title("SAFE TN")
        st.markdown("#### Safety Analytics & Forecasting Environment")
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                if check_login(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        st.caption("Contact: ctiermemphis@gmail.com")

def logout():
    st.session_state.authenticated = False
    st.rerun()

# ============================================================================
# CONFIG
# ============================================================================
st.set_page_config(page_title="SAFE TN – Crash Risk Forecast", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    h1, h2, h3 {color:#1f77b4;}
    .stMetric {background:#f0f2f6; padding:18px; border-radius:12px; box-shadow:0 3px 10px rgba(0,0,0,0.1);}
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
# YOUR ORIGINAL HISTORICAL + FORECAST PLOT (100% PRESERVED)
# ============================================================================
def create_forecast_plot_with_intervals(weekly_df, future_df):
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
    
    # Confidence band
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], y=future_df['predicted_upper'],
        mode='lines', name='Upper Bound (95% CI)',
        line=dict(color='rgba(100,149,237,0)', width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=future_df['week_start'], y=future_df['predicted_lower'],
        mode='lines', name='95% Confidence Interval',
        line=dict(color='rgba(100,149,237,0)', width=0),
        fill='tonexty', fillcolor='rgba(100,149,237,0.2)',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Range: %{y:.1f}<extra></extra>'
    ))
    
    # Bounds
    fig.add_trace(go.Scatter(x=future_df['week_start'], y=future_df['predicted_lower'],
                             mode='lines', name='Lower Bound', line=dict(color='#6495ED', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=future_df['week_start'], y=future_df['predicted_mean'],
                             mode='lines+markers', name='Mean Prediction',
                             line=dict(color='#ff7f0e', width=3, dash='dash'),
                             marker=dict(size=8, symbol='diamond')))
    fig.add_trace(go.Scatter(x=future_df['week_start'], y=future_df['predicted_upper'],
                             mode='lines', name='Upper Bound', line=dict(color='#6495ED', width=1.5, dash='dot')))
    
    # Forecast start
    last_date = weekly_df['week_start'].max()
    fig.add_shape(type="line", x0=last_date, x1=last_date, y0=0, y1=1, yref="paper",
                  line=dict(color="red", width=2, dash="dot"))
    fig.add_annotation(x=last_date, y=1, yref="paper", text="Forecast Start",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=12, color="red"))
    
    fig.update_layout(
        title='Weekly Crash History & Probabilistic Forecast',
        xaxis_title='Date', yaxis_title='Expected Crashes per Week',
        hovermode='x unified', height=520, template='plotly_white',
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ============================================================================
# OTHER PLOTS & UTILS
# ============================================================================
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
    
    fig = go.Figure(go.Bar(
        x=future_df['week_start'], y=future_df['predicted_mean'],
        marker_color=colors,
        error_y=dict(type='data', symmetric=False, array=upper_err, arrayminus=lower_err,
                     thickness=2, width=8),
        text=future_df['crash_range'], textposition='outside'
    ))
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
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"},
               'steps': [{'range': [0,30], 'color': '#ffebee'}, {'range': [30,60], 'color': '#fff3e0'},
                         {'range': [60,100], 'color': '#e8f5e8'}]},
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

    with st.sidebar:
        st.image("images/Speeding_Crashes.jpg", width=130)
        st.title("SAFE TN")
        page = st.radio("Navigation", ["Probablistic Crash Forecast", "Help & Guide"], label_visibility="collapsed")
        st.markdown("---")
        st.metric("Forecast Period", f"{len(future_df)} weeks")
        st.metric("Total Expected", f"{future_df['predicted_mean'].sum():.1f} crashes")
        st.metric("Weekly Average", f"{future_df['predicted_mean'].mean():.2f}")
        st.markdown("---")
        st.info("Support\nctiermemphis@gmail.com")

    # ========================================================================
    # DASHBOARD OVERVIEW
    # ========================================================================
    if page == "Probablistic Crash Forecast":
        st.title("Traffic Crash Risk Forecast – Shelby County")
        st.markdown("### Probabilistic weekly prediction with uncertainty")

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