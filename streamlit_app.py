"""
SAFE TN — Daily Crash Risk Prediction Dashboard
================================================
Updated from weekly to daily prediction pipeline.
- Segments identified by MSLINK (from daily_crash_prediction_pipeline.py)
- Raw crash CSV loaded via st.file_uploader() in the sidebar
- Pipeline output data still read from outputs/daily_risk_score/MSLINK_<id>/data/
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
BASE_OUTPUT_DIR  = 'outputs/daily_risk_score'   # pipeline writes here
MAX_MAP_POINTS   = 40_000                        # cap points sent to Plotly map traces
MAX_CHAT_HISTORY = 15                            # message pairs kept in session state

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
            submit   = st.form_submit_button("Login", use_container_width=True)

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
    st.session_state.username      = None
    st.session_state.crash_df      = None
    st.session_state.crash_data_loaded = False
    st.rerun()


# ============================================================================
# SEGMENT / MSLINK DISCOVERY
# Reads actual output folders — no hard-coded segment list needed.
# ============================================================================
@st.cache_data(ttl=3600)
def discover_mslinks():
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
# RAW CRASH CSV LOADER
# ============================================================================
def load_crash_shapefile_data(uploaded_file):
    """
    Load raw crash-level data from the uploaded CSV file.
    Replaces the previous hardcoded file path:
        'data/Unlocked_Segmented_2022_2025_Crashes_Final.csv'
    """
    try:
        usecols = [
            'GPS Coordi', 'GPS Coor_1', 'MSLINK', 'Type of Cr',
            'Year Of Cr', 'Total Kill', 'Total Inj',
            'Hit and Ru', 'RTE_NME', 'CNTY_SEAT', 'Date of Cr', 'Light Cond'
        ]
        df = pd.read_csv(
            uploaded_file,          # ← st.file_uploader object, not a path
            usecols=usecols,
            dtype={
                'MSLINK': 'category', 'Type of Cr': 'category',
                'RTE_NME': 'category', 'CNTY_SEAT': 'category',
                'Hit and Ru': 'category', 'Light Cond': 'category'
            },
            low_memory=False
        )
        df = df.rename(columns={
            'GPS Coordi': 'latitude',  'GPS Coor_1': 'longitude',
            'MSLINK': 'segment_id',    'Type of Cr': 'severity',
            'Year Of Cr': 'year',      'Total Kill': 'fatalities',
            'Total Inj': 'injuries',   'Hit and Ru': 'hit_and_run',
            'RTE_NME': 'route',        'CNTY_SEAT': 'city',
            'Date of Cr': 'Date of Crash', 'Light Cond': 'Light Condition'
        })
        df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce').astype('float32')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').astype('float32')
        df['year']       = pd.to_numeric(df['year'],       errors='coerce').astype('int16')
        df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0).astype('int8')
        df['injuries']   = pd.to_numeric(df['injuries'],   errors='coerce').fillna(0).astype('int8')
        df = df[df['year'] >= 2021]
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude']  >= 34.9) & (df['latitude']  <= 36.7)]
        df = df[(df['longitude'] >= -90.5) & (df['longitude'] <= -88.0)]
        return df
    except Exception as e:
        st.error(f"Error loading crash data: {e}")
        return None


# ============================================================================
# PIPELINE OUTPUT LOADERS  (paths unchanged — still read from disk)
# ============================================================================
@st.cache_data(ttl=3600)
def load_segment_data(mslink):
    """Load future daily predictions for a given MSLINK."""
    folder = segment_folder(mslink)
    path   = os.path.join(folder, "data", f"MSLINK_{mslink}_future_predictions_with_risk.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None


@st.cache_data(ttl=3600)
def load_historical_data(mslink):
    """Load historical daily crashes for a given MSLINK."""
    folder = segment_folder(mslink)
    path   = os.path.join(folder, "data", f"MSLINK_{mslink}_daily_crashes.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame(columns=['date', 'crash_count'])


@st.cache_data(ttl=3600)
def load_segmented_data():
    """Load segmented interstate data for background analysis."""
    try:
        return pd.read_csv("data/Segmented_Shelby_Interstates.csv")
    except Exception as e:
        print(f"Error loading segmented data: {e}")
        return None


# ============================================================================
# MSLINK → ROUTE MAPPING  (built from the uploaded crash CSV)
# ============================================================================
def build_mslink_route_map(crash_df):
    """
    Build MSLINK → route mapping from the already-loaded crash DataFrame.
    Previously this re-read the CSV from disk; now it uses the in-memory df.
    """
    try:
        df = crash_df[['segment_id', 'route']].drop_duplicates('segment_id').copy()
        df['segment_id'] = df['segment_id'].astype(str)
        return df.set_index('segment_id')['route'].to_dict()
    except Exception:
        return {}


# ============================================================================
# HELPER — RISK LEVEL (daily thresholds from pipeline)
# ============================================================================
def get_risk_level(lam):
    if lam >= 1.0:  return "High Risk",     "#dc3545"
    elif lam >= 0.5: return "Medium Risk",  "#fd7e14"
    elif lam >= 0.2: return "Low Risk",     "#ffc107"
    else:            return "Very Low Risk", "#28a745"


def get_most_likely_from_probabilities(row):
    probs = [
        row['prob_0_crash'], row['prob_1_crash'], row['prob_2_crash'],
        row['prob_3_crash'], row['prob_ge4_crash']
    ]
    most_likely_idx = probs.index(max(probs))
    crash_counts    = [0, 1, 2, 3, "4+"]
    return crash_counts[most_likely_idx], probs[most_likely_idx]


# ============================================================================
# CHATBOT — RULE-BASED RESPONSES
# ============================================================================
def chatbot_crash_response(msg, crash_df):
    m = msg.lower().strip()

    if crash_df is None:
        return "Please upload crash data first to enable queries."

    # Build MSLINK → route lookup; normalise keys to str
    route_map = (crash_df[['segment_id', 'route']]
                 .drop_duplicates('segment_id')
                 .assign(segment_id=lambda df: df['segment_id'].astype(str))
                 .set_index('segment_id')['route']
                 .to_dict())

    def _lbl(seg_id):
        sid   = str(seg_id)
        route = route_map.get(sid, '')
        return f"{route} {sid}".strip() if route else sid

    # ── Greetings & help ────────────────────────────────────────────────────
    if any(k in m for k in ['hello', 'hi', 'hey', 'howdy']):
        return ("Hi! I'm CrashBot 🤖\n\n"
                "Ask me anything about the crash data — segments, fatalities, "
                "severity, routes, years, hit-and-run, city, night crashes, and more.\n\n"
                "Type **help** for a full list of topics.")

    if any(k in m for k in ['help', 'what can you', 'what do you', 'capabilities', 'topics', 'options']):
        return ("**CrashBot can answer questions about:**\n\n"
                "📊 **Totals** — crashes, fatalities, injuries\n"
                "🏆 **Rankings** — top 5 / top 10 segments, worst route\n"
                "📅 **Time** — by year, busiest month, most recent year\n"
                "🔴 **Severity** — fatal, injury, property damage breakdown\n"
                "🌙 **Time of day** — night vs day crashes\n"
                "🚗 **Hit & run** — count and rate\n"
                "🏙️ **City** — crashes by city\n"
                "🛣️ **Route** — crashes by route/highway\n"
                "📍 **Segment** — crashes for a specific segment\n"
                "🟢 **Risk levels** — what High/Medium/Low/Very Low mean\n"
                "📈 **Averages** — avg crashes per segment, injury rate\n"
                "✅ **Safest** — lowest-crash segment or route")

    # ── Totals ───────────────────────────────────────────────────────────────
    if any(k in m for k in ['total crash', 'how many crash', 'number of crash', 'crash count', 'overall crash']):
        n = len(crash_df)
        years = sorted(crash_df['year'].dropna().unique().astype(int).tolist())
        return (f"**{n:,} total crashes** in the dataset "
                f"({years[0]}–{years[-1]}).")

    if any(k in m for k in ['fatal', 'death', 'killed', 'fatality']):
        total  = int(crash_df['fatalities'].sum())
        by_seg = crash_df.groupby('segment_id')['fatalities'].sum()
        worst, worst_n = by_seg.idxmax(), int(by_seg.max())
        pct = total / len(crash_df) * 100
        return (f"**{total:,} total fatalities** ({pct:.1f}% of all crashes).\n"
                f"Most fatal segment: **{_lbl(worst)}** ({worst_n:,} fatalities).")

    if any(k in m for k in ['injur', 'how many injur', 'total injur']):
        total = int(crash_df['injuries'].sum())
        rate  = total / len(crash_df)
        return (f"**{total:,} total injuries** across all crashes.\n"
                f"Average injury rate: **{rate:.2f} injuries per crash**.")

    # ── Severity breakdown ───────────────────────────────────────────────────
    if any(k in m for k in ['severity', 'severity breakdown', 'crash type', 'property damage', 'incapacitat', 'possible injury']):
        if 'severity' in crash_df.columns:
            sv   = crash_df['severity'].value_counts()
            total = len(crash_df)
            lines = [f"- **{k}**: {v:,} ({v/total*100:.1f}%)" for k, v in sv.items()]
            return "**Crash Severity Breakdown:**\n" + "\n".join(lines)
        return "Severity data not available."

    # ── Segment rankings ─────────────────────────────────────────────────────
    if any(k in m for k in ['worst segment', 'most crash', 'highest crash', 'most dangerous segment', 'deadliest segment']):
        by_seg = crash_df.groupby('segment_id').size()
        seg, cnt = by_seg.idxmax(), by_seg.max()
        fat = int(crash_df[crash_df['segment_id'] == seg]['fatalities'].sum())
        return (f"**{_lbl(seg)}** — most dangerous with **{cnt:,} crashes** "
                f"and **{fat:,} fatalities**.")

    if any(k in m for k in ['safest segment', 'least crash', 'lowest crash', 'best segment']):
        by_seg = crash_df.groupby('segment_id').size()
        seg, cnt = by_seg.idxmin(), by_seg.min()
        return f"**{_lbl(seg)}** has the fewest crashes with **{cnt:,} crash(es)**."

    if any(k in m for k in ['top 10', 'top ten']):
        top = crash_df.groupby('segment_id').size().nlargest(10)
        lines = [f"**{i+1}. {_lbl(s)}**: {c:,} crashes" for i, (s, c) in enumerate(top.items())]
        return "**Top 10 Segments by Crashes:**\n" + "\n".join(lines)

    if any(k in m for k in ['top 5', 'top five', 'top segment', 'ranking', 'ranked']):
        top = crash_df.groupby('segment_id').size().nlargest(5)
        lines = [f"**{i+1}. {_lbl(s)}**: {c:,} crashes" for i, (s, c) in enumerate(top.items())]
        return "**Top 5 Segments by Crashes:**\n" + "\n".join(lines)

    if any(k in m for k in ['average crash', 'avg crash', 'mean crash', 'crashes per segment']):
        avg = len(crash_df) / crash_df['segment_id'].nunique()
        return f"**Average crashes per segment:** {avg:.1f} crashes."

    # ── Specific segment lookup ──────────────────────────────────────────────
    if 'segment' in m and any(c.isdigit() for c in m):
        import re
        nums = re.findall(r'\d+', m)
        for num in nums:
            seg_data = crash_df[crash_df['segment_id'].astype(str) == num]
            if not seg_data.empty:
                cnt  = len(seg_data)
                fat  = int(seg_data['fatalities'].sum())
                inj  = int(seg_data['injuries'].sum())
                route = seg_data['route'].mode()[0] if 'route' in seg_data.columns else '—'
                return (f"**{_lbl(num)}:**\n"
                        f"Total crashes: **{cnt:,}**\n"
                        f"Fatalities: **{fat:,}** | Injuries: **{inj:,}**")
        return f"No data found for that segment number."

    # ── Hit & run ────────────────────────────────────────────────────────────
    if any(k in m for k in ['hit and run', 'hit & run', 'hit-and-run', 'hit run']):
        hr  = crash_df['hit_and_run'].eq('Yes').sum() if 'hit_and_run' in crash_df.columns else 0
        pct = hr / len(crash_df) * 100
        by_seg = crash_df[crash_df['hit_and_run'] == 'Yes'].groupby('segment_id').size()
        worst  = by_seg.idxmax() if not by_seg.empty else '—'
        worst_n = int(by_seg.max()) if not by_seg.empty else 0
        return (f"**{hr:,} hit-and-run crashes** — {pct:.1f}% of all crashes.\n"
                f"Worst segment: **{_lbl(worst)}** with {worst_n:,} hit-and-run cases.")

    # ── Segment count ────────────────────────────────────────────────────────
    if any(k in m for k in ['unique segment', 'how many segment', 'number of segment', 'total segment', 'segment count']):
        n = crash_df['segment_id'].nunique()
        return f"There are **{n:,} unique road segments** monitored in the dataset."

    # ── Time / year ──────────────────────────────────────────────────────────
    if any(k in m for k in ['by year', 'yearly', 'annual', 'each year', 'per year', 'year breakdown', 'trend']):
        yearly = crash_df.groupby('year').size().sort_index()
        lines  = [f"**{int(yr)}**: {cnt:,} crashes" for yr, cnt in yearly.items()]
        peak_yr = int(yearly.idxmax())
        return ("**Crashes by Year:**\n" + "\n".join(lines) +
                f"\n\nPeak year: **{peak_yr}** ({int(yearly.max()):,} crashes)")

    if any(k in m for k in ['recent year', 'latest year', 'last year', 'most recent']):
        latest = int(crash_df['year'].max())
        cnt    = int((crash_df['year'] == latest).sum())
        return f"Most recent year in the dataset: **{latest}** with **{cnt:,} crashes**."

    if any(k in m for k in ['month', 'monthly', 'busiest month', 'worst month']):
        if 'Date of Crash' in crash_df.columns:
            months = pd.to_datetime(crash_df['Date of Crash'], errors='coerce').dt.month
            mc     = months.value_counts().sort_index()
            names  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            lines  = [f"**{names[m-1]}**: {c:,}" for m, c in mc.items()]
            peak_m = names[int(mc.idxmax()) - 1]
            return ("**Crashes by Month:**\n" + "\n".join(lines) +
                    f"\n\nBusiest month: **{peak_m}**")
        return "Date information not available for monthly breakdown."

    # ── Night vs day ─────────────────────────────────────────────────────────
    if any(k in m for k in ['night', 'dark', 'day vs night', 'daytime', 'nighttime', 'light condition', 'time of day']):
        if 'Light Condition' in crash_df.columns:
            lc = crash_df['Light Condition'].value_counts()
            dark_cnt = sum(v for k, v in lc.items() if any(t in str(k).lower() for t in ['dark', 'night']))
            day_cnt  = sum(v for k, v in lc.items() if not any(t in str(k).lower() for t in ['dark', 'night']))
            total    = len(crash_df)
            return (f"**Day vs Night Crashes:**\n"
                    f"☀️ Daytime: **{day_cnt:,}** ({day_cnt/total*100:.1f}%)\n"
                    f"🌙 Nighttime/Dark: **{dark_cnt:,}** ({dark_cnt/total*100:.1f}%)")
        return "Light condition data not available."

    # ── City ─────────────────────────────────────────────────────────────────
    if any(k in m for k in ['city', 'cities', 'by city', 'which city', 'memphis']):
        if 'city' in crash_df.columns:
            by_city = crash_df.groupby('city').size().nlargest(5)
            lines   = [f"**{c}**: {n:,} crashes" for c, n in by_city.items()]
            return "**Top 5 Cities by Crashes:**\n" + "\n".join(lines)
        return "City data not available."

    # ── Route ────────────────────────────────────────────────────────────────
    if any(k in m for k in ['route', 'highway', 'interstate', 'i-240', 'i-40', 'i-55', 'which road', 'by route']):
        if 'route' in crash_df.columns:
            by_route = crash_df.groupby('route').size().nlargest(5)
            lines    = [f"**{r}**: {n:,} crashes" for r, n in by_route.items()]
            return "**Top 5 Routes by Crashes:**\n" + "\n".join(lines)
        return "Route data not available."

    # ── Risk level explanation ───────────────────────────────────────────────
    if any(k in m for k in ['risk level', 'risk mean', 'what is risk', 'high risk', 'low risk', 'very low risk', 'medium risk']):
        return ("**Risk Levels (based on daily expected crashes λ):**\n"
                "🟢 **Very Low** (λ < 0.2): Low probability of any crash. Standard patrol.\n"
                "🟡 **Low** (λ 0.2–0.5): Moderate probability. Maintain readiness.\n"
                "🟠 **Medium** (λ 0.5–1.0): Likely at least one crash. Increase vigilance.\n"
                "🔴 **High** (λ ≥ 1.0): One or more crashes highly likely. Max enforcement.")

    # ── Summary / overview ───────────────────────────────────────────────────
    if any(k in m for k in ['summary', 'overview', 'give me a summary', 'overall', 'snapshot']):
        n      = len(crash_df)
        fat    = int(crash_df['fatalities'].sum())
        inj    = int(crash_df['injuries'].sum())
        segs   = crash_df['segment_id'].nunique()
        years  = sorted(crash_df['year'].dropna().unique().astype(int).tolist())
        by_seg = crash_df.groupby('segment_id').size()
        worst  = by_seg.idxmax()
        return (f"**Dataset Summary:**\n"
                f"📅 Period: **{years[0]}–{years[-1]}**\n"
                f"💥 Total crashes: **{n:,}**\n"
                f"💀 Fatalities: **{fat:,}** | 🤕 Injuries: **{inj:,}**\n"
                f"📍 Segments monitored: **{segs:,}**\n"
                f"⚠️ Most dangerous segment: **{_lbl(worst)}**")

    return ("I didn't quite understand that. Try asking:\n"
            "- *'Give me a summary'*\n"
            "- *'Top 10 segments'*\n"
            "- *'Crashes by year'*\n"
            "- *'Night vs day crashes'*\n"
            "- *'Severity breakdown'*\n"
            "- *'Crashes in segment 12345'*\n\n"
            "Type **help** for the full topic list.")


def chatbot_forecast_response(msg, future_df, historical_df, mslink, route_name=''):
    m         = msg.lower().strip()
    today     = datetime.now().date()
    seg_label = f"{route_name} {mslink}".strip() if route_name else str(mslink)

    if future_df is None or future_df.empty:
        return "No forecast data loaded. Please select a segment first."

    # ── Greetings & help ────────────────────────────────────────────────────
    if any(k in m for k in ['hello', 'hi', 'hey', 'howdy']):
        return (f"Hi! I'm CrashBot 🤖\n\n"
                f"I have the full forecast for **{seg_label}**.\n"
                f"Ask me about today's risk, peak days, weekly summaries, lambda, "
                f"confidence intervals, and more.\n\nType **help** for all topics.")

    if any(k in m for k in ['help', 'what can you', 'what do you', 'capabilities', 'topics', 'options']):
        return ("**CrashBot can answer questions about:**\n\n"
                "📅 **Today / Tomorrow** — risk level and expected crashes\n"
                "📆 **This week / Next week** — weekly risk summary\n"
                "🔴 **Peak day** — highest risk day in the forecast\n"
                "🟢 **Safest day** — lowest risk day in the forecast\n"
                "📊 **Risk breakdown** — count of High/Medium/Low/Very Low days\n"
                "📈 **Average / Total λ** — expected crash statistics\n"
                "🎲 **Lambda** — what it means and how to interpret it\n"
                "📏 **Confidence interval** — what the upper/lower bounds mean\n"
                "🗓️ **Forecast range** — start/end dates, number of days\n"
                "📉 **Historical data** — past crash totals and busiest day\n"
                "🤖 **Model** — how the predictions are made\n"
                "🛡️ **Risk levels** — what High/Medium/Low/Very Low mean\n"
                "🗂️ **Summary** — full overview of this segment's forecast")

    # ── Today ────────────────────────────────────────────────────────────────
    if any(k in m for k in ['today', 'current risk', 'risk today', 'right now']):
        rows = future_df[future_df['date'].dt.date == today]
        if not rows.empty:
            row = rows.iloc[0]
            level, _ = get_risk_level(row['lambda'])
            ml = row.get('most_likely_crashes', '—')
            return (f"**{seg_label} — Today ({today.strftime('%b %d, %Y')}):**\n"
                    f"🔴 Risk Level: **{level}**\n"
                    f"λ (expected crashes): **{row['lambda']:.3f}**\n"
                    f"Most likely outcome: **{ml} crash(es)**\n"
                    f"Lower bound: {row.get('predicted_lower', '—')} | "
                    f"Upper bound: {row.get('predicted_upper', '—')}")
        return (f"Today ({today.strftime('%b %d')}) is outside the forecast window.\n"
                f"Forecast runs: **{future_df['date'].min().strftime('%b %d')}** → "
                f"**{future_df['date'].max().strftime('%b %d, %Y')}**")

    # ── Tomorrow ─────────────────────────────────────────────────────────────
    if any(k in m for k in ['tomorrow', 'next day', 'tomorrow risk']):
        tomorrow = today + timedelta(days=1)
        rows = future_df[future_df['date'].dt.date == tomorrow]
        if not rows.empty:
            row = rows.iloc[0]
            level, _ = get_risk_level(row['lambda'])
            return (f"**{seg_label} — Tomorrow ({tomorrow.strftime('%b %d, %Y')}):**\n"
                    f"Risk Level: **{level}**\n"
                    f"λ: **{row['lambda']:.3f}**\n"
                    f"Most likely: **{row.get('most_likely_crashes', '—')} crash(es)**")
        return "Tomorrow is outside the forecast window."

    # ── This week ────────────────────────────────────────────────────────────
    if any(k in m for k in ['this week', 'current week', 'week ahead']):
        week_end = today + timedelta(days=6)
        wdf = future_df[(future_df['date'].dt.date >= today) &
                        (future_df['date'].dt.date <= week_end)]
        if wdf.empty:
            return "No forecast data for this week."
        peak  = wdf.loc[wdf['lambda'].idxmax()]
        level, _ = get_risk_level(peak['lambda'])
        total = wdf['lambda'].sum()
        return (f"**This Week — {seg_label}:**\n"
                f"Days covered: **{len(wdf)}**\n"
                f"Total expected crashes: **{total:.2f}**\n"
                f"Peak day: **{peak['date'].strftime('%A, %b %d')}** ({level}, λ={peak['lambda']:.3f})")

    # ── Next week ────────────────────────────────────────────────────────────
    if any(k in m for k in ['next week', 'following week']):
        nw_start = today + timedelta(days=7)
        nw_end   = today + timedelta(days=13)
        wdf = future_df[(future_df['date'].dt.date >= nw_start) &
                        (future_df['date'].dt.date <= nw_end)]
        if wdf.empty:
            return "Next week is outside the forecast window."
        peak  = wdf.loc[wdf['lambda'].idxmax()]
        level, _ = get_risk_level(peak['lambda'])
        total = wdf['lambda'].sum()
        return (f"**Next Week ({nw_start.strftime('%b %d')}–{nw_end.strftime('%b %d')}) — {seg_label}:**\n"
                f"Days covered: **{len(wdf)}**\n"
                f"Total expected crashes: **{total:.2f}**\n"
                f"Peak day: **{peak['date'].strftime('%A, %b %d')}** ({level}, λ={peak['lambda']:.3f})")

    # ── Weekend ──────────────────────────────────────────────────────────────
    if any(k in m for k in ['weekend', 'saturday', 'sunday']):
        wdf = future_df[future_df['date'].dt.dayofweek >= 5]
        if wdf.empty:
            return "No weekend days in the forecast window."
        avg   = wdf['lambda'].mean()
        level, _ = get_risk_level(avg)
        return (f"**Weekend Risk — {seg_label}:**\n"
                f"Weekend days in forecast: **{len(wdf)}**\n"
                f"Average λ: **{avg:.3f}** ({level})\n"
                f"Total expected crashes: **{wdf['lambda'].sum():.2f}**")

    # ── Weekday ──────────────────────────────────────────────────────────────
    if any(k in m for k in ['weekday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
        wdf = future_df[future_df['date'].dt.dayofweek < 5]
        avg = wdf['lambda'].mean()
        level, _ = get_risk_level(avg)
        return (f"**Weekday Risk — {seg_label}:**\n"
                f"Weekdays in forecast: **{len(wdf)}**\n"
                f"Average λ: **{avg:.3f}** ({level})\n"
                f"Total expected crashes: **{wdf['lambda'].sum():.2f}**")

    # ── Peak / worst day ─────────────────────────────────────────────────────
    if any(k in m for k in ['peak', 'worst day', 'highest risk', 'most dangerous day', 'maximum', 'riskiest']):
        row      = future_df.loc[future_df['lambda'].idxmax()]
        level, _ = get_risk_level(row['lambda'])
        return (f"**Peak Risk Day — {seg_label}:**\n"
                f"📅 **{row['date'].strftime('%A, %b %d, %Y')}**\n"
                f"Risk: **{level}**\n"
                f"λ = **{row['lambda']:.3f}**\n"
                f"Most likely: **{row.get('most_likely_crashes', '—')} crash(es)**")

    # ── Safest day ───────────────────────────────────────────────────────────
    if any(k in m for k in ['safest', 'lowest risk', 'best day', 'minimum risk', 'least dangerous']):
        row      = future_df.loc[future_df['lambda'].idxmin()]
        level, _ = get_risk_level(row['lambda'])
        return (f"**Safest Day — {seg_label}:**\n"
                f"📅 **{row['date'].strftime('%A, %b %d, %Y')}**\n"
                f"Risk: **{level}**\n"
                f"λ = **{row['lambda']:.3f}**")

    # ── Risk breakdown ───────────────────────────────────────────────────────
    if any(k in m for k in ['risk breakdown', 'risk day', 'how many high', 'how many low',
                             'how many medium', 'breakdown', 'distribution']):
        if 'risk_level' in future_df.columns:
            counts = future_df['risk_level'].value_counts()
            total  = len(future_df)
            lines  = [f"- 🔴 **High**: {counts.get('High', 0)} days ({counts.get('High', 0)/total*100:.0f}%)",
                      f"- 🟠 **Medium**: {counts.get('Medium', 0)} days ({counts.get('Medium', 0)/total*100:.0f}%)",
                      f"- 🟡 **Low**: {counts.get('Low', 0)} days ({counts.get('Low', 0)/total*100:.0f}%)",
                      f"- 🟢 **Very Low**: {counts.get('Very Low', 0)} days ({counts.get('Very Low', 0)/total*100:.0f}%)"]
            return f"**Risk Day Breakdown — {seg_label} ({total} days):**\n" + "\n".join(lines)
        n = int((future_df['lambda'] >= 1.0).sum())
        return f"**{n} High-Risk days** (λ ≥ 1.0) in the forecast period."

    # ── Average / total ──────────────────────────────────────────────────────
    if any(k in m for k in ['average', 'mean', 'avg lambda', 'average lambda', 'average risk']):
        avg      = future_df['lambda'].mean()
        level, _ = get_risk_level(avg)
        mn, mx   = future_df['lambda'].min(), future_df['lambda'].max()
        return (f"**Average Daily λ — {seg_label}:**\n"
                f"Mean: **{avg:.3f}** ({level})\n"
                f"Min: {mn:.3f} | Max: {mx:.3f}")

    if any(k in m for k in ['total crash', 'total expected', 'total lambda', 'sum', 'overall expected']):
        total = future_df['lambda'].sum()
        n     = len(future_df)
        return (f"**Total Expected Crashes — {seg_label}:**\n"
                f"Over **{n} days**: Σλ = **{total:.2f}** (≈ **{round(total)} crashes**)")

    # ── Lambda explanation ───────────────────────────────────────────────────
    if any(k in m for k in ['lambda', 'what is λ', 'λ mean', 'poisson', 'expected crash']):
        return ("**Lambda (λ)** is the expected (average) number of crashes on a given day, "
                "from a Poisson statistical model.\n\n"
                "**How to read it:**\n"
                "- λ = 0.1 → ~9% chance of any crash\n"
                "- λ = 0.2 → ~18% chance of a crash\n"
                "- λ = 0.5 → ~39% chance of a crash\n"
                "- λ = 1.0 → ~63% chance of at least 1 crash\n"
                "- λ = 2.0 → ~86% chance of at least 1 crash\n\n"
                "A higher λ = higher expected crash frequency.")

    # ── Confidence interval ──────────────────────────────────────────────────
    if any(k in m for k in ['confidence', 'interval', 'upper bound', 'lower bound', 'uncertainty', '95%']):
        return ("**Confidence Interval (95%):**\n\n"
                "The upper and lower bounds represent the range within which the actual "
                "crash count is expected to fall **95 out of 100 similar days**.\n\n"
                "- **Lower bound** — optimistic scenario (fewer crashes)\n"
                "- **Upper bound** — pessimistic scenario (more crashes)\n"
                "- The interval widens further into the future as uncertainty grows.\n\n"
                "Use the upper bound for conservative resource deployment.")

    # ── How the model works ──────────────────────────────────────────────────
    if any(k in m for k in ['model', 'how is it predicted', 'how does it work', 'algorithm',
                             'machine learning', 'prediction method', 'how predict']):
        return ("**How Predictions Are Made:**\n\n"
                "The forecast uses an **ensemble of 3 ML models** (Random Forest, "
                "Gradient Boosting, Ridge Regression) combined with seasonal patterns:\n\n"
                "| Days ahead | Method |\n"
                "|---|---|\n"
                "| 1–14 | Pure ML ensemble |\n"
                "| 15–30 | ML + light seasonal blend |\n"
                "| 31–60 | Heavier seasonal blend |\n"
                "| 61+ | Historical seasonal mean |\n\n"
                "Each segment has its own trained model based on its individual crash history.")

    # ── Forecast range ───────────────────────────────────────────────────────
    if any(k in m for k in ['forecast period', 'how many day', 'date range', 'forecast range',
                             'start date', 'end date', 'when does']):
        start = future_df['date'].min().strftime('%b %d, %Y')
        end   = future_df['date'].max().strftime('%b %d, %Y')
        return (f"**Forecast Range — {seg_label}:**\n"
                f"📅 **{start}** to **{end}**\n"
                f"Total: **{len(future_df)} days**")

    # ── Historical data ──────────────────────────────────────────────────────
    if any(k in m for k in ['historical', 'past crash', 'history', 'previous crash', 'past data']):
        if historical_df is not None and not historical_df.empty:
            total    = int(historical_df['crash_count'].sum())
            avg      = historical_df['crash_count'].mean()
            peak_day = historical_df.loc[historical_df['crash_count'].idxmax(), 'date']
            zero_days = int((historical_df['crash_count'] == 0).sum())
            return (f"**Historical Data — {seg_label}:**\n"
                    f"Total recorded crashes: **{total:,}**\n"
                    f"Daily average: **{avg:.2f}**\n"
                    f"Busiest day: **{peak_day.strftime('%b %d, %Y')}**\n"
                    f"Zero-crash days: **{zero_days:,}**")
        return "No historical data available for this segment."

    # ── Risk level explanation ───────────────────────────────────────────────
    if any(k in m for k in ['risk level', 'risk mean', 'what is risk', 'high risk mean',
                             'very low mean', 'medium mean']):
        return ("**Daily Risk Levels:**\n"
                "🟢 **Very Low** (λ < 0.2): Low probability. Standard patrol.\n"
                "🟡 **Low** (λ 0.2–0.5): Moderate. Maintain readiness.\n"
                "🟠 **Medium** (λ 0.5–1.0): Likely one crash. Increase vigilance.\n"
                "🔴 **High** (λ ≥ 1.0): One or more crashes highly likely. Max enforcement.")

    # ── Summary / overview ───────────────────────────────────────────────────
    if any(k in m for k in ['summary', 'overview', 'give me a summary', 'overall', 'snapshot', 'brief']):
        avg      = future_df['lambda'].mean()
        total    = future_df['lambda'].sum()
        peak     = future_df.loc[future_df['lambda'].idxmax()]
        safe     = future_df.loc[future_df['lambda'].idxmin()]
        lv, _    = get_risk_level(avg)
        if 'risk_level' in future_df.columns:
            high_n = int((future_df['risk_level'] == 'High').sum())
        else:
            high_n = int((future_df['lambda'] >= 1.0).sum())
        return (f"**Forecast Summary — {seg_label}:**\n"
                f"📅 {future_df['date'].min().strftime('%b %d')} → "
                f"{future_df['date'].max().strftime('%b %d, %Y')} ({len(future_df)} days)\n"
                f"📈 Avg daily λ: **{avg:.3f}** ({lv})\n"
                f"💥 Total expected: **{total:.1f} crashes**\n"
                f"🔴 High-risk days: **{high_n}**\n"
                f"⚠️ Peak: **{peak['date'].strftime('%A, %b %d')}** (λ={peak['lambda']:.3f})\n"
                f"✅ Safest: **{safe['date'].strftime('%A, %b %d')}** (λ={safe['lambda']:.3f})")

    return ("I didn't quite understand that. Try:\n"
            "- *'Give me a summary'*\n"
            "- *'What is the risk today?'*\n"
            "- *'Risk this week'*\n"
            "- *'Peak risk day'*\n"
            "- *'Risk breakdown'*\n"
            "- *'How does the model work?'*\n\n"
            "Type **help** for the full topic list.")


def show_crashbot_sidebar(history_key, response_fn):
    """Render CrashBot in the sidebar. Uses a form to avoid the double-render loop."""
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 CrashBot")
        st.caption("Ask questions about the data on this page.")

        # Show last 8 messages (4 pairs) — keeps sidebar compact
        recent = st.session_state[history_key][-8:]
        for role, content in recent:
            if role == "user":
                st.markdown(
                    f'<div style="background:#dbeafe;padding:8px 10px;border-radius:8px;'
                    f'margin:4px 0;font-size:12px;"><b>You:</b><br>{content}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background:#f0fdf4;padding:8px 10px;border-radius:8px;'
                    f'margin:4px 0;font-size:12px;"><b>🤖 CrashBot:</b><br>{content}</div>',
                    unsafe_allow_html=True
                )

        if st.session_state[history_key]:
            if st.button("🗑 Clear chat", key=f"clear_{history_key}", use_container_width=True):
                st.session_state[history_key] = []
                st.rerun()

        with st.form(key=f"crashbot_form_{history_key}", clear_on_submit=True):
            user_input = st.text_input(
                "message", placeholder="Type your question...",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("➤ Ask CrashBot", use_container_width=True)

        if submitted and user_input.strip():
            reply = response_fn(user_input.strip())
            st.session_state[history_key].extend(
                [("user", user_input.strip()), ("assistant", reply)]
            )
            if len(st.session_state[history_key]) > MAX_CHAT_HISTORY * 2:
                st.session_state[history_key] = st.session_state[history_key][-MAX_CHAT_HISTORY * 2:]
            st.rerun()


# ============================================================================
# PLOTS — FORECAST PAGE
# ============================================================================
def create_historical_plot(historical_df):
    fig = go.Figure()
    if not historical_df.empty:
        hist = historical_df.sort_values('date')
        ma   = hist['crash_count'].rolling(window=7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=hist['date'], y=hist['crash_count'], mode='lines', name='Daily Crashes',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Crashes: %{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=hist['date'], y=ma, mode='lines', name='7-Day Moving Avg',
            line=dict(color='red', width=2, dash='dot'),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Avg: %{y:,.2f}<extra></extra>'
        ))
    fig.update_layout(
        title=dict(text='<b>Historical Daily Crashes</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Date</b>',              font=dict(size=20, family='Arial', color='black')),
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
    fig = go.Figure()
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
        x=future_df['date'], y=future_df['lambda'], mode='lines+markers', name='Mean Prediction (λ)',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=5, color='#ff7f0e', line=dict(color='black', width=0.8)),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>λ: %{y:,.3f}<extra></extra>'
    ))

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
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Date</b>',                      font=dict(size=20, family='Arial', color='black')),
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
    exp   = 4 if most_likely_crashes == "4+" else int(most_likely_crashes)
    lam   = row['lambda']
    level, color = get_risk_level(lam)

    suffix = " crash" if exp == 1 else " crashes"
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

    fig2 = go.Figure(go.Indicator(
        mode="gauge+number", value=prob_percent,
        title={'text': "<b>Probability of Most Likely</b>", 'font': {'size': 22, 'family': 'Arial', 'color': 'black'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'size': 16, 'color': 'black'}},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 25],   'color': '#fee2e2'}, {'range': [25, 50],  'color': '#fef3c7'},
                {'range': [50, 75],  'color': '#d1fae5'}, {'range': [75, 100], 'color': '#a7f3d0'}
            ]
        },
        number={'suffix': "%", 'font': {'size': 28, 'family': 'Arial', 'color': 'black'}}
    ))
    fig2.update_layout(height=310, margin=dict(t=100, b=10, l=20, r=20),
                       paper_bgcolor='white', plot_bgcolor='white')

    return fig1, fig2, level, color


def create_probability_pie_chart(row):
    probs  = [row['prob_0_crash'], row['prob_1_crash'], row['prob_2_crash'],
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
                   font=dict(size=22, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        height=620, showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
                    font=dict(size=16, family='Arial', color='black')),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family="Arial", size=16, color='black'),
        margin=dict(l=20, r=140, t=80, b=20)
    )
    return fig


def create_risk_calendar_heatmap(future_df):
    df = future_df.copy()
    df['dow']      = df['date'].dt.dayofweek
    df['week_num'] = ((df['date'] - df['date'].min()).dt.days // 7)

    # Ensure risk_level column exists
    if 'risk_level' not in df.columns:
        df['risk_level'] = df['lambda'].apply(
            lambda x: 'High' if x >= 1.0 else 'Medium' if x >= 0.5 else 'Low' if x >= 0.2 else 'Very Low'
        )

    pivot_lam  = df.pivot(index='week_num', columns='dow', values='lambda')
    pivot_risk = df.pivot(index='week_num', columns='dow', values='risk_level')
    pivot_date = df.pivot(index='week_num', columns='dow', values='date')

    day_names   = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    week_labels = []
    for wn in pivot_lam.index:
        first = df[df['week_num'] == wn]['date'].min()
        week_labels.append(first.strftime('%b %d'))

    # Build rich hover text: actual date + λ + risk level per cell
    hover = []
    for wn in pivot_lam.index:
        row_hover = []
        for dow in pivot_lam.columns:
            lam  = pivot_lam.loc[wn, dow]
            risk = pivot_risk.loc[wn, dow] if dow in pivot_risk.columns else '—'
            dt   = pivot_date.loc[wn, dow]
            dt_s = pd.Timestamp(dt).strftime('%a, %b %d %Y') if pd.notna(dt) else '—'
            lam_s = f"{lam:.3f}" if pd.notna(lam) else '—'
            row_hover.append(f"<b>{dt_s}</b><br>λ: {lam_s}<br>Risk: <b>{risk}</b>")
        hover.append(row_hover)

    # ── Fixed colorscale anchored to exact risk thresholds ──────────────────
    # zmax = 2.0  →  boundary fractions: Very Low ends at 0.2/2=0.10,
    #                Low ends at 0.5/2=0.25, High starts at 1.0/2=0.50
    ZMAX = 2.0
    risk_colorscale = [
        [0.000, '#28a745'],   # Very Low — green
        [0.099, '#28a745'],
        [0.100, '#ffc107'],   # Low — yellow   (λ = 0.2)
        [0.249, '#ffc107'],
        [0.250, '#fd7e14'],   # Medium — orange (λ = 0.5)
        [0.499, '#fd7e14'],
        [0.500, '#dc3545'],   # High — red      (λ = 1.0)
        [1.000, '#6b0000'],   # deep red for λ ≥ 2.0
    ]

    fig = go.Figure(go.Heatmap(
        z=pivot_lam.values,
        x=[day_names[i] for i in pivot_lam.columns],
        y=week_labels,
        colorscale=risk_colorscale,
        zmin=0,
        zmax=ZMAX,
        xgap=2,
        ygap=2,
        colorbar=dict(
            title=dict(text='<b>Risk Level</b>', font=dict(size=14, family='Arial')),
            thickness=20,
            tickvals=[0.1, 0.35, 0.75, 1.5],
            ticktext=['🟢 Very Low (λ<0.2)',
                      '🟡 Low (λ 0.2–0.5)',
                      '🟠 Medium (λ 0.5–1.0)',
                      '🔴 High (λ≥1.0)'],
            tickfont=dict(size=12, family='Arial'),
            len=0.85,
        ),
        text=hover,
        hovertemplate='%{text}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text='<b>Daily Risk Calendar</b>',
                   font=dict(size=22, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        xaxis=dict(title='<b>Day of Week</b>', tickfont=dict(size=14, family='Arial'),
                   linecolor='black', linewidth=2, mirror=True),
        yaxis=dict(title='<b>Week</b>', tickfont=dict(size=13, family='Arial'),
                   autorange='reversed', linecolor='black', linewidth=2, mirror=True),
        height=max(350, 70 * len(week_labels)),
        margin=dict(l=110, r=180, t=80, b=60),
        paper_bgcolor='white', plot_bgcolor='#444444'
    )
    return fig


# ============================================================================
# PLOTS — HOTSPOT PAGE
# ============================================================================
def create_monthly_crashes_plot(df, selected_year=None, selected_route=None, selected_segment=None):
    filtered_df = df
    if selected_year  and selected_year  != "All Years":  filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes": filtered_df = filtered_df[filtered_df['Route'] == selected_route]
    if selected_segment and selected_segment != "All Segments": filtered_df = filtered_df[filtered_df['Segment ID'] == selected_segment]
    filtered_df = filtered_df.copy()

    filtered_df['Crash Date'] = pd.to_datetime(filtered_df['Date of Crash'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['Crash Date'])
    filtered_df['Month_Num'] = filtered_df['Crash Date'].dt.month

    monthly_crashes = filtered_df.groupby('Month_Num').size().reset_index(name='Crashes')
    all_months      = pd.DataFrame({'Month_Num': range(1, 13)})
    monthly_crashes = all_months.merge(monthly_crashes, on='Month_Num', how='left').fillna(0)
    month_names     = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
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
    fig.add_shape(type="line", x0=-0.5, x1=11.5, y0=0, y1=0, line=dict(color="black", width=2))
    fig.update_layout(
        title=dict(text='<b>✨ Monthly Crash Variation</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Month</b>',         font=dict(size=20, family='Arial', color='black')),
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
    filtered_df = df
    if selected_year  and selected_year  != "All Years":  filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes": filtered_df = filtered_df[filtered_df['Route'] == selected_route]

    light_col = next((c for c in filtered_df.columns if 'light' in c.lower()), None)
    if light_col is None:
        fig = go.Figure()
        fig.add_annotation(text="Light Condition data not available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=20, color="red"))
        return fig

    day_night_data = filtered_df.groupby(['Segment ID', light_col]).size().reset_index(name='Crashes')
    pivot_data     = day_night_data.pivot(index='Segment ID', columns=light_col, values='Crashes').fillna(0)
    dark_cols      = [c for c in pivot_data.columns if any(t in str(c).lower() for t in ['dark', 'night'])]
    daylight_cols  = [c for c in pivot_data.columns if c not in dark_cols]

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
        xaxis_title=dict(text='<b>Number of Crashes</b>',              font=dict(size=20, family='Arial', color='black')),
        yaxis_title=dict(text='<b>Segment (Ranked by Total Crashes)</b>', font=dict(size=20, family='Arial', color='black')),
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
    filtered_df = df
    if selected_year  and selected_year  != "All Years":  filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes": filtered_df = filtered_df[filtered_df['Route'] == selected_route]

    crash_counts = filtered_df.groupby(['Segment ID', 'Route']).size().reset_index(name='Total Crashes')
    crash_counts = crash_counts.sort_values('Total Crashes', ascending=False).head(10)
    crash_counts['Rank'] = range(1, len(crash_counts) + 1)
    n = len(crash_counts)
    blue_gradient = [f'rgb({int(0+(173-0)*i/max(n-1,1))},{int(51+(216-51)*i/max(n-1,1))},{int(153+(230-153)*i/max(n-1,1))})' for i in range(n)]
    crash_counts['Color'] = blue_gradient

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=crash_counts['Rank'], y=crash_counts['Total Crashes'],
        marker=dict(color=crash_counts['Color'], line=dict(color='black', width=2)), width=0.6,
        text=[f"<b>{seg}</b><br><b>{count:,}</b>" for seg, count in zip(crash_counts['Segment ID'], crash_counts['Total Crashes'])],
        textposition='outside', textfont=dict(size=15, color='black', family='Arial Black'),
        cliponaxis=False,
        hovertemplate='<b>Rank %{x}</b><br>Segment: %{customdata[0]}<br>Route: %{customdata[1]}<br>Crashes: %{y:,.0f}<extra></extra>',
        customdata=crash_counts[['Segment ID', 'Route']].values
    ))
    _max1 = crash_counts['Total Crashes'].max() if len(crash_counts) else 1
    fig1.update_layout(
        title=dict(text='<b>✨ Top 10 Segments by Total Crashes</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Rank</b>',               font=dict(size=19, family='Arial', color='black')),
        yaxis_title=dict(text='<b>Number of Crashes</b>', font=dict(size=19, family='Arial', color='black')),
        height=600, template='plotly_white', showlegend=False,
        paper_bgcolor='white', plot_bgcolor='#f8f9fa',
        uniformtext=dict(minsize=11, mode='hide'),
        xaxis=dict(showgrid=False, tickfont=dict(size=16, family='Arial Black', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   tickmode='linear', tick0=1, dtick=1, range=[0.5, 10.5]),
        yaxis=dict(showgrid=True, gridwidth=1.5, gridcolor='rgba(200,200,200,0.4)',
                   tickfont=dict(size=16, family='Arial', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   zeroline=True, zerolinewidth=2, zerolinecolor='black', separatethousands=True,
                   range=[0, _max1 * 1.35]),
        margin=dict(l=80, r=40, t=90, b=80)
    )

    hit_run_df     = filtered_df[filtered_df['Hit and Run'] == 'Yes'].copy()
    hit_run_counts = hit_run_df.groupby(['Segment ID', 'Route']).size().reset_index(name='Hit and Run Cases')
    hit_run_counts = hit_run_counts.sort_values('Hit and Run Cases', ascending=False).head(10)
    hit_run_counts['Rank'] = range(1, len(hit_run_counts) + 1)
    nr = len(hit_run_counts)
    red_gradient = [f'rgb({int(139+(255-139)*i/max(nr-1,1))},{int(0+(182-0)*i/max(nr-1,1))},{int(0+(193-0)*i/max(nr-1,1))})' for i in range(nr)]
    hit_run_counts['Color'] = red_gradient

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=hit_run_counts['Rank'], y=hit_run_counts['Hit and Run Cases'],
        marker=dict(color=hit_run_counts['Color'], line=dict(color='black', width=2)), width=0.6,
        text=[f"<b>{seg}</b><br><b>{count:,}</b>" for seg, count in zip(hit_run_counts['Segment ID'], hit_run_counts['Hit and Run Cases'])],
        textposition='outside', textfont=dict(size=15, color='black', family='Arial Black'),
        cliponaxis=False,
        hovertemplate='<b>Rank %{x}</b><br>Segment: %{customdata[0]}<br>Route: %{customdata[1]}<br>Hit & Run: %{y:,.0f}<extra></extra>',
        customdata=hit_run_counts[['Segment ID', 'Route']].values
    ))
    _max2 = hit_run_counts['Hit and Run Cases'].max() if len(hit_run_counts) else 1
    fig2.update_layout(
        title=dict(text='<b>✨ Top 10 Segments by Hit and Run Cases</b>',
                   font=dict(size=24, family='Arial', color='#dc3545'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Rank</b>',                       font=dict(size=19, family='Arial', color='black')),
        yaxis_title=dict(text='<b>Number of Hit and Run Cases</b>', font=dict(size=19, family='Arial', color='black')),
        height=600, template='plotly_white', showlegend=False,
        paper_bgcolor='white', plot_bgcolor='#f8f9fa',
        uniformtext=dict(minsize=11, mode='hide'),
        xaxis=dict(showgrid=False, tickfont=dict(size=16, family='Arial Black', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   tickmode='linear', tick0=1, dtick=1, range=[0.5, 10.5]),
        yaxis=dict(showgrid=True, gridwidth=1.5, gridcolor='rgba(200,200,200,0.4)',
                   tickfont=dict(size=16, family='Arial', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   zeroline=True, zerolinewidth=2, zerolinecolor='black', separatethousands=True,
                   range=[0, _max2 * 1.35]),
        margin=dict(l=80, r=40, t=90, b=80)
    )
    return fig1, fig2


def create_fatality_ranking_plot(df, selected_year=None, selected_route=None):
    filtered_df = df
    if selected_year  and selected_year  != "All Years":  filtered_df = filtered_df[filtered_df['Year Of Crash'] == int(selected_year)]
    if selected_route and selected_route != "All Routes": filtered_df = filtered_df[filtered_df['Route'] == selected_route]

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
        marker=dict(color=fatality_counts['Color'], line=dict(color='black', width=2)), width=0.6,
        text=[f"<b>{seg}</b><br><b>{int(c):,}</b>" for seg, c in zip(fatality_counts['Segment ID'], fatality_counts['Total_Fatalities'])],
        textposition='outside', textfont=dict(size=15, color='black', family='Arial Black'),
        cliponaxis=False,
        hovertemplate='<b>Rank %{x}</b><br>Segment: %{customdata[0]}<br>Route: %{customdata[1]}<br>Fatalities: %{y:,.0f}<extra></extra>',
        customdata=fatality_counts[['Segment ID', 'Route']].values
    ))
    _maxf = fatality_counts['Total_Fatalities'].max() if len(fatality_counts) else 1
    fig.update_layout(
        title=dict(text='<b>✨ Top 10 Segments by Fatalities</b>',
                   font=dict(size=24, family='Arial', color='#dc3545'), x=0.5, xanchor='center'),
        xaxis_title=dict(text='<b>Rank</b>',                   font=dict(size=19, family='Arial', color='black')),
        yaxis_title=dict(text='<b>Number of Fatalities</b>', font=dict(size=19, family='Arial', color='black')),
        height=600, template='plotly_white', showlegend=False,
        paper_bgcolor='white', plot_bgcolor='#f8f9fa',
        uniformtext=dict(minsize=11, mode='hide'),
        xaxis=dict(showgrid=False, tickfont=dict(size=16, family='Arial Black', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   tickmode='linear', tick0=1, dtick=1, range=[0.5, 10.5]),
        yaxis=dict(showgrid=True, gridwidth=1.5, gridcolor='rgba(200,200,200,0.4)',
                   tickfont=dict(size=16, family='Arial', color='black'),
                   linecolor='black', linewidth=2.5, mirror=True,
                   zeroline=True, zerolinewidth=2, zerolinecolor='black', separatethousands=True,
                   range=[0, _maxf * 1.35]),
        margin=dict(l=80, r=40, t=90, b=80)
    )
    return fig


# ============================================================================
# HOTSPOT MAP PLOTS
# ============================================================================
def create_crash_frequency_heatmap(df, selected_year=None, selected_route=None):
    filtered_df = df
    if selected_year  and selected_year  != "All Years":  filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
    if selected_route and selected_route != "All Routes": filtered_df = filtered_df[filtered_df['route'] == selected_route]

    total = len(filtered_df)
    if total > MAX_MAP_POINTS:
        filtered_df = filtered_df.sample(MAX_MAP_POINTS, random_state=42)
    hover_text = (
        '<b>Segment:</b> '   + filtered_df['segment_id'].astype(str) + '<br>' +
        '<b>City:</b> '      + filtered_df['city'].astype(str)       + '<br>' +
        '<b>Route:</b> '     + filtered_df['route'].astype(str)      + '<br>' +
        '<b>Injuries:</b> '  + filtered_df['injuries'].astype(str)   + '<br>' +
        '<b>Fatalities:</b> '+ filtered_df['fatalities'].astype(str) + '<br>' +
        '<b>Year:</b> '      + filtered_df['year'].astype(str)
    )
    fig = go.Figure()
    fig.add_trace(go.Densitymapbox(
        lat=filtered_df['latitude'], lon=filtered_df['longitude'],
        radius=15, colorscale='Reds', showscale=True,
        text=hover_text, hovertemplate='%{text}<extra></extra>',
        opacity=0.7,
        colorbar=dict(title="<b>Density</b>", thickness=15, len=0.7)
    ))
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=35.15, lon=-90.05), zoom=9),
        title=dict(text='<b>✨ Crash Frequency Heatmap</b>',
                   font=dict(size=24, family='Arial', color='#1f77b4'), x=0.5, xanchor='center'),
        height=700, margin=dict(l=0, r=0, t=60, b=0), paper_bgcolor='white', hovermode='closest'
    )
    return fig


def create_severity_scatter_map(df, selected_year=None, selected_route=None):
    filtered_df = df
    if selected_year  and selected_year  != "All Years":  filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
    if selected_route and selected_route != "All Routes": filtered_df = filtered_df[filtered_df['route'] == selected_route]

    if len(filtered_df) > MAX_MAP_POINTS:
        # Always keep fatal & incapacitating crashes; sample from the rest
        priority = filtered_df['severity'].isin(['Fatal Injury', 'Incapacitating Injury'])
        keep     = filtered_df[priority]
        rest     = filtered_df[~priority]
        remaining = max(0, MAX_MAP_POINTS - len(keep))
        if remaining > 0 and len(rest) > remaining:
            rest = rest.sample(remaining, random_state=42)
        filtered_df = pd.concat([keep, rest])

    severity_colors = {
        'Fatal Injury': '#96092B', 'Incapacitating Injury': '#FF4500',
        'Non-Incapacitating Injury': '#FFD700', 'Possible Injury': '#32CD32',
        'Property Damage Only': '#1E90FF'
    }
    fig = px.scatter_mapbox(
        filtered_df, lat='latitude', lon='longitude',
        color='severity', color_discrete_map=severity_colors,
        hover_name='segment_id',
        hover_data={'latitude': False, 'longitude': False,
                    'route': True, 'severity': True, 'fatalities': True, 'year': True},
        center=dict(lat=35.15, lon=-90.05), zoom=9,
        mapbox_style="open-street-map", height=700, opacity=0.85
    )
    fig.update_traces(marker_size=10)
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
    filtered_df = df
    if selected_year  and selected_year  != "All Years":  filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
    if selected_route and selected_route != "All Routes": filtered_df = filtered_df[filtered_df['route'] == selected_route]

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

    # ---- Check if CSV has been uploaded via sidebar ----
    if st.session_state.get('crash_df') is None:
        st.warning("⬅️ Please upload the crash CSV file using the uploader in the sidebar to begin analysis.")
        return

    crash_df = st.session_state.crash_df

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🔅 Total Crashes",    f"{len(crash_df):,}")
    with c2: st.metric("🔅 Total Fatalities",  f"{int(crash_df['fatalities'].sum()):,}")
    with c3: st.metric("🔅 Total Injuries",    f"{int(crash_df['injuries'].sum()):,}")
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

    map_choice = st.selectbox(
        "Select Map View",
        ["🔅 Frequency Heatmap", "🔅 Severity Map", "🔅 Top Hotspots"],
        key="map_choice"
    )

    if map_choice == "🔅 Frequency Heatmap":
        st.markdown("#### 🔅 Crash Density Heatmap")
        st.markdown("*Red zones indicate areas with the highest concentration of crashes.*")
        st.plotly_chart(create_crash_frequency_heatmap(crash_df, selected_year, selected_route),
                        use_container_width=True)

    elif map_choice == "🔅 Severity Map":
        st.markdown("#### 🔅 Crash Severity Distribution")
        st.markdown("*Each point represents a crash, coloured by severity level.*")
        st.plotly_chart(create_severity_scatter_map(crash_df, selected_year, selected_route),
                        use_container_width=True)

    elif map_choice == "🔅 Top Hotspots":
        st.markdown(f"#### 🔅 Top {top_n} Highest-Risk Segments")
        st.plotly_chart(create_segment_hotspot_map(crash_df, selected_year, selected_route, top_n),
                        use_container_width=True)

    # ── Summary table always visible (below whichever map is selected) ────────
    st.markdown("---")
    st.markdown(f"##### 📋 Top {top_n} Segments Summary Table")

    _route_map = (crash_df[['segment_id', 'route']]
                  .drop_duplicates('segment_id')
                  .assign(segment_id=lambda d: d['segment_id'].astype(str))
                  .set_index('segment_id')['route']
                  .to_dict())

    _filt = crash_df
    if selected_year  != "All Years":  _filt = _filt[_filt['year']  == int(selected_year)]
    if selected_route != "All Routes": _filt = _filt[_filt['route'] == selected_route]

    seg_stats = (
        _filt.groupby('segment_id', observed=True)
        .agg(total_crashes=('severity', 'count'),
             fatalities=('fatalities', 'sum'),
             injuries=('injuries', 'sum'))
        .reset_index()
        .nlargest(top_n, 'total_crashes')
    )
    seg_stats['rank'] = range(1, len(seg_stats) + 1)
    seg_stats['segment_id'] = seg_stats['segment_id'].astype(str)
    seg_stats['Segment'] = seg_stats['segment_id'].apply(
        lambda sid: f"{_route_map.get(sid, '')} {sid}".strip()
    )
    seg_stats = seg_stats.rename(columns={
        'total_crashes': 'Total Crashes',
        'fatalities': 'Fatalities',
        'injuries': 'Injuries',
        'rank': 'Rank'
    })[['Rank', 'Segment', 'Total Crashes', 'Fatalities', 'Injuries']]
    seg_stats['Total Crashes'] = seg_stats['Total Crashes'].fillna(0).apply(lambda x: f"{int(x):,}")
    seg_stats['Fatalities']    = seg_stats['Fatalities'].fillna(0).apply(lambda x: f"{int(x):,}")
    seg_stats['Injuries']      = seg_stats['Injuries'].fillna(0).apply(lambda x: f"{int(x):,}")

    if seg_stats.empty:
        st.info("No segment data available for the selected filters.")
    else:
        TH = ("background-color:#1f77b4;color:white;font-size:20px;font-weight:bold;"
              "padding:14px 18px;text-align:center;border:1px solid #155a8a;")
        header_html = "".join(f'<th style="{TH}">{c}</th>' for c in seg_stats.columns)
        rows_html = ""
        for i, row in enumerate(seg_stats.itertuples(index=False)):
            bg  = "#f4f8ff" if i % 2 == 0 else "#ffffff"
            TD  = (f"background-color:{bg};font-size:18px;"
                   "padding:12px 18px;text-align:center;border-bottom:1px solid #d0d0d0;")
            cells = "".join(f'<td style="{TD}">{v}</td>' for v in row)
            rows_html += f"<tr>{cells}</tr>"
        table_html = (
            '<table style="width:100%;border-collapse:collapse;font-family:Arial;">'
            f"<thead><tr>{header_html}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔅 Historical Background of Crash Data Analysis")

    _needed = ['year', 'route', 'segment_id', 'hit_and_run', 'Date of Crash', 'Light Condition', 'fatalities']
    analysis_df = crash_df[[c for c in _needed if c in crash_df.columns]].rename(columns={
        'year': 'Year Of Crash', 'route': 'Route',
        'segment_id': 'Segment ID', 'hit_and_run': 'Hit and Run'
    })

    st.markdown("#### 🔅 Monthly Crash Variation")
    st.plotly_chart(create_monthly_crashes_plot(analysis_df, selected_year, selected_route),
                    use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔅 Day vs Night Crash Comparison")
    st.plotly_chart(create_day_night_crashes_plot(analysis_df, selected_year, selected_route),
                    use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔅 Top 10 Segment Rankings")
    fig_crash, fig_hitrun = create_segment_ranking_plots(analysis_df, selected_year, selected_route)
    fig_fat               = create_fatality_ranking_plot(analysis_df, selected_year, selected_route)
    st.plotly_chart(fig_crash,  use_container_width=True)
    st.plotly_chart(fig_hitrun, use_container_width=True)
    st.plotly_chart(fig_fat,    use_container_width=True)

    show_crashbot_sidebar(
        "chat_maps",
        lambda msg: chatbot_crash_response(msg, st.session_state.crash_df)
    )

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
    st.title("🏎️ Daily Traffic Crash Risk Prediction - Shelby County")

    mslinks = discover_mslinks()
    if not mslinks:
        st.warning(f"No prediction outputs found in `{BASE_OUTPUT_DIR}`. "
                   "Run the pipeline first.")
        return

    # Build MSLINK → route map from the uploaded crash_df (if available)
    mslink_route_map = {}
    if st.session_state.get('crash_df') is not None:
        mslink_route_map = build_mslink_route_map(st.session_state.crash_df)

    route_groups: dict = {"All Routes": mslinks}
    for ml in mslinks:
        route = mslink_route_map.get(str(ml), "Unknown")
        route_groups.setdefault(route, []).append(ml)

    st.markdown("### 🔎 Filter Options")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_route = st.selectbox("Select Route", list(route_groups.keys()), key="fc_route")
    with col_f2:
        available_mslinks = route_groups[selected_route]
        selected_mslink   = st.selectbox(
            "Select Segment", available_mslinks,
            format_func=lambda m: f"Segment {m}  ({mslink_route_map.get(str(m), '—')})",
            key="fc_mslink"
        )

    future_df     = load_segment_data(selected_mslink)
    historical_df = load_historical_data(selected_mslink)

    if future_df is None or future_df.empty:
        st.error(f"No prediction data found for Segment {selected_mslink}.")
        st.info(f"Expected: `{segment_folder(selected_mslink)}/data/MSLINK_{selected_mslink}_future_predictions_with_risk.csv`")
        return

    route_label = mslink_route_map.get(str(selected_mslink), "")
    st.markdown(f"### 🔅 Segment **{selected_mslink}** &nbsp;|&nbsp; {route_label} &nbsp;|&nbsp; {len(future_df)}-day forecast")

    min_date = future_df['date'].min().date()
    max_date = future_df['date'].max().date()
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Forecast Start", value=min_date,
                                   min_value=min_date, max_value=max_date, key="fc_start")
    with col_d2:
        end_date   = st.date_input("Forecast End",   value=max_date,
                                   min_value=min_date, max_value=max_date, key="fc_end")

    mask        = (future_df['date'].dt.date >= start_date) & (future_df['date'].dt.date <= end_date)
    future_filt = future_df[mask].copy()

    if future_filt.empty:
        st.warning("No forecast data in the selected date range.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("✨ Total Expected λ", f"{future_filt['lambda'].sum():.2f}")
    with c2: st.metric("✨ Daily Average λ",   f"{future_filt['lambda'].mean():.3f}")
    with c3:
        peak = future_filt.loc[future_filt['lambda'].idxmax()]
        st.metric("✨ Peak Risk Day", peak['date'].strftime("%b %d, %Y"), delta=f"λ = {peak['lambda']:.3f}")
    with c4:
        high_days = int((future_filt['risk_level'] == 'High').sum()) if 'risk_level' in future_filt.columns else 0
        st.metric("✨ High-Risk Days", str(high_days))

    st.markdown("---")
    st.subheader("🔅 Historical Daily Crashes")
    if not historical_df.empty:
        hist_years = sorted(historical_df['date'].dt.year.unique().tolist())
        sel_years  = st.multiselect("Filter Historical Years", hist_years, default=hist_years,
                                    key="hist_year_filter")
        hist_filt  = historical_df[historical_df['date'].dt.year.isin(sel_years)] if sel_years else historical_df
        st.markdown("""<div style="border:1px solid #1f77b4; border-radius:10px; padding:20px;
                        background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">""",
                    unsafe_allow_html=True)
        st.plotly_chart(create_historical_plot(hist_filt), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No historical data available for this Segment.")

    st.markdown("---")
    st.subheader("🔅 Daily Probabilistic Crash Forecast")
    st.markdown("""<div style="border:1px solid #1f77b4; border-radius:10px; padding:20px;
                    background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">""",
                unsafe_allow_html=True)
    st.plotly_chart(create_forecast_plot(future_filt), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔅 Daily Risk Calendar")
    st.markdown("""<div style="border:1px solid #1f77b4; border-radius:10px; padding:20px;
                    background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">""",
                unsafe_allow_html=True)
    st.plotly_chart(create_risk_calendar_heatmap(future_filt), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔅 In-Depth Daily Risk Assessment")
    day_labels = [r['date'].strftime('%A, %b %d, %Y') for _, r in future_filt.iterrows()]
    chosen     = st.selectbox("Select Day for Detailed Analysis", day_labels, index=0, key="day_detail_sel")
    row        = future_filt.iloc[day_labels.index(chosen)]

    risk_level, risk_color = get_risk_level(row['lambda'])
    st.markdown(f"""
    <div style="text-align:center; margin:20px 0;">
        <span style="display:inline-block; padding:14px 40px; font-size:22px;
                     font-weight:bold; color:white; background:{risk_color};
                     border-radius:50px; box-shadow:0 8px 20px {risk_color}40;
                     text-transform:uppercase; letter-spacing:1.5px;">{risk_level}</span>
    </div>""", unsafe_allow_html=True)

    md1, md2, md3, md4 = st.columns(4)
    with md1: st.metric("λ (Expected Crashes)", f"{row['lambda']:.4f}")
    with md2: st.metric("Lower Bound (95%)",     str(int(row.get('predicted_lower', 0))))
    with md3: st.metric("Upper Bound (95%)",     str(int(row.get('predicted_upper', row['lambda'] * 2))))
    with md4: st.metric("Forecast Method",       str(row.get('method', '—')))

    st.markdown("<br>", unsafe_allow_html=True)
    colL, colR = st.columns(2)
    with colL:
        g1, g2, _, _ = create_dual_gauges(row)
        st.markdown("""<div style="border:1px solid #1f77b4; border-radius:10px; padding:15px;
                        background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1); margin-bottom:20px;">""",
                    unsafe_allow_html=True)
        st.plotly_chart(g1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""<div style="border:1px solid #1f77b4; border-radius:10px; padding:15px;
                        background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">""",
                    unsafe_allow_html=True)
        st.plotly_chart(g2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with colR:
        st.markdown("""<div style="border:1px solid #1f77b4; border-radius:10px; padding:15px;
                        background-color:white; box-shadow:0 4px 6px rgba(0,0,0,0.1);">""",
                    unsafe_allow_html=True)
        st.plotly_chart(create_probability_pie_chart(row), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Uncertainty Breakdown — commented out for now ──────────────────────────
    # st.markdown("---")
    # st.subheader("🔅 Uncertainty Breakdown")
    # ucols = ['model_uncertainty', 'residual_uncertainty', 'total_uncertainty']
    # if all(c in row.index for c in ucols):
    #     uc1, uc2, uc3 = st.columns(3)
    #     with uc1: st.metric("Model Uncertainty (σ_model)",    f"{row['model_uncertainty']:.4f}")
    #     with uc2: st.metric("Residual Uncertainty (σ_resid)", f"{row['residual_uncertainty']:.4f}")
    #     with uc3: st.metric("Total Uncertainty (σ_total)",    f"{row['total_uncertainty']:.4f}")

    # ── Crash Probability Table — commented out for now ────────────────────────
    # st.markdown("---")
    # st.subheader("🔅 Crash Probability Table")
    # prob_cols = {
    #     'P(0 crashes)':  'prob_0_crash',  'P(1 crash)':    'prob_1_crash',
    #     'P(2 crashes)':  'prob_2_crash',  'P(3 crashes)':  'prob_3_crash',
    #     'P(≥4 crashes)': 'prob_ge4_crash',
    # }
    # prob_data = {k: f"{row.get(v, 0):.1f}%" for k, v in prob_cols.items()}
    # st.dataframe(pd.DataFrame(prob_data, index=[chosen]).T.rename(columns={chosen: 'Probability'}),
    #              use_container_width=True)

    # ── Full Forecast Table — commented out for now ────────────────────────────
    # st.markdown("---")
    # st.subheader("📋 Full Forecast Table")
    # _tbl_cols = ['date', 'lambda', 'predicted_lower', 'predicted_upper', 'risk_level',
    #              'method', 'most_likely_crashes']
    # _show = [c for c in _tbl_cols if c in future_filt.columns]
    # tbl = future_filt[_show].copy()
    # tbl['date']   = tbl['date'].dt.strftime('%Y-%m-%d')
    # if 'lambda' in tbl.columns:
    #     tbl['lambda'] = tbl['lambda'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '—')
    # for _c in ['predicted_lower', 'predicted_upper', 'most_likely_crashes']:
    #     if _c in tbl.columns:
    #         tbl[_c] = tbl[_c].apply(lambda x: str(int(x)) if pd.notna(x) else '—')
    # col_labels = {
    #     'date': 'Date', 'lambda': 'λ (Expected)', 'predicted_lower': 'Lower (95%)',
    #     'predicted_upper': 'Upper (95%)', 'risk_level': 'Risk Level',
    #     'method': 'Method', 'most_likely_crashes': 'Most Likely Crashes'
    # }
    # tbl = tbl.rename(columns={k: v for k, v in col_labels.items() if k in tbl.columns})
    # TH = ("background-color:#1f77b4;color:white;font-size:16px;font-weight:bold;"
    #       "padding:10px 14px;text-align:center;border:1px solid #155a8a;white-space:nowrap;")
    # _risk_colors = {'High': '#ffe0e0', 'Medium': '#fff3cd', 'Low': '#fff9e6', 'Very Low': '#e6f9ee'}
    # header_html = "".join(f'<th style="{TH}">{c}</th>' for c in tbl.columns)
    # rows_html = ""
    # for i, row_t in enumerate(tbl.itertuples(index=False)):
    #     vals = list(row_t)
    #     risk_val = vals[tbl.columns.tolist().index('Risk Level')] if 'Risk Level' in tbl.columns.tolist() else ''
    #     row_bg = _risk_colors.get(str(risk_val), ('#f4f8ff' if i % 2 == 0 else '#ffffff'))
    #     cells = "".join(
    #         f'<td style="background-color:{row_bg};font-size:15px;font-weight:bold;'
    #         f'padding:9px 14px;text-align:center;border-bottom:1px solid #d0d0d0;">{v}</td>'
    #         for v in vals
    #     )
    #     rows_html += f"<tr>{cells}</tr>"
    # forecast_table_html = (
    #     '<div style="overflow-x:auto;">'
    #     '<table style="width:100%;border-collapse:collapse;font-family:Arial;">'
    #     f"<thead><tr>{header_html}</tr></thead>"
    #     f"<tbody>{rows_html}</tbody>"
    #     "</table></div>"
    # )
    # st.markdown(forecast_table_html, unsafe_allow_html=True)

    show_crashbot_sidebar(
        "chat_forecast",
        lambda msg: chatbot_forecast_response(
            msg, future_df, historical_df, selected_mslink,
            route_name=mslink_route_map.get(str(selected_mslink), '')
        )
    )


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
        st.markdown("The calendar heatmap shows the forecast period by day-of-week and week. Greener = lower risk, Redder = higher risk.")
    with st.expander("Probability Distribution Pie Chart"):
        st.markdown("Shows Poisson probability of 0, 1, 2, 3, or 4+ crashes on the selected day. Most-likely outcome is highlighted.")
    with st.expander("Uncertainty Breakdown"):
        st.markdown("""
        - **σ_model** — disagreement between RF, GBR, Ridge ensemble models
        - **σ_resid** — historical residual variance from training fit
        - **σ_total** — combined uncertainty for 95% CI; grows with forecast horizon
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
    with st.expander("How is Segment used?"):
        st.markdown("Each Segment is a unique road section identified by its MSLINK ID. Each Segment gets its own model trained on its own crash history.")

    st.markdown("---")
    st.info("**Support:** ctiermemphis@gmail.com | Mon–Fri 9 AM–5 PM CST | C-TIER, The University of Memphis")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":

    # ---- Session state defaults ----
    if 'authenticated'      not in st.session_state: st.session_state.authenticated      = False
    if 'crash_df'           not in st.session_state: st.session_state.crash_df           = None
    if 'crash_data_loaded'  not in st.session_state: st.session_state.crash_data_loaded  = False
    if 'chat_maps'          not in st.session_state: st.session_state.chat_maps          = []
    if 'chat_forecast'      not in st.session_state: st.session_state.chat_forecast      = []

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

            # ----------------------------------------------------------------
            # KEY CHANGE: st.file_uploader in the sidebar
            # Replaces the hardcoded path used by load_crash_shapefile_data()
            # ----------------------------------------------------------------
            st.markdown("### 📂 Upload Crash Data")
            uploaded_file = st.file_uploader(
                "Upload crash CSV",
                type=["csv"],
                help="Required columns: GPS Coordi, GPS Coor_1, MSLINK, Type of Cr, "
                     "Year Of Cr, Total Kill, Total Inj, Hit and Ru, RTE_NME, "
                     "CNTY_SEAT, Date of Cr, Light Cond"
            )

            if uploaded_file is not None:
                if not st.session_state.crash_data_loaded:
                    with st.spinner("Loading crash data..."):
                        df = load_crash_shapefile_data(uploaded_file)
                        if df is not None:
                            st.session_state.crash_df          = df
                            st.session_state.crash_data_loaded = True
                            st.success(f"✅ Loaded {len(df):,} records")
                else:
                    st.success(f"✅ {len(st.session_state.crash_df):,} records loaded")
            else:
                # Reset if user removes the file
                st.session_state.crash_df         = None
                st.session_state.crash_data_loaded = False
                st.info("⬆️ Upload CSV to enable maps & route filters")

            st.markdown("---")
            st.markdown("""
            <div class="about-box">
                <b>About SAFE TN</b><br><br>
                SAFE TN (<i>Safety Analytics & Forecasting Environment for Tennessee</i>) is a
                probabilistic crash-risk prediction tool developed by the
                <b>Center for Transportation Innovation, Education and Research (C-TIER)</b>
                at The University of Memphis.<br><br>
                Using advanced machine-learning, it delivers <b>daily</b> crash predictions
                with 95% confidence intervals for Shelby County.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="about-box">
                <b>About C-TIER</b><br><br>
                The <i>Center for Transportation Innovation, Education and Research (C-TIER)</i>
                at The University of Memphis developed this tool to enhance traffic safety
                across Tennessee by integrating real-time traffic and crash data.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.info("**Support**\nctiermemphis@gmail.com")
            st.write(f"**👤 Logged in as:** {st.session_state.username}")
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                logout()

        if   page == "Interactive Crash Maps":   show_hotspot_maps_page()
        elif page == "Daily Crash Prediction":   show_forecast_page()
        elif page == "Help & Guide":             show_help_page()
