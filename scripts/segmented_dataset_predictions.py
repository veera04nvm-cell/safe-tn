"""
DAILY CRASH PREDICTION PIPELINE
================================
Adapted from the weekly QuantileEnsemble forecasting code.
- Reads raw crash CSV (with MSLINK as segment ID)
- Prepares daily aggregated dataset with temporal, lag, and rolling features
- Trains RF + GBR + Ridge ensemble per segment
- Produces daily forecasts with Poisson probabilities, confidence intervals,
  risk levels, and uncertainty quantification

Usage:
    python daily_crash_prediction_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
import os
import gc
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION  — edit these paths for your machine
# ============================================================================
INPUT_PATH  = 'data/Unlocked_Segmented_2022_2025_Crashes_Final.csv'
BASE_OUTPUT_DIR = 'outputs/daily_risk_score/'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

MIN_CRASHES_PER_SEGMENT = 30     # skip segments with too few crashes
FORECAST_DAYS           = 90     # how many days into the future to predict

# --- 3-way chronological split ---
TEST_DAYS               = 30     # last 30 days  → final held-out test
VAL_DAYS                = 60     # preceding 60 days → validation / tuning
                                 # everything before → training

# ============================================================================
# STEP 1 — LOAD & CLEAN RAW DATA
# ============================================================================

def load_raw_data(filepath):
    """Load crash-level CSV and parse date/time"""
    print("=" * 70)
    print("LOADING RAW CRASH DATA")
    print("=" * 70)

    USE_COLS = [
        'MSLINK', 'Date of Cr', 'Time of Cr',
        'Total Kill', 'Total Inj', 'Total Veh',
        'Weather Co', 'Light Cond', 'Spd Limit', 'No. Lns.'
    ]
    df = pd.read_csv(filepath, usecols=USE_COLS, low_memory=False)
    print(f"Loaded {len(df):,} crash records")

    # --- filter segments ---
    seg_counts = df['MSLINK'].value_counts()
    keep = seg_counts[seg_counts >= MIN_CRASHES_PER_SEGMENT].index
    df = df[df['MSLINK'].isin(keep)].copy()
    print(f"Segments with >= {MIN_CRASHES_PER_SEGMENT} crashes: {len(keep):,}")
    print(f"Remaining records: {len(df):,}")

    # --- parse date ---
    df['date'] = pd.to_datetime(df['Date of Cr'], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['date']).copy()

    # --- parse hour from military time ---
    df['hour'] = df['Time of Cr'].fillna(-100).astype(int) // 100
    df.loc[df['hour'] < 0, 'hour'] = np.nan
    df.loc[df['hour'] > 23, 'hour'] = 23

    # --- per-crash binary flags ---
    rush  = {7, 8, 9, 16, 17, 18}
    night = {22, 23, 0, 1, 2, 3, 4, 5}
    peak  = {15, 16, 17, 18}

    df['is_rush_hour']      = df['hour'].isin(rush).astype(float)
    df['is_night']          = df['hour'].isin(night).astype(float)
    df['is_peak_crash_hour']= df['hour'].isin(peak).astype(float)
    for c in ['is_rush_hour','is_night','is_peak_crash_hour']:
        df.loc[df['hour'].isna(), c] = np.nan

    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_fatal']   = (df['Total Kill'] > 0).astype(int)
    df['is_injury']  = (df['Total Inj'] > 0).astype(int)

    wc = df['Weather Co'].str.strip().str.lower()
    df['is_rain'] = wc.eq('rain').astype(int)
    df['is_adverse_weather'] = wc.isin(
        ['rain','snow','fog','sleet/hail','blowing snow',
         'severe cross-winds','blowing sand/soil/dirt','smog/smoke']
    ).astype(int)
    df['is_dark'] = df['Light Cond'].str.strip().str.lower().str.contains(
        'dark', na=False).astype(int)

    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# ============================================================================
# STEP 2 — BUILD DAILY DATASET FOR ONE SEGMENT
# ============================================================================

def build_daily_segment(crash_df, mslink):
    """
    For a single segment:
      1. aggregate crashes to daily
      2. fill zero-crash days
      3. add temporal + lag + rolling features
    Returns a ready-to-model DataFrame.
    """
    seg = crash_df[crash_df['MSLINK'] == mslink].copy()

    # --- daily aggregation ---
    daily_agg = seg.groupby('date').agg(
        crash_count               = ('date', 'size'),
        rush_hour_crashes         = ('is_rush_hour', 'sum'),
        night_crashes             = ('is_night', 'sum'),
        peak_crash_hour_crashes   = ('is_peak_crash_hour', 'sum'),
        total_killed              = ('Total Kill', 'sum'),
        total_injured             = ('Total Inj', 'sum'),
        fatal_crashes             = ('is_fatal', 'sum'),
        injury_crashes            = ('is_injury', 'sum'),
        total_vehicles            = ('Total Veh', 'sum'),
        rain_crashes              = ('is_rain', 'sum'),
        adverse_weather_crashes   = ('is_adverse_weather', 'sum'),
        dark_crashes              = ('is_dark', 'sum'),
    ).reset_index()

    # --- full date spine (fill 0-crash days) ---
    date_range = pd.date_range(seg['date'].min(), seg['date'].max(), freq='D')
    spine = pd.DataFrame({'date': date_range})
    df = spine.merge(daily_agg, on='date', how='left')

    fill_cols = [c for c in df.columns if c != 'date']
    df[fill_cols] = df[fill_cols].fillna(0).astype(int)

    # --- temporal / calendar features ---
    df['year']          = df['date'].dt.year
    df['month']         = df['date'].dt.month
    df['day']           = df['date'].dt.day
    df['day_of_week']   = df['date'].dt.dayofweek
    df['day_of_year']   = df['date'].dt.dayofyear
    df['week_of_year']  = df['date'].dt.isocalendar().week.astype(int)
    df['quarter']       = df['date'].dt.quarter
    df['is_weekend']    = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_monday']     = (df['day_of_week'] == 0).astype(int)
    df['is_friday']     = (df['day_of_week'] == 4).astype(int)

    # cyclical encodings
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['week_sin']        = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos']        = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['month_sin']       = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']       = np.cos(2 * np.pi * df['month'] / 12)

    # --- lag features ---
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f'crashes_lag_{lag}d'] = df['crash_count'].shift(lag)

    # --- rolling features ---
    for w in [3, 7, 14, 28, 56, 84]:
        df[f'crashes_rolling_mean_{w}d'] = df['crash_count'].rolling(w, min_periods=1).mean()
        df[f'crashes_rolling_std_{w}d']  = df['crash_count'].rolling(w, min_periods=1).std()
        df[f'crashes_rolling_sum_{w}d']  = df['crash_count'].rolling(w, min_periods=1).sum()
        df[f'crashes_rolling_max_{w}d']  = df['crash_count'].rolling(w, min_periods=1).max()

    # --- change features ---
    df['crash_change_1d']  = df['crash_count'].diff(1)
    df['crash_change_7d']  = df['crash_count'].diff(7)
    df['crash_change_28d'] = df['crash_count'].diff(28)

    # fill NaN lags/rolling with 0
    lag_roll = [c for c in df.columns if 'lag_' in c or 'rolling_' in c or 'crash_change' in c]
    df[lag_roll] = df[lag_roll].fillna(0)

    # --- segment-level static features ---
    n_total = len(seg)
    df['total_crashes_hist'] = n_total
    df['pct_rush_hour']  = seg['is_rush_hour'].sum()  / max(n_total, 1)
    df['pct_night']      = seg['is_night'].sum()       / max(n_total, 1)
    df['pct_peak_hour']  = seg['is_peak_crash_hour'].sum() / max(n_total, 1)
    df['pct_weekend']    = seg['is_weekend'].sum()     / max(n_total, 1)
    df['pct_rain']       = seg['is_rain'].sum()        / max(n_total, 1)
    df['pct_dark']       = seg['is_dark'].sum()        / max(n_total, 1)
    df['avg_spd_limit']  = seg['Spd Limit'].mean()
    df['avg_lanes']      = seg['No. Lns.'].mean()

    return df


# ============================================================================
# STEP 3 — QUANTILE ENSEMBLE (same architecture, daily)
# ============================================================================

class QuantileEnsemble:
    """RF + GBR + Ridge ensemble with dual-uncertainty intervals"""

    def __init__(self):
        self.models = {}
        self.residual_std = 0.0

    def fit(self, X, y):
        for name, cls in [('rf', RandomForestRegressor),
                          ('gb', GradientBoostingRegressor),
                          ('ridge', Ridge)]:
            self.models[name] = cls(random_state=42)
            self.models[name].fit(X, y)

        preds = np.mean([m.predict(X) for m in self.models.values()], axis=0)
        self.residuals = y.values - preds
        self.residual_std = np.std(self.residuals)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models.values()], axis=0)

    def predict_interval(self, X):
        all_preds = np.array([m.predict(X) for m in self.models.values()])
        mean_pred = np.mean(all_preds, axis=0)
        model_std = np.std(all_preds, axis=0)
        total_std = np.sqrt(self.residual_std**2 + model_std**2)
        lower = np.maximum(mean_pred - 1.96 * total_std, 0)
        upper = np.maximum(mean_pred + 1.96 * total_std, 0)
        mean_pred = np.maximum(mean_pred, 0)
        return mean_pred, lower, upper


# ============================================================================
# STEP 4 — TRAIN, EVALUATE, FORECAST FOR ONE SEGMENT
# ============================================================================

def prepare_features(df):
    """Return X, y, feature_cols from daily DataFrame"""
    exclude = ['date', 'crash_count', 'year']
    feat_cols = [c for c in df.columns if c not in exclude]
    return df[feat_cols], df['crash_count'], feat_cols


def train_and_evaluate(df, segment_id, images_dir):
    """
    3-way chronological split:
        TRAIN  = all days except last 90  (remaining)
        VAL    = next 60 days             (second-last 60)
        TEST   = last 30 days             (held-out)

    Workflow:
        1. Train on TRAIN, evaluate on VAL  → validation metrics
        2. Retrain on TRAIN + VAL            → final model
        3. Evaluate final model on TEST      → test metrics
    """

    n = len(df)
    test_start = n - TEST_DAYS
    val_start  = test_start - VAL_DAYS

    if val_start < 90:   # need at least ~90 days for training
        raise ValueError(f"Not enough data: {n} days total, need ≥ {TEST_DAYS+VAL_DAYS+90}")

    X, y, feat_cols = prepare_features(df)

    X_train, y_train = X.iloc[:val_start],            y.iloc[:val_start]
    X_val,   y_val   = X.iloc[val_start:test_start],   y.iloc[val_start:test_start]
    X_test,  y_test  = X.iloc[test_start:],            y.iloc[test_start:]

    print(f"  Train: {len(X_train)} days | Val: {len(X_val)} days | Test: {len(X_test)} days")

    # --- seasonal baseline (fitted on train only) ---
    train_seasonal = df.iloc[:val_start].groupby('day_of_year')['crash_count'].mean()
    train_mean     = df.iloc[:val_start]['crash_count'].mean()

    y_base_val  = df.iloc[val_start:test_start]['day_of_year'].map(train_seasonal).fillna(train_mean)
    y_base_test = df.iloc[test_start:]['day_of_year'].map(train_seasonal).fillna(train_mean)

    # ================================================================
    # PHASE 1: train on TRAIN → evaluate on VAL
    # ================================================================
    model_v1 = QuantileEnsemble()
    model_v1.fit(X_train, y_train)

    yp_val_mean, yp_val_lower, yp_val_upper = model_v1.predict_interval(X_val)

    val_metrics = pd.DataFrame({
        'Model': ['Baseline (Seasonal)', 'ML Ensemble (Mean)'],
        'MAE':  [mean_absolute_error(y_val, y_base_val),
                 mean_absolute_error(y_val, yp_val_mean)],
        'RMSE': [np.sqrt(mean_squared_error(y_val, y_base_val)),
                 np.sqrt(mean_squared_error(y_val, yp_val_mean))],
        'R²':   [r2_score(y_val, y_base_val),
                 r2_score(y_val, yp_val_mean)],
    })
    val_coverage = np.mean((y_val.values >= yp_val_lower) &
                            (y_val.values <= yp_val_upper)) * 100

    print(f"  [VAL]  MAE: {val_metrics.loc[1,'MAE']:.4f} | "
          f"RMSE: {val_metrics.loc[1,'RMSE']:.4f} | "
          f"R²: {val_metrics.loc[1,'R²']:.3f} | Coverage: {val_coverage:.1f}%")

    # ================================================================
    # PHASE 2: retrain on TRAIN + VAL → evaluate on TEST
    # ================================================================
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    model_final = QuantileEnsemble()
    model_final.fit(X_train_val, y_train_val)

    yp_test_mean, yp_test_lower, yp_test_upper = model_final.predict_interval(X_test)

    test_metrics = pd.DataFrame({
        'Model': ['Baseline (Seasonal)', 'ML Ensemble (Mean)'],
        'MAE':  [mean_absolute_error(y_test, y_base_test),
                 mean_absolute_error(y_test, yp_test_mean)],
        'RMSE': [np.sqrt(mean_squared_error(y_test, y_base_test)),
                 np.sqrt(mean_squared_error(y_test, yp_test_mean))],
        'R²':   [r2_score(y_test, y_base_test),
                 r2_score(y_test, yp_test_mean)],
    })
    test_coverage = np.mean((y_test.values >= yp_test_lower) &
                             (y_test.values <= yp_test_upper)) * 100

    print(f"  [TEST] MAE: {test_metrics.loc[1,'MAE']:.4f} | "
          f"RMSE: {test_metrics.loc[1,'RMSE']:.4f} | "
          f"R²: {test_metrics.loc[1,'R²']:.3f} | Coverage: {test_coverage:.1f}%")

    # ================================================================
    # COMBINED METRICS TABLE (saved to CSV)
    # ================================================================
    combined_metrics = pd.DataFrame({
        'Model': ['Baseline (Seasonal)', 'ML Ensemble (Mean)'],
        'VAL_MAE':  val_metrics['MAE'].values,
        'VAL_RMSE': val_metrics['RMSE'].values,
        'VAL_R2':   val_metrics['R²'].values,
        'VAL_Coverage': [np.nan, val_coverage],
        'TEST_MAE':  test_metrics['MAE'].values,
        'TEST_RMSE': test_metrics['RMSE'].values,
        'TEST_R2':   test_metrics['R²'].values,
        'TEST_Coverage': [np.nan, test_coverage],
    })

    # ================================================================
    # PLOT — 3-panel: Validation | Test | Actual vs Predicted
    # ================================================================
    fig, axes = plt.subplots(3, 1, figsize=(15, 14))

    # Panel 1: Validation period
    val_dates = df.iloc[val_start:test_start]['date'].values
    ax = axes[0]
    ax.fill_between(val_dates, yp_val_lower, yp_val_upper,
                    alpha=0.3, color='lightblue', label='95% CI')
    ax.plot(val_dates, yp_val_mean, 'r--', lw=1.5, label='Mean Prediction')
    ax.plot(val_dates, y_val.values, 'k-', lw=1, alpha=0.7, label='Actual')
    ax.set_title(f'{segment_id}: VALIDATION (60 days) — trained on preceding data',
                 fontweight='bold')
    ax.set_ylabel('Daily Crashes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Test period (model retrained on train+val)
    test_dates = df.iloc[test_start:]['date'].values
    ax = axes[1]
    ax.fill_between(test_dates, yp_test_lower, yp_test_upper,
                    alpha=0.3, color='lightyellow', label='95% CI')
    ax.plot(test_dates, yp_test_mean, 'r--', lw=1.5, label='Mean Prediction')
    ax.plot(test_dates, y_test.values, 'k-', lw=1, alpha=0.7, label='Actual')
    ax.set_title(f'{segment_id}: TEST (last 30 days) — retrained on train+val',
                 fontweight='bold')
    ax.set_ylabel('Daily Crashes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Actual vs Predicted scatter (test set)
    ax2 = axes[2]
    lower_err = yp_test_mean - yp_test_lower
    upper_err = yp_test_upper - yp_test_mean
    ax2.errorbar(y_test, yp_test_mean, yerr=[lower_err, upper_err],
                 fmt='o', alpha=0.5, capsize=2, color='steelblue', markersize=4)
    mx = max(y_test.max(), yp_test_upper.max(), 1)
    ax2.plot([0, mx], [0, mx], 'k--', lw=1.5)
    ax2.set_xlabel('Actual'); ax2.set_ylabel('Predicted')
    ax2.set_title(f'{segment_id}: Test-Set Prediction Accuracy', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(images_dir, f'{segment_id}_model_comparison.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    return model_final, feat_cols, combined_metrics


def forecast_future(df, model, feat_cols, n_days, segment_id):
    """
    N-day ahead forecast with Poisson probabilities + dual uncertainty,
    matching the original weekly code's approach but at daily resolution.
    """
    extended = df.copy()
    last_date = df['date'].max()
    seasonal_mean = df.groupby('day_of_year')['crash_count'].mean()
    overall_mean  = df['crash_count'].mean()
    overall_std   = df['crash_count'].std()

    predictions = []

    for i in range(n_days):
        next_day = last_date + pd.Timedelta(days=i + 1)
        doy = next_day.timetuple().tm_yday

        # ---- build feature row ----
        row = pd.Series(0.0, index=extended.columns)
        row['date']          = next_day
        row['month']         = next_day.month
        row['day']           = next_day.day
        row['day_of_week']   = next_day.weekday()
        row['day_of_year']   = doy
        row['week_of_year']  = int(next_day.isocalendar().week)
        row['quarter']       = (next_day.month - 1) // 3 + 1
        row['is_weekend']    = int(next_day.weekday() in [5, 6])
        row['is_monday']     = int(next_day.weekday() == 0)
        row['is_friday']     = int(next_day.weekday() == 4)

        row['day_of_week_sin'] = np.sin(2 * np.pi * row['day_of_week'] / 7)
        row['day_of_week_cos'] = np.cos(2 * np.pi * row['day_of_week'] / 7)
        row['day_of_year_sin'] = np.sin(2 * np.pi * doy / 365)
        row['day_of_year_cos'] = np.cos(2 * np.pi * doy / 365)
        row['week_sin']        = np.sin(2 * np.pi * row['week_of_year'] / 52)
        row['week_cos']        = np.cos(2 * np.pi * row['week_of_year'] / 52)
        row['month_sin']       = np.sin(2 * np.pi * row['month'] / 12)
        row['month_cos']       = np.cos(2 * np.pi * row['month'] / 12)

        # carry forward static segment features
        last = extended.iloc[-1]
        for c in ['total_crashes_hist', 'pct_rush_hour', 'pct_night',
                  'pct_peak_hour', 'pct_weekend', 'pct_rain', 'pct_dark',
                  'avg_spd_limit', 'avg_lanes']:
            if c in extended.columns:
                row[c] = last[c]

        # lag features
        for lag in [1, 2, 3, 7, 14, 28]:
            col = f'crashes_lag_{lag}d'
            if col in feat_cols and len(extended) >= lag:
                row[col] = extended.iloc[-lag]['crash_count']

        # rolling features
        for w in [3, 7, 14, 28, 56, 84]:
            if len(extended) >= w:
                recent = extended['crash_count'].tail(w)
                row[f'crashes_rolling_mean_{w}d'] = recent.mean()
                row[f'crashes_rolling_std_{w}d']  = recent.std() if len(recent) > 1 else 0
                row[f'crashes_rolling_sum_{w}d']  = recent.sum()
                row[f'crashes_rolling_max_{w}d']  = recent.max()

        # crash change
        if len(extended) >= 1:
            row['crash_change_1d'] = extended.iloc[-1]['crash_count'] - (
                extended.iloc[-2]['crash_count'] if len(extended) >= 2 else 0)
        if len(extended) >= 7:
            row['crash_change_7d'] = extended.iloc[-1]['crash_count'] - extended.iloc[-7]['crash_count']
        if len(extended) >= 28:
            row['crash_change_28d'] = extended.iloc[-1]['crash_count'] - extended.iloc[-28]['crash_count']

        # --- zero-out any remaining crash-count aggregates for future day ---
        for c in ['rush_hour_crashes','night_crashes','peak_crash_hour_crashes',
                  'total_killed','total_injured','fatal_crashes','injury_crashes',
                  'total_vehicles','rain_crashes','adverse_weather_crashes','dark_crashes']:
            if c in row.index:
                row[c] = 0

        row = row.fillna(0)
        X_new = row[feat_cols].values.reshape(1, -1)

        # ================================================================
        # PREDICTIONS FROM ALL 3 MODELS (epistemic uncertainty)
        # ================================================================
        pred_rf    = model.models['rf'].predict(X_new)[0]
        pred_gb    = model.models['gb'].predict(X_new)[0]
        pred_ridge = model.models['ridge'].predict(X_new)[0]

        ml_pred    = np.mean([pred_rf, pred_gb, pred_ridge])
        model_std  = np.std([pred_rf, pred_gb, pred_ridge])
        residual_std = model.residual_std

        # ================================================================
        # HYBRID BLENDING (ML → seasonal as horizon grows)
        # ================================================================
        seasonal_pred = seasonal_mean.get(doy, overall_mean)

        if i < 14:                      # first 2 weeks: trust ML
            lam      = ml_pred
            base_std = model_std
            method   = "ML"
        elif i < 30:                    # weeks 3-4: blend
            w = min((i - 14) / 16, 1) * 0.3
            lam      = ml_pred * (1 - w) + seasonal_pred * w
            base_std = model_std * (1 - w) + overall_std * 0.3 * w
            method   = "ML+Season"
        elif i < 60:                    # months 2-3: heavier blend
            w = 0.3 + (i - 30) / 30 * 0.4
            lam      = ml_pred * (1 - w) + seasonal_pred * w
            base_std = model_std * (1 - w) + overall_std * 0.5 * w
            method   = "Hybrid"
        else:                           # beyond: seasonal only
            lam      = seasonal_pred
            base_std = overall_std
            method   = "Seasonal"

        lam = max(lam, 0.001)

        # ================================================================
        # DUAL-UNCERTAINTY CONFIDENCE INTERVALS
        # ================================================================
        horizon_factor = 1 + (i / n_days) * 0.5   # 1.0 → 1.5
        total_std = np.sqrt(residual_std**2 + base_std**2) * horizon_factor

        lower_ci = max(lam - 1.96 * total_std, 0)
        upper_ci = max(lam + 1.96 * total_std, lam)

        # Poisson quantiles (better for small counts)
        lower_poisson = stats.poisson.ppf(0.025, lam) if lam > 0 else 0
        upper_poisson = stats.poisson.ppf(0.975, lam) if lam > 0 else 0

        if lam < 5:
            lower_bound = lower_poisson
            upper_bound = upper_poisson
            ci_method   = "Poisson"
        else:
            lower_bound = max(lower_ci, lower_poisson)
            upper_bound = max(upper_ci, upper_poisson)
            ci_method   = "Hybrid-Uncertainty"

        # ================================================================
        # RISK LEVEL (daily thresholds — lower than weekly)
        # ================================================================
        if lam >= 1.0:
            risk = "High"
        elif lam >= 0.5:
            risk = "Medium"
        elif lam >= 0.2:
            risk = "Low"
        else:
            risk = "Very Low"

        # ================================================================
        # POISSON PROBABILITIES
        # ================================================================
        prob_0   = stats.poisson.pmf(0, lam) * 100
        prob_1   = stats.poisson.pmf(1, lam) * 100
        prob_2   = stats.poisson.pmf(2, lam) * 100
        prob_3   = stats.poisson.pmf(3, lam) * 100
        prob_ge4 = (1 - stats.poisson.cdf(3, lam)) * 100

        k_vals = np.arange(0, 15)
        pmfs   = stats.poisson.pmf(k_vals, lam)
        most_likely_k    = int(k_vals[np.argmax(pmfs)])
        prob_most_likely = pmfs.max() * 100

        # ================================================================
        # STORE PREDICTION
        # ================================================================
        predictions.append({
            'date':                 next_day,
            'lambda':               round(lam, 4),
            'predicted_lower':      int(lower_bound),
            'predicted_upper':      int(upper_bound),
            'ci_method':            ci_method,
            'model_uncertainty':    round(model_std, 4),
            'residual_uncertainty': round(residual_std, 4),
            'total_uncertainty':    round(total_std, 4),
            'most_likely_crashes':  most_likely_k,
            'probability_%':        round(prob_most_likely, 1),
            'risk_level':           risk,
            'method':               method,
            'prob_0_crash':         round(prob_0, 1),
            'prob_1_crash':         round(prob_1, 1),
            'prob_2_crash':         round(prob_2, 1),
            'prob_3_crash':         round(prob_3, 1),
            'prob_ge4_crash':       round(prob_ge4, 1),
        })

        # feed prediction back as "observed" for next iteration's lags
        row['crash_count'] = lam
        extended = pd.concat([extended, pd.DataFrame([row])], ignore_index=True)

    result = pd.DataFrame(predictions)

    print(f"  Forecast summary — avg λ: {result['lambda'].mean():.4f}, "
          f"avg CI width: {(result['predicted_upper'] - result['predicted_lower']).mean():.1f}, "
          f"avg σ_total: {result['total_uncertainty'].mean():.4f}")

    return result


# ============================================================================
# STEP 5 — PROCESS ONE SEGMENT END-TO-END
# ============================================================================

def process_segment(crash_df, mslink):
    """Full pipeline for a single segment"""
    segment_id = f"MSLINK_{mslink}"
    print(f"\n{'='*60}")
    print(f"PROCESSING SEGMENT: {segment_id}")
    print(f"{'='*60}")

    try:
        # directories
        seg_dir    = os.path.join(BASE_OUTPUT_DIR, segment_id)
        images_dir = os.path.join(seg_dir, 'images')
        data_dir   = os.path.join(seg_dir, 'data')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # build daily dataset
        daily_df = build_daily_segment(crash_df, mslink)
        print(f"  Daily records: {len(daily_df)} | "
              f"Date range: {daily_df['date'].min().date()} — {daily_df['date'].max().date()}")

        # train & evaluate
        model, feat_cols, metrics = train_and_evaluate(daily_df, segment_id, images_dir)

        # future forecast
        future = forecast_future(daily_df, model, feat_cols, FORECAST_DAYS, segment_id)

        # save outputs
        daily_df.to_csv(os.path.join(data_dir, f'{segment_id}_daily_crashes.csv'), index=False)
        future.to_csv(os.path.join(data_dir, f'{segment_id}_future_predictions_with_risk.csv'), index=False)
        metrics.to_csv(os.path.join(data_dir, f'{segment_id}_model_metrics.csv'), index=False)

        # forecast visualisation
        fig, ax = plt.subplots(figsize=(14, 5))
        colors = {'Very Low': 'green', 'Low': '#FFD700', 'Medium': 'orange', 'High': 'red'}
        for _, row in future.iterrows():
            ax.bar(row['date'], row['lambda'], color=colors.get(row['risk_level'], 'grey'),
                   width=0.8, alpha=0.7)
        ax.fill_between(future['date'], future['predicted_lower'], future['predicted_upper'],
                        alpha=0.15, color='blue', label='95% CI')
        ax.set_title(f'{segment_id}: {FORECAST_DAYS}-Day Crash Forecast with Risk Levels',
                     fontweight='bold')
        ax.set_ylabel('Expected Daily Crashes (λ)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f'{segment_id}_forecast.png'), dpi=200, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Outputs saved to: {seg_dir}")

        return {
            'segment_id': segment_id,
            'mslink': mslink,
            'status': 'SUCCESS',
            'metrics': metrics,
            'future_predictions': future,
        }

    except Exception as e:
        import traceback
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        return {'segment_id': segment_id, 'mslink': mslink,
                'status': 'FAILED', 'error': str(e)}


# ============================================================================
# STEP 6 — MAIN
# ============================================================================

def main():
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    crash_df = load_raw_data(INPUT_PATH)

    segments = sorted(crash_df['MSLINK'].unique())
    print(f"\nTotal segments to process: {len(segments)}")

    results = []
    for idx, mslink in enumerate(segments, 1):
        print(f"\n[{idx}/{len(segments)}]", end="")
        res = process_segment(crash_df, mslink)
        results.append(res)
        gc.collect()

    # ---- summary report ----
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed     = [r for r in results if r['status'] == 'FAILED']

    print(f"\n{'='*60}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(successful)}/{len(segments)}")
    print(f"✗ Failed:     {len(failed)}/{len(segments)}")

    if failed:
        print("\n--- Failed Segments ---")
        for r in failed:
            print(f"  {r['segment_id']}: {r.get('error','unknown')}")

    if successful:
        summary_rows = []
        for r in successful:
            m = r['metrics']
            summary_rows.append({
                'segment_id':    r['segment_id'],
                'mslink':        r['mslink'],
                'VAL_MAE_baseline':  m.loc[0, 'VAL_MAE'],
                'VAL_MAE_ensemble':  m.loc[1, 'VAL_MAE'],
                'VAL_RMSE_ensemble': m.loc[1, 'VAL_RMSE'],
                'VAL_R2_ensemble':   m.loc[1, 'VAL_R2'],
                'VAL_Coverage':      m.loc[1, 'VAL_Coverage'],
                'TEST_MAE_baseline': m.loc[0, 'TEST_MAE'],
                'TEST_MAE_ensemble': m.loc[1, 'TEST_MAE'],
                'TEST_RMSE_ensemble':m.loc[1, 'TEST_RMSE'],
                'TEST_R2_ensemble':  m.loc[1, 'TEST_R2'],
                'TEST_Coverage':     m.loc[1, 'TEST_Coverage'],
                f'{FORECAST_DAYS}d_forecast_total_lambda':
                    r['future_predictions']['lambda'].sum(),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df['VAL_MAE_improvement']  = summary_df['VAL_MAE_baseline']  - summary_df['VAL_MAE_ensemble']
        summary_df['TEST_MAE_improvement'] = summary_df['TEST_MAE_baseline'] - summary_df['TEST_MAE_ensemble']

        summary_path = os.path.join(BASE_OUTPUT_DIR, 'ALL_SEGMENTS_SUMMARY.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nCombined summary saved to: {summary_path}")

        print(f"\nTOP 10 SEGMENTS BY TEST MAE IMPROVEMENT:")
        top = summary_df.sort_values('TEST_MAE_improvement', ascending=False).head(10)
        print(top[['segment_id','VAL_MAE_ensemble','TEST_MAE_baseline',
                    'TEST_MAE_ensemble','TEST_MAE_improvement']].round(4).to_string(index=False))

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All outputs in: {os.path.abspath(BASE_OUTPUT_DIR)}")
    return results


if __name__ == "__main__":
    final_results = main()