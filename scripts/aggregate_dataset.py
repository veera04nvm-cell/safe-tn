import pandas as pd
import numpy as np
import gc
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD RAW CRASH DATA (only needed columns)
# ============================================================================
print("=" * 70)
print("DAILY CRASH DATASET PREPARATION PIPELINE")
print("=" * 70)

INPUT_PATH = 'data/Unlocked_Segmented_2022_2025_Crashes_Final.csv'
USE_COLS = [
    'MSLINK', 'Date of Cr', 'Time of Cr',
    'Total Kill', 'Total Inj', 'Total Veh',
    'Weather Co', 'Light Cond', 'Spd Limit', 'No. Lns.'
]
df = pd.read_csv(INPUT_PATH, usecols=USE_COLS, low_memory=False)
print(f"Loaded {len(df):,} crash records")

# ============================================================================
# STEP 2: FILTER SEGMENTS (>= 10 crashes)
# ============================================================================
MIN_CRASHES = 10
seg_counts = df['MSLINK'].value_counts()
valid_segments = seg_counts[seg_counts >= MIN_CRASHES].index.tolist()
df = df[df['MSLINK'].isin(valid_segments)].copy()
print(f"Segments with >= {MIN_CRASHES} crashes: {len(valid_segments):,}")
print(f"Remaining records: {len(df):,}")

# ============================================================================
# STEP 3: PARSE DATE AND TIME
# ============================================================================
df['date'] = pd.to_datetime(df['Date of Cr'], format='%m/%d/%Y', errors='coerce')
df = df.dropna(subset=['date']).copy()

# Military time -> hour
df['hour'] = df['Time of Cr'].fillna(-100).astype(int) // 100
df.loc[df['hour'] < 0, 'hour'] = np.nan
df.loc[df['hour'] > 23, 'hour'] = 23

print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ============================================================================
# STEP 4: PER-CRASH BINARY FLAGS
# ============================================================================
rush = {7, 8, 9, 16, 17, 18}
night = {22, 23, 0, 1, 2, 3, 4, 5}
peak = {15, 16, 17, 18}

df['is_rush_hour'] = df['hour'].isin(rush).astype(float)
df.loc[df['hour'].isna(), 'is_rush_hour'] = np.nan
df['is_night'] = df['hour'].isin(night).astype(float)
df.loc[df['hour'].isna(), 'is_night'] = np.nan
df['is_peak_crash_hour'] = df['hour'].isin(peak).astype(float)
df.loc[df['hour'].isna(), 'is_peak_crash_hour'] = np.nan
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
df['is_fatal'] = (df['Total Kill'] > 0).astype(int)
df['is_injury'] = (df['Total Inj'] > 0).astype(int)

wc = df['Weather Co'].str.strip().str.lower()
df['is_rain'] = wc.eq('rain').astype(int)
df['is_adverse_weather'] = wc.isin(
    ['rain', 'snow', 'fog', 'sleet/hail', 'blowing snow',
     'severe cross-winds', 'blowing sand/soil/dirt', 'smog/smoke']
).astype(int)
df['is_dark'] = df['Light Cond'].str.strip().str.lower().str.contains('dark', na=False).astype(int)

print("Per-crash features created")

# ============================================================================
# STEP 5: COMPUTE SEGMENT-LEVEL STATIC FEATURES
# ============================================================================
print("Computing segment-level static features...")
seg_totals = df.groupby('MSLINK').agg(
    total_crashes_hist=('date', 'size'),
    total_rush_hour=('is_rush_hour', 'sum'),
    total_night=('is_night', 'sum'),
    total_peak_hour=('is_peak_crash_hour', 'sum'),
    total_weekend=('is_weekend', 'sum'),
    total_rain=('is_rain', 'sum'),
    total_dark=('is_dark', 'sum'),
    avg_spd_limit=('Spd Limit', 'mean'),
    avg_lanes=('No. Lns.', 'mean'),
).reset_index()

for col, tot_col in [('pct_rush_hour', 'total_rush_hour'), ('pct_night', 'total_night'),
                      ('pct_peak_hour', 'total_peak_hour'), ('pct_weekend', 'total_weekend'),
                      ('pct_rain', 'total_rain'), ('pct_dark', 'total_dark')]:
    seg_totals[col] = seg_totals[tot_col] / seg_totals['total_crashes_hist']

seg_features = seg_totals[['MSLINK', 'total_crashes_hist', 'pct_rush_hour', 'pct_night',
                            'pct_peak_hour', 'pct_weekend', 'pct_rain', 'pct_dark',
                            'avg_spd_limit', 'avg_lanes']]

# ============================================================================
# STEP 6: AGGREGATE TO DAILY PER SEGMENT
# ============================================================================
print("Aggregating to daily per segment...")
daily_agg = df.groupby(['MSLINK', 'date']).agg(
    crash_count=('date', 'size'),
    rush_hour_crashes=('is_rush_hour', 'sum'),
    night_crashes=('is_night', 'sum'),
    peak_crash_hour_crashes=('is_peak_crash_hour', 'sum'),
    weekend_crashes=('is_weekend', 'sum'),
    total_killed=('Total Kill', 'sum'),
    total_injured=('Total Inj', 'sum'),
    fatal_crashes=('is_fatal', 'sum'),
    injury_crashes=('is_injury', 'sum'),
    total_vehicles=('Total Veh', 'sum'),
    rain_crashes=('is_rain', 'sum'),
    adverse_weather_crashes=('is_adverse_weather', 'sum'),
    dark_crashes=('is_dark', 'sum'),
    avg_crash_hour=('hour', 'mean'),
).reset_index()

# Free raw data
del df
gc.collect()

# ============================================================================
# STEP 7: PROCESS SEGMENTS IN BATCHES (memory-efficient)
# ============================================================================
date_min = daily_agg['date'].min()
date_max = daily_agg['date'].max()
all_dates = pd.date_range(date_min, date_max, freq='D')
n_days = len(all_dates)

fill_zero_cols = [
    'crash_count', 'rush_hour_crashes', 'night_crashes',
    'peak_crash_hour_crashes', 'weekend_crashes',
    'total_killed', 'total_injured', 'fatal_crashes', 'injury_crashes',
    'total_vehicles', 'rain_crashes', 'adverse_weather_crashes', 'dark_crashes'
]

lag_days = [1, 2, 3, 7, 14, 28]
rolling_windows = [3, 7, 14, 28, 56, 84]

segments = sorted(daily_agg['MSLINK'].unique())
n_segments = len(segments)
BATCH_SIZE = 200

print(f"\nProcessing {n_segments} segments in batches of {BATCH_SIZE}...")
print(f"Date spine: {n_days} days per segment")

OUTPUT_PATH = 'outputs/segmented_crashes/daily_crash_dataset_prepared.csv'
first_batch = True

for batch_start in range(0, n_segments, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, n_segments)
    batch_segments = segments[batch_start:batch_end]
    print(f"\n  Batch {batch_start//BATCH_SIZE + 1}: segments {batch_start+1}-{batch_end}")

    # Build spine for this batch
    spine = pd.DataFrame({
        'MSLINK': np.repeat(batch_segments, n_days),
        'date': np.tile(all_dates, len(batch_segments))
    })

    # Merge daily crash aggregates
    batch_agg = daily_agg[daily_agg['MSLINK'].isin(batch_segments)]
    batch_df = spine.merge(batch_agg, on=['MSLINK', 'date'], how='left')
    del spine

    # Fill zeros
    batch_df[fill_zero_cols] = batch_df[fill_zero_cols].fillna(0).astype(int)
    batch_df['avg_crash_hour'] = batch_df['avg_crash_hour'].fillna(-1)

    # ---- Temporal / calendar features ----
    batch_df['year'] = batch_df['date'].dt.year
    batch_df['month'] = batch_df['date'].dt.month
    batch_df['day'] = batch_df['date'].dt.day
    batch_df['day_of_week'] = batch_df['date'].dt.dayofweek
    batch_df['day_of_year'] = batch_df['date'].dt.dayofyear
    batch_df['week_of_year'] = batch_df['date'].dt.isocalendar().week.astype(int)
    batch_df['quarter'] = batch_df['date'].dt.quarter
    batch_df['is_weekend'] = batch_df['day_of_week'].isin([5, 6]).astype(int)
    batch_df['is_monday'] = (batch_df['day_of_week'] == 0).astype(int)
    batch_df['is_friday'] = (batch_df['day_of_week'] == 4).astype(int)

    # Cyclical encodings
    batch_df['day_of_week_sin'] = np.sin(2 * np.pi * batch_df['day_of_week'] / 7)
    batch_df['day_of_week_cos'] = np.cos(2 * np.pi * batch_df['day_of_week'] / 7)
    batch_df['day_of_year_sin'] = np.sin(2 * np.pi * batch_df['day_of_year'] / 365)
    batch_df['day_of_year_cos'] = np.cos(2 * np.pi * batch_df['day_of_year'] / 365)
    batch_df['week_sin'] = np.sin(2 * np.pi * batch_df['week_of_year'] / 52)
    batch_df['week_cos'] = np.cos(2 * np.pi * batch_df['week_of_year'] / 52)
    batch_df['month_sin'] = np.sin(2 * np.pi * batch_df['month'] / 12)
    batch_df['month_cos'] = np.cos(2 * np.pi * batch_df['month'] / 12)

    # ---- Lag features ----
    batch_df = batch_df.sort_values(['MSLINK', 'date']).reset_index(drop=True)
    for lag in lag_days:
        batch_df[f'crashes_lag_{lag}d'] = batch_df.groupby('MSLINK')['crash_count'].shift(lag)

    # ---- Rolling features ----
    for window in rolling_windows:
        grp = batch_df.groupby('MSLINK')['crash_count']
        batch_df[f'crashes_rolling_mean_{window}d'] = grp.transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        batch_df[f'crashes_rolling_std_{window}d'] = grp.transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        batch_df[f'crashes_rolling_sum_{window}d'] = grp.transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )
        batch_df[f'crashes_rolling_max_{window}d'] = grp.transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )

    # ---- Change features ----
    batch_df['crash_change_1d'] = batch_df.groupby('MSLINK')['crash_count'].diff(1)
    batch_df['crash_change_7d'] = batch_df.groupby('MSLINK')['crash_count'].diff(7)
    batch_df['crash_change_28d'] = batch_df.groupby('MSLINK')['crash_count'].diff(28)

    # Fill NaN lags/rolling with 0
    lag_roll_cols = [c for c in batch_df.columns if 'lag_' in c or 'rolling_' in c or 'crash_change' in c]
    batch_df[lag_roll_cols] = batch_df[lag_roll_cols].fillna(0)

    # ---- Merge segment-level static features ----
    batch_df = batch_df.merge(seg_features, on='MSLINK', how='left')

    # ---- Write to CSV ----
    batch_df.to_csv(OUTPUT_PATH, mode='w' if first_batch else 'a',
                    header=first_batch, index=False)
    first_batch = False

    print(f"    Written {len(batch_df):,} rows | Columns: {len(batch_df.columns)}")
    del batch_df
    gc.collect()

# ============================================================================
# STEP 8: PRINT SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("READING BACK FOR SUMMARY...")
sample = pd.read_csv(OUTPUT_PATH, nrows=5)
print(f"\nColumns ({len(sample.columns)}):")
for i, col in enumerate(sample.columns):
    print(f"  {i+1:3d}. {col}")

# Count total rows (Windows-compatible)
total_lines = sum(1 for _ in open(OUTPUT_PATH)) - 1  # subtract header

print(f"\n{'='*70}")
print("DATASET SUMMARY")
print(f"{'='*70}")
print(f"Total rows:           {total_lines:,}")
print(f"Unique segments:      {n_segments:,}")
print(f"Date range:           {date_min.date()} to {date_max.date()}")
print(f"Total days:           {n_days:,}")
print(f"Total features:       {len(sample.columns)}")
print(f"\nOutput saved to: {OUTPUT_PATH}")
print("=" * 70)