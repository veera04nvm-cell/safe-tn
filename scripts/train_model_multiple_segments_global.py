import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
import os
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dictionary mapping segment IDs to their data files
SEGMENT_DATA_FILES = {
    'segment_01': 'data/I0040_Seg26_traffic_crash_merged.csv',
    'segment_02': 'data/I0040_Seg27_traffic_crash_merged.csv',
    'segment_03': 'data/I0040_Seg28_traffic_crash_merged.csv',
    'segment_04': 'data/I55_Seg05_traffic_crash_merged.csv',
    'segment_05': 'data/I240_Seg02_traffic_crash_merged.csv',
    'segment_06': 'data/I240_Seg03_traffic_crash_merged.csv',
    'segment_07': 'data/I240_Seg05_traffic_crash_merged.csv',
    'segment_08': 'data/I240_Seg08_traffic_crash_merged.csv',
    'segment_09': 'data/I240_Seg11_traffic_crash_merged.csv',
    'segment_10': 'data/I240_Seg12_traffic_crash_merged.csv',
    'segment_11': 'data/I240_Seg13_traffic_crash_merged.csv',
}

OUTPUT_DIR = 'data/multi_segment/'
IMAGES_DIR = 'images/multi_segment/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ============================================================================
# STEP 1: LOAD ALL SEGMENT DATA
# ============================================================================

def load_all_segments(segment_files):
    """Load data from all segments and combine into unified dataset"""
    print("\n" + "="*60)
    print("LOADING MULTI-SEGMENT DATA")
    print("="*60)
    
    all_segments = []
    segment_stats = []
    
    for segment_id, filepath in segment_files.items():
        print(f"\nLoading {segment_id}: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['segment_id'] = segment_id
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate segment statistics
            stats_row = {
                'segment_id': segment_id,
                'total_records': len(df),
                'date_start': df['timestamp'].min(),
                'date_end': df['timestamp'].max(),
                'avg_crashes_per_hour': df['crash_count'].mean(),
                'total_crashes': df['crash_count'].sum(),
                'avg_speed': df['speed'].mean(),
            }
            segment_stats.append(stats_row)
            
            all_segments.append(df)
            print(f"  ✓ Loaded {len(df)} records")
            print(f"  ✓ Date range: {stats_row['date_start']} to {stats_row['date_end']}")
            print(f"  ✓ Total crashes: {stats_row['total_crashes']}")
            
        except FileNotFoundError:
            print(f"  ⚠ Warning: File not found, skipping segment")
            continue
        except Exception as e:
            print(f"  ❌ Error loading segment: {str(e)}")
            continue
    
    if not all_segments:
        raise ValueError("No segments loaded successfully!")
    
    # Combine all segments
    combined_df = pd.concat(all_segments, ignore_index=True)
    stats_df = pd.DataFrame(segment_stats)
    
    print("\n" + "="*60)
    print("COMBINED DATASET SUMMARY")
    print("="*60)
    print(f"Total segments loaded: {len(all_segments)}")
    print(f"Total records: {len(combined_df):,}")
    print(f"Total crashes: {combined_df['crash_count'].sum():,.0f}")
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print("\nPer-segment statistics:")
    print(stats_df.to_string(index=False))
    
    # Save segment statistics
    stats_path = os.path.join(OUTPUT_DIR, 'segment_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✓ Segment statistics saved to: {stats_path}")
    
    return combined_df, stats_df

# ============================================================================
# STEP 2: ENHANCED FEATURE ENGINEERING (WITH SEGMENT FEATURES)
# ============================================================================

def create_segment_features(df, stats_df):
    """
    Create segment-specific features that help the model distinguish segments
    """
    print("\nCreating segment-specific features...")
    df = df.copy()
    
    # Basic segment encoding (one-hot encoding will be done later)
    segment_map = {seg: idx for idx, seg in enumerate(df['segment_id'].unique())}
    df['segment_numeric'] = df['segment_id'].map(segment_map)
    
    # Add segment historical characteristics
    segment_profiles = stats_df.set_index('segment_id')[
        ['avg_crashes_per_hour', 'avg_speed']
    ].to_dict('index')
    
    df['segment_avg_crash_rate'] = df['segment_id'].map(
        lambda x: segment_profiles[x]['avg_crashes_per_hour']
    )
    df['segment_avg_speed'] = df['segment_id'].map(
        lambda x: segment_profiles[x]['avg_speed']
    )
    
    # Relative metrics (how does current value compare to segment baseline?)
    df['speed_vs_segment_avg'] = df['speed'] - df['segment_avg_speed']
    df['crash_vs_segment_avg'] = df['crash_count'] - df['segment_avg_crash_rate']
    
    print(f"✓ Created segment-specific features")
    return df

def create_enhanced_features(df):
    """Create comprehensive features from hourly data"""
    print("Creating enhanced temporal and traffic features...")
    df = df.copy()
    
    # Traffic features
    df['speed_deviation'] = np.abs(df['speed'] - df['historical_average_speed'])
    df['speed_ratio'] = df['speed'] / df['reference_speed'].replace(0, 1)
    df['congestion_indicator'] = (df['speed'] < df['reference_speed'] * 0.7).astype(int)
    
    # Time features
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df['is_peak_crash_hour'] = df['hour'].isin([15, 16, 17, 18]).astype(int)
    
    print(f"✓ Created enhanced features")
    return df

def aggregate_to_weekly_multi_segment(df):
    """
    Aggregate to weekly data while preserving segment identity
    """
    print("Aggregating to weekly data per segment...")
    
    # Group by both segment and week
    df = df.set_index('timestamp')
    
    weekly_list = []
    
    for segment_id in df['segment_id'].unique():
        segment_data = df[df['segment_id'] == segment_id]
        
        weekly_segment = segment_data.resample('W-MON').agg({
            'crash_count': 'sum',
            'speed': ['mean', 'std', 'min', 'max'],
            'NPMRDS2': ['mean', 'std'],
            'speed_deviation': ['mean', 'max'],
            'speed_ratio': ['mean', 'min'],
            'congestion_indicator': 'sum',
            'is_weekend': 'sum',
            'is_rush_hour': 'sum',
            'is_night': 'sum',
            'is_peak_crash_hour': 'sum',
            'segment_avg_crash_rate': 'first',
            'segment_avg_speed': 'first',
        }).reset_index()
        
        weekly_segment.columns = ['week_start', 'total_crashes',
                                   'avg_speed', 'std_speed', 'min_speed', 'max_speed',
                                   'avg_npmrds', 'std_npmrds',
                                   'avg_speed_deviation', 'max_speed_deviation',
                                   'avg_speed_ratio', 'min_speed_ratio',
                                   'total_congestion_hours', 'total_weekend_hours',
                                   'total_rush_hours', 'total_night_hours', 
                                   'total_peak_crash_hours',
                                   'segment_avg_crash_rate', 'segment_avg_speed']
        
        weekly_segment['segment_id'] = segment_id
        weekly_list.append(weekly_segment)
    
    weekly_df = pd.concat(weekly_list, ignore_index=True)
    
    # Derived features
    weekly_df['speed_variability'] = weekly_df['std_speed'] / weekly_df['avg_speed'].replace(0, 1)
    weekly_df['speed_range'] = weekly_df['max_speed'] - weekly_df['min_speed']
    weekly_df['pct_congested'] = weekly_df['total_congestion_hours'] / 168
    weekly_df['pct_rush_hour'] = weekly_df['total_rush_hours'] / 168
    
    # Temporal features
    weekly_df['year'] = weekly_df['week_start'].dt.year
    weekly_df['month'] = weekly_df['week_start'].dt.month
    weekly_df['quarter'] = weekly_df['week_start'].dt.quarter
    weekly_df['week_of_year'] = weekly_df['week_start'].dt.isocalendar().week
    
    # Cyclical encoding
    weekly_df['week_sin'] = np.sin(2 * np.pi * weekly_df['week_of_year'] / 52)
    weekly_df['week_cos'] = np.cos(2 * np.pi * weekly_df['week_of_year'] / 52)
    weekly_df['month_sin'] = np.sin(2 * np.pi * weekly_df['month'] / 12)
    weekly_df['month_cos'] = np.cos(2 * np.pi * weekly_df['month'] / 12)
    
    print(f"✓ Created {len(weekly_df)} weekly records across {weekly_df['segment_id'].nunique()} segments")
    return weekly_df

# ============================================================================
# STEP 3: SEGMENT-AWARE LAG FEATURES
# ============================================================================

def add_segment_lag_features(df, lag_weeks=[1, 2, 3, 4, 8, 12]):
    """
    Add lag features that are segment-specific
    CRITICAL: Lags must be calculated within each segment!
    """
    print("Adding segment-aware lag and rolling features...")
    
    df = df.sort_values(['segment_id', 'week_start']).reset_index(drop=True)
    
    # Group by segment for lag calculations
    for segment_id in df['segment_id'].unique():
        mask = df['segment_id'] == segment_id
        segment_data = df.loc[mask, 'total_crashes']
        
        # Lag features
        for lag in lag_weeks:
            df.loc[mask, f'crashes_lag_{lag}w'] = segment_data.shift(lag)
        
        # Rolling features
        for window in [2, 4, 8, 12]:
            df.loc[mask, f'crashes_rolling_mean_{window}w'] = \
                segment_data.rolling(window=window, min_periods=1).mean()
            df.loc[mask, f'crashes_rolling_std_{window}w'] = \
                segment_data.rolling(window=window, min_periods=1).std()
            df.loc[mask, f'crashes_rolling_max_{window}w'] = \
                segment_data.rolling(window=window, min_periods=1).max()
        
        # Change features
        df.loc[mask, 'crash_change_1w'] = segment_data.diff(1)
        df.loc[mask, 'crash_change_4w'] = segment_data.diff(4)
    
    df = df.fillna(0)
    
    print(f"✓ Total features: {len(df.columns)}")
    return df

# ============================================================================
# STEP 4: ONE-HOT ENCODE SEGMENTS
# ============================================================================

def prepare_ml_features_multi_segment(df):
    """
    Prepare features including one-hot encoded segments
    """
    print("\nPreparing ML features with segment encoding...")
    
    # One-hot encode segment_id
    segment_dummies = pd.get_dummies(df['segment_id'], prefix='segment')
    df_encoded = pd.concat([df, segment_dummies], axis=1)
    
    # Exclude columns from features
    exclude_cols = ['week_start', 'total_crashes', 'year', 'segment_id', 
                    'segment_numeric']
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X = df_encoded[feature_cols]
    y = df_encoded['total_crashes']
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Segments encoded: {len([c for c in feature_cols if c.startswith('segment_')])}")
    
    return X, y, feature_cols, df_encoded

# ============================================================================
# STEP 5: QUANTILE ENSEMBLE (UNCHANGED FROM ORIGINAL)
# ============================================================================

class QuantileEnsemble:
    """Ensemble model that predicts mean + confidence intervals"""
    
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        self.quantiles = quantiles
        self.models = {}
        
    def fit(self, X, y):
        print(f"Training quantile models for intervals: {self.quantiles}")
        
        for name, model_class in [
            ('rf', RandomForestRegressor),
            ('gb', GradientBoostingRegressor),
            ('ridge', Ridge)
        ]:
            self.models[name] = model_class(random_state=42)
            self.models[name].fit(X, y)
        
        predictions = np.mean([
            self.models['rf'].predict(X),
            self.models['gb'].predict(X),
            self.models['ridge'].predict(X)
        ], axis=0)
        
        self.residuals = y - predictions
        self.residual_std = np.std(self.residuals)
        
        self.quantile_multipliers = {
            q: np.percentile(self.residuals, q * 100) 
            for q in self.quantiles
        }
        
        print(f"✓ Residual std: {self.residual_std:.3f}")
        
        return self
    
    def predict_interval(self, X):
        preds = np.array([
            self.models['rf'].predict(X),
            self.models['gb'].predict(X),
            self.models['ridge'].predict(X)
        ])
        
        mean_pred = np.mean(preds, axis=0)
        model_std = np.std(preds, axis=0)
        total_std = np.sqrt(self.residual_std**2 + model_std**2)
        
        z_score = 1.96
        lower_bound = mean_pred - z_score * total_std
        upper_bound = mean_pred + z_score * total_std
        
        mean_pred = np.maximum(mean_pred, 0)
        lower_bound = np.maximum(lower_bound, 0)
        upper_bound = np.maximum(upper_bound, 0)
        
        return mean_pred, lower_bound, upper_bound
    
    def predict(self, X):
        mean_pred, _, _ = self.predict_interval(X)
        return mean_pred

# ============================================================================
# STEP 6: TRAIN AND EVALUATE
# ============================================================================

def train_multi_segment_model(X_train, y_train, X_test, y_test):
    """Train single global model on all segments"""
    
    print("\n" + "="*60)
    print("TRAINING GLOBAL MULTI-SEGMENT MODEL")
    print("="*60)
    
    quantile_model = QuantileEnsemble(quantiles=[0.05, 0.5, 0.95])
    quantile_model.fit(X_train, y_train)
    
    y_pred_mean, y_pred_lower, y_pred_upper = quantile_model.predict_interval(X_test)
    
    mae = mean_absolute_error(y_test, y_pred_mean)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    r2 = r2_score(y_test, y_pred_mean)
    coverage = np.mean((y_test >= y_pred_lower) & (y_test <= y_pred_upper)) * 100
    avg_width = np.mean(y_pred_upper - y_pred_lower)
    
    print(f"\nGlobal Model Performance:")
    print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    print(f"  Coverage: {coverage:.1f}% | Avg interval: {avg_width:.2f}")
    
    return quantile_model, y_pred_mean, y_pred_lower, y_pred_upper

def evaluate_per_segment(df_test, y_test, y_pred_mean):
    """Calculate metrics for each segment individually"""
    
    print("\n" + "="*60)
    print("PER-SEGMENT PERFORMANCE")
    print("="*60)
    
    results = []
    
    for segment_id in df_test['segment_id'].unique():
        mask = df_test['segment_id'] == segment_id
        
        seg_y_test = y_test[mask]
        seg_y_pred = y_pred_mean[mask]
        
        if len(seg_y_test) > 0:
            mae = mean_absolute_error(seg_y_test, seg_y_pred)
            rmse = np.sqrt(mean_squared_error(seg_y_test, seg_y_pred))
            r2 = r2_score(seg_y_test, seg_y_pred)
            
            results.append({
                'segment_id': segment_id,
                'n_test_weeks': len(seg_y_test),
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'R2': round(r2, 3)
            })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return results_df

# ============================================================================
# STEP 7: FORECAST FUTURE FOR ALL SEGMENTS
# ============================================================================

def forecast_all_segments(df_full, quantile_model, feature_cols, n_weeks=36):
    """
    Generate forecasts for all segments
    """
    print("\n" + "="*60)
    print(f"FORECASTING {n_weeks} WEEKS FOR ALL SEGMENTS")
    print("="*60)
    
    all_forecasts = []
    
    for segment_id in df_full['segment_id'].unique():
        print(f"\nForecasting for {segment_id}...")
        
        segment_df = df_full[df_full['segment_id'] == segment_id].copy()
        segment_future = predict_segment_future(
            segment_df, segment_id, quantile_model, feature_cols, n_weeks
        )
        
        all_forecasts.append(segment_future)
    
    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
    
    print("\n" + "="*60)
    print("FORECAST SUMMARY (ALL SEGMENTS)")
    print("="*60)
    summary = combined_forecasts.groupby('segment_id').agg({
        'predicted_mean': ['sum', 'mean'],
        'predicted_lower': 'sum',
        'predicted_upper': 'sum'
    }).round(2)
    print(summary)
    
    return combined_forecasts

def predict_segment_future(segment_df, segment_id, quantile_model, 
                          feature_cols, n_weeks=36):
    """Predict future for a single segment"""
    
    extended_df = segment_df.copy()
    last_date = segment_df['week_start'].max()
    
    seasonal_stats = segment_df.groupby('week_of_year')['total_crashes'].agg(['mean', 'std'])
    
    future_preds = []
    
    for i in range(n_weeks):
        next_date = last_date + pd.Timedelta(weeks=i+1)
        week_num = next_date.isocalendar().week
        
        # Create new row with segment ID preserved
        new_row = pd.Series(index=extended_df.columns, dtype='float64')
        new_row['week_start'] = next_date
        new_row['segment_id'] = segment_id
        new_row['month'] = next_date.month
        new_row['quarter'] = next_date.quarter
        new_row['week_of_year'] = week_num
        new_row['week_sin'] = np.sin(2 * np.pi * week_num / 52)
        new_row['week_cos'] = np.cos(2 * np.pi * week_num / 52)
        new_row['month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
        
        # Copy segment characteristics
        last_row = extended_df.iloc[-1]
        for col in ['segment_avg_crash_rate', 'segment_avg_speed', 'avg_speed', 
                    'std_speed', 'min_speed', 'max_speed']:
            if col in extended_df.columns:
                new_row[col] = last_row[col]
        
        # Update lags
        for lag in [1, 2, 3, 4, 8, 12]:
            if f'crashes_lag_{lag}w' in feature_cols and lag <= len(extended_df):
                new_row[f'crashes_lag_{lag}w'] = extended_df.iloc[-lag]['total_crashes']
        
        new_row = new_row.fillna(0)
        
        # One-hot encode segment
        segment_encoded = pd.get_dummies(pd.Series([segment_id]), prefix='segment')
        for col in segment_encoded.columns:
            new_row[col] = segment_encoded[col].values[0]
        
        # Ensure all required features exist
        for col in feature_cols:
            if col not in new_row.index:
                new_row[col] = 0
        
        # Predict
        X_new = new_row[feature_cols].values.reshape(1, -1)
        X_new = np.nan_to_num(X_new, nan=0.0)
        
        ml_mean, ml_lower, ml_upper = quantile_model.predict_interval(X_new)
        
        new_row['total_crashes'] = ml_mean[0]
        extended_df = pd.concat([extended_df, pd.DataFrame([new_row])], ignore_index=True)
        
        future_preds.append({
            'segment_id': segment_id,
            'week_start': next_date,
            'predicted_mean': round(ml_mean[0], 2),
            'predicted_lower': round(ml_lower[0], 2),
            'predicted_upper': round(ml_upper[0], 2)
        })
    
    return pd.DataFrame(future_preds)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main multi-segment training pipeline"""
    
    print("\n" + "="*80)
    print("MULTI-SEGMENT TRAFFIC CRASH FORECASTING SYSTEM")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load all segments
        combined_df, stats_df = load_all_segments(SEGMENT_DATA_FILES)
        
        # Feature engineering
        combined_df = create_segment_features(combined_df, stats_df)
        combined_df = create_enhanced_features(combined_df)
        
        # Weekly aggregation
        weekly_df = aggregate_to_weekly_multi_segment(combined_df)
        
        # Add segment-aware lags
        weekly_df = add_segment_lag_features(weekly_df)
        
        # Prepare features
        X, y, feature_cols, df_encoded = prepare_ml_features_multi_segment(weekly_df)
        
        # Train/test split (chronological, last 25% for testing)
        split_idx = int(len(X) * 0.75)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        df_test = df_encoded.iloc[split_idx:]
        
        print(f"\n✓ Train: {len(X_train)} weeks | Test: {len(X_test)} weeks")
        
        # Train global model
        model, y_pred_mean, y_pred_lower, y_pred_upper = \
            train_multi_segment_model(X_train, y_train, X_test, y_test)
        
        # Evaluate per segment
        segment_performance = evaluate_per_segment(df_test, y_test, y_pred_mean)
        
        # Forecast future
        future_forecasts = forecast_all_segments(df_encoded, model, feature_cols, n_weeks=36)
        
        # Save outputs
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        weekly_df.to_csv(f'{OUTPUT_DIR}/weekly_all_segments.csv', index=False)
        segment_performance.to_csv(f'{OUTPUT_DIR}/segment_performance.csv', index=False)
        future_forecasts.to_csv(f'{OUTPUT_DIR}/future_forecasts_all_segments.csv', index=False)
        
        print(f"✓ All results saved to {OUTPUT_DIR}")
        print("\n✅ MULTI-SEGMENT FORECASTING COMPLETED!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()