import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
import os
from datetime import datetime
import glob
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define your 11 segments
SEGMENTS = {
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

# Or automatically detect all CSV files in data folder:
# SEGMENTS = {f"segment_{i:02d}": path 
#             for i, path in enumerate(glob.glob('data/*_traffic_crash_merged.csv'), 1)}

BASE_OUTPUT_DIR = 'outputs/risk_score/'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# SEGMENT-SPECIFIC OUTPUT DIRECTORIES
# ============================================================================

def create_segment_directories(segment_id):
    """Create output directories for a specific segment"""
    segment_dir = os.path.join(BASE_OUTPUT_DIR, segment_id)
    images_dir = os.path.join(segment_dir, 'images')
    data_dir = os.path.join(segment_dir, 'data')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    return segment_dir, images_dir, data_dir

# ============================================================================
# REUSABLE FUNCTIONS (Same as your original code)
# ============================================================================

def load_and_prepare_data(filepath, segment_id):
    """Load hourly crash data and prepare for analysis"""
    print(f"\n{'='*60}")
    print(f"PROCESSING SEGMENT: {segment_id}")
    print(f"{'='*60}")
    print(f"Loading data from: {filepath}")
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✓ Data loaded: {len(df)} hourly records")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def create_enhanced_features(df):
    """Create comprehensive features from hourly data"""
    print("Creating enhanced features...")
    df = df.copy()
    
    df['speed_deviation'] = np.abs(df['speed'] - df['historical_average_speed'])
    df['speed_ratio'] = df['speed'] / df['reference_speed'].replace(0, 1)
    df['congestion_indicator'] = (df['speed'] < df['reference_speed'] * 0.7).astype(int)
    
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df['is_peak_crash_hour'] = df['hour'].isin([15, 16, 17, 18]).astype(int)
    
    print(f"✓ Created enhanced features")
    return df

def aggregate_to_weekly_advanced(df):
    """Aggregate to weekly with rich feature set"""
    print("Aggregating to weekly data...")
    df = df.set_index('timestamp')
    
    weekly_df = df.resample('W-MON').agg({
        'crash_count': 'sum',
        'speed': ['mean', 'std', 'min', 'max'],
        'NPMRDS2': ['mean', 'std'],
        'speed_deviation': ['mean', 'max'],
        'speed_ratio': ['mean', 'min'],
        'congestion_indicator': 'sum',
        'is_weekend': 'sum',
        'is_rush_hour': 'sum',
        'is_night': 'sum',
        'is_peak_crash_hour': 'sum'
    }).reset_index()
    
    weekly_df.columns = ['week_start', 'total_crashes',
                         'avg_speed', 'std_speed', 'min_speed', 'max_speed',
                         'avg_npmrds', 'std_npmrds',
                         'avg_speed_deviation', 'max_speed_deviation',
                         'avg_speed_ratio', 'min_speed_ratio',
                         'total_congestion_hours', 'total_weekend_hours',
                         'total_rush_hours', 'total_night_hours', 
                         'total_peak_crash_hours']
    
    weekly_df['speed_variability'] = weekly_df['std_speed'] / weekly_df['avg_speed'].replace(0, 1)
    weekly_df['speed_range'] = weekly_df['max_speed'] - weekly_df['min_speed']
    weekly_df['pct_congested'] = weekly_df['total_congestion_hours'] / 168
    weekly_df['pct_rush_hour'] = weekly_df['total_rush_hours'] / 168
    
    weekly_df['year'] = weekly_df['week_start'].dt.year
    weekly_df['month'] = weekly_df['week_start'].dt.month
    weekly_df['quarter'] = weekly_df['week_start'].dt.quarter
    weekly_df['week_of_year'] = weekly_df['week_start'].dt.isocalendar().week
    
    weekly_df['week_sin'] = np.sin(2 * np.pi * weekly_df['week_of_year'] / 52)
    weekly_df['week_cos'] = np.cos(2 * np.pi * weekly_df['week_of_year'] / 52)
    weekly_df['month_sin'] = np.sin(2 * np.pi * weekly_df['month'] / 12)
    weekly_df['month_cos'] = np.cos(2 * np.pi * weekly_df['month'] / 12)
    
    print(f"✓ Created {len(weekly_df)} weekly records")
    return weekly_df

def add_lag_features(df, lag_weeks=[1, 2, 3, 4, 8, 12]):
    """Add lagged crash counts and rolling statistics"""
    print("Adding lag and rolling features...")
    df = df.copy()
    
    for lag in lag_weeks:
        df[f'crashes_lag_{lag}w'] = df['total_crashes'].shift(lag)
    
    for window in [2, 4, 8, 12]:
        df[f'crashes_rolling_mean_{window}w'] = df['total_crashes'].rolling(
            window=window, min_periods=1).mean()
        df[f'crashes_rolling_std_{window}w'] = df['total_crashes'].rolling(
            window=window, min_periods=1).std()
        df[f'crashes_rolling_max_{window}w'] = df['total_crashes'].rolling(
            window=window, min_periods=1).max()
    
    df['crash_change_1w'] = df['total_crashes'].diff(1)
    df['crash_change_4w'] = df['total_crashes'].diff(4)
    df = df.fillna(0)
    
    print(f"✓ Total features: {len(df.columns)}")
    return df

def prepare_ml_features(df, feature_cols=None):
    """Prepare features for ML models"""
    if feature_cols is None:
        exclude_cols = ['week_start', 'total_crashes', 'year']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['total_crashes']
    
    return X, y, feature_cols

class QuantileEnsemble:
    """Ensemble model that predicts mean + confidence intervals"""
    
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        self.quantiles = quantiles
        self.models = {}
        
    def fit(self, X, y):
        """Train separate models for each quantile"""
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
        """Predict with confidence intervals"""
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
        """Standard prediction (mean only)"""
        mean_pred, _, _ = self.predict_interval(X)
        return mean_pred

def train_ensemble_with_intervals(X_train, y_train, X_test, y_test):
    """Train ensemble model with prediction intervals"""
    
    print("\n" + "="*60)
    print("TRAINING QUANTILE ENSEMBLE MODEL")
    print("="*60)
    
    quantile_model = QuantileEnsemble(quantiles=[0.05, 0.5, 0.95])
    quantile_model.fit(X_train, y_train)
    
    y_pred_mean, y_pred_lower, y_pred_upper = quantile_model.predict_interval(X_test)
    
    mae = mean_absolute_error(y_test, y_pred_mean)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    r2 = r2_score(y_test, y_pred_mean)
    
    coverage = np.mean((y_test >= y_pred_lower) & (y_test <= y_pred_upper)) * 100
    avg_width = np.mean(y_pred_upper - y_pred_lower)
    
    print(f"\nModel Performance:")
    print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    print(f"\nInterval Quality:")
    print(f"  Coverage: {coverage:.1f}% (target: 95%)")
    print(f"  Avg interval width: {avg_width:.2f} crashes")
    
    return quantile_model, y_pred_mean, y_pred_lower, y_pred_upper

def comprehensive_evaluation_with_intervals(df, y_test, y_pred_baseline, 
                                           y_pred_mean, y_pred_lower, y_pred_upper,
                                           images_dir, segment_id):
    """Compare models and visualize prediction intervals"""
    
    test_df = df.iloc[-len(y_test):].copy()
    test_df['pred_baseline'] = y_pred_baseline
    test_df['pred_mean'] = y_pred_mean
    test_df['pred_lower'] = y_pred_lower
    test_df['pred_upper'] = y_pred_upper
    
    metrics = pd.DataFrame({
        'Model': ['Baseline (Seasonal)', 'ML Ensemble (Mean)'],
        'MAE': [
            mean_absolute_error(y_test, y_pred_baseline),
            mean_absolute_error(y_test, y_pred_mean)
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(y_test, y_pred_baseline)),
            np.sqrt(mean_squared_error(y_test, y_pred_mean))
        ],
        'R²': [
            r2_score(y_test, y_pred_baseline),
            r2_score(y_test, y_pred_mean)
        ]
    })
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(metrics.to_string(index=False))
    
    coverage = np.mean((y_test >= y_pred_lower) & (y_test <= y_pred_upper)) * 100
    print(f"\nPrediction Interval Coverage: {coverage:.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    ax1 = axes[0]
    ax1.fill_between(test_df['week_start'], y_pred_lower, y_pred_upper,
                     alpha=0.3, color='lightblue', label='95% Prediction Interval')
    ax1.plot(test_df['week_start'], y_pred_lower, 
             color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label='Lower Bound')
    ax1.plot(test_df['week_start'], y_pred_mean,
             color='red', linestyle='--', linewidth=2, label='Mean Prediction')
    ax1.plot(test_df['week_start'], y_pred_upper,
             color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label='Upper Bound')
    ax1.plot(test_df['week_start'], y_test.values,
             color='black', linewidth=2.5, marker='o', markersize=5,
             label='Actual Crashes', zorder=5)
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Weekly Crashes', fontsize=12)
    ax1.set_title(f'{segment_id}: Predictions with 95% Confidence Intervals', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    lower_error = y_pred_mean - y_pred_lower
    upper_error = y_pred_upper - y_pred_mean
    
    ax2.errorbar(y_test, y_pred_mean, 
                 yerr=[lower_error, upper_error],
                 fmt='o', alpha=0.6, capsize=3, capthick=1,
                 label='Predictions with 95% CI', color='steelblue')
    
    max_val = max(y_test.max(), y_pred_upper.max())
    ax2.plot([0, max_val], [0, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Weekly Crashes', fontsize=12)
    ax2.set_ylabel('Predicted Weekly Crashes', fontsize=12)
    ax2.set_title(f'{segment_id}: Prediction Accuracy with Uncertainty Bounds', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(images_dir, f'{segment_id}_model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()
    
    return metrics

def predict_future_with_intervals(df, quantile_model, feature_cols, n_weeks=36):
    """36-week forecast with full Poisson probabilities + risk scoring"""
    extended_df = df.copy()
    last_date = df['week_start'].max()
    seasonal_mean = df.groupby('week_of_year')['total_crashes'].mean()
    overall_mean = df['total_crashes'].mean()

    print("\n" + "="*70)
    print(f"GENERATING {n_weeks}-WEEK FORECAST WITH POISSON RISK SCORING")
    print("="*70)

    predictions = []

    for i in range(n_weeks):
        next_week = last_date + pd.Timedelta(weeks=i+1)
        week_num = int(next_week.isocalendar().week)

        # === Build feature row (unchanged) ===
        row = pd.Series(index=extended_df.columns, dtype='float64')
        row['week_start'] = next_week
        row['week_of_year'] = week_num
        row['month'] = next_week.month
        row['quarter'] = next_week.quarter
        row['week_sin'] = np.sin(2 * np.pi * week_num / 52)
        row['week_cos'] = np.cos(2 * np.pi * week_num / 52)
        row['month_sin'] = np.sin(2 * np.pi * next_week.month / 12)
        row['month_cos'] = np.cos(2 * np.pi * next_week.month / 12)

        last = extended_df.iloc[-1]
        for col in ['avg_speed', 'std_speed', 'min_speed', 'max_speed', 'avg_npmrds', 'std_npmrds',
                    'avg_speed_deviation', 'max_speed_deviation', 'avg_speed_ratio', 'min_speed_ratio',
                    'total_congestion_hours', 'total_weekend_hours', 'total_rush_hours', 'total_night_hours',
                    'total_peak_crash_hours', 'speed_variability', 'speed_range', 'pct_congested', 'pct_rush_hour']:
            if col in extended_df.columns:
                row[col] = last[col]

        for lag in [1,2,3,4,8,12]:
            if f'crashes_lag_{lag}w' in feature_cols and len(extended_df) >= lag:
                row[f'crashes_lag_{lag}w'] = extended_df.iloc[-lag]['total_crashes']
        for w in [2,4,8,12]:
            if len(extended_df) >= w:
                recent = extended_df['total_crashes'].tail(w)
                row[f'crashes_rolling_mean_{w}w'] = recent.mean()
                row[f'crashes_rolling_std_{w}w'] = recent.std() if len(recent)>1 else 0
                row[f'crashes_rolling_max_{w}w'] = recent.max()

        row = row.fillna(0)
        X_new = row[feature_cols].values.reshape(1, -1)

        # === Model prediction + hybrid blending ===
        ml_pred = float(quantile_model.predict(X_new)[0])
        seasonal_pred = seasonal_mean.get(week_num, overall_mean)

        if i < 4:
            λ = ml_pred
            method = "ML"
        elif i < 12:
            w = min((i-4)/8, 1) * 0.3
            λ = ml_pred * (1-w) + seasonal_pred * w
            method = "ML+Season"
        elif i < 24:
            w = 0.3 + (i-12)/12 * 0.4
            λ = ml_pred * (1-w) + seasonal_pred * w
            method = "Hybrid"
        else:
            λ = seasonal_pred
            method = "Seasonal"

        λ = max(λ, 0.01)

        # === Risk level ===
        if λ >= 4.0:
            risk = "High"
        elif λ >= 2.5:
            risk = "Medium"
        elif λ >= 1.0:
            risk = "Low"
        else:
            risk = "Very Low"

        # === FULL POISSON PROBABILITIES (calculated BEFORE using them) ===
        prob_0 = stats.poisson.pmf(0, λ) * 100
        prob_1 = stats.poisson.pmf(1, λ) * 100
        prob_2 = stats.poisson.pmf(2, λ) * 100
        prob_3 = stats.poisson.pmf(3, λ) * 100
        prob_ge4 = (1 - stats.poisson.cdf(3, λ)) * 100

        # Most likely number of crashes
        k_values = np.arange(0, 20)  # safe upper bound
        pmfs = stats.poisson.pmf(k_values, λ)
        most_likely_k = int(k_values[np.argmax(pmfs)])
        prob_most_likely = pmfs.max() * 100

        # === Console output ===
        print(f"{next_week.strftime('%Y-%m-%d')}: λ={λ:.3f} → P({most_likely_k}) = {prob_most_likely:.1f}% | Risk: {risk} [{method}]")

        # === Save row ===
        predictions.append({
            'week_start': next_week,
            'lambda': round(λ, 3),
            'most_likely_crashes': most_likely_k,
            'probability_%': round(prob_most_likely, 1),
            'risk_level': risk,
            'method': method,
            'prob_0_crash': round(prob_0, 1),
            'prob_1_crash': round(prob_1, 1),
            'prob_2_crash': round(prob_2, 1),
            'prob_3_crash': round(prob_3, 1),
            'prob_ge4_crash': round(prob_ge4, 1)
        })

        # Update history for next iteration
        row['total_crashes'] = λ
        extended_df = pd.concat([extended_df, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(predictions)

# ============================================================================
# SEGMENT PROCESSING FUNCTION
# ============================================================================

def process_segment(segment_id, filepath):
    """Process a single segment through entire pipeline"""
    
    try:
        # Create directories
        segment_dir, images_dir, data_dir = create_segment_directories(segment_id)
        
        # Step 1: Load data
        df = load_and_prepare_data(filepath, segment_id)
        
        # Step 2: Feature engineering
        df = create_enhanced_features(df)
        
        # Step 3: Weekly aggregation
        weekly_df = aggregate_to_weekly_advanced(df)
        
        # Step 4: Lag features
        weekly_df = add_lag_features(weekly_df)
        
        # Step 5: Train/test split
        split_idx = int(len(weekly_df) * 0.75)
        train_df = weekly_df.iloc[:split_idx].copy()
        test_df = weekly_df.iloc[split_idx:].copy()
        
        seasonal_avg = train_df.groupby('week_of_year')['total_crashes'].mean()
        test_df['baseline_pred'] = test_df['week_of_year'].map(seasonal_avg)
        
        X, y, feature_cols = prepare_ml_features(weekly_df)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        
        print(f"✓ Train: {len(X_train)} weeks | Test: {len(X_test)} weeks")
        
        # Step 6: Train model
        quantile_model, y_pred_mean, y_pred_lower, y_pred_upper = \
            train_ensemble_with_intervals(X_train, y_train, X_test, y_test)
        
        # Step 7: Evaluation
        metrics = comprehensive_evaluation_with_intervals(
            weekly_df, y_test, test_df['baseline_pred'].values,
            y_pred_mean, y_pred_lower, y_pred_upper,
            images_dir, segment_id
        )
        
        # Step 8: Future predictions
        future_predictions = predict_future_with_intervals(
            weekly_df, quantile_model, feature_cols, n_weeks=4
        )
        
        # Save outputs
        weekly_path = os.path.join(data_dir, f'{segment_id}_weekly_crashes.csv')
        weekly_df.to_csv(weekly_path, index=False)
        
        future_path = os.path.join(data_dir, f'{segment_id}_future_predictions_with_risk.csv')
        future_predictions.to_csv(future_path, index=False)
        
        metrics_path = os.path.join(data_dir, f'{segment_id}_model_metrics.csv')
        metrics.to_csv(metrics_path, index=False)
        
        print(f"\n✓ {segment_id} outputs saved to: {segment_dir}")
        
        return {
            'segment_id': segment_id,
            'status': 'SUCCESS',
            'metrics': metrics,
            'future_predictions': future_predictions
        }
        
    except Exception as e:
        print(f"\n❌ ERROR processing {segment_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'segment_id': segment_id,
            'status': 'FAILED',
            'error': str(e)
        }

# ============================================================================
# MAIN EXECUTION - LOOP THROUGH ALL SEGMENTS
# ============================================================================

def main():
    """Main pipeline for all segments"""
    
    print("\n" + "="*60)
    print("MULTI-SEGMENT CRASH FORECASTING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total segments to process: {len(SEGMENTS)}")
    print("="*60)
    
    results = []
    
    for segment_id, filepath in SEGMENTS.items():
        result = process_segment(segment_id, filepath)
        results.append(result)
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']
    
    print(f"✓ Successful: {len(successful)}/{len(SEGMENTS)}")
    print(f"✗ Failed: {len(failed)}/{len(SEGMENTS)}")
    
    if successful:
        print("\n--- Successful Segments ---")
        for result in successful:
            print(f"  {result['segment_id']}: Model trained and predictions generated")
    
    if failed:
        print("\n--- Failed Segments ---")
        for result in failed:
            print(f"  {result['segment_id']}: {result['error']}")
    
    # Create combined summary
            if successful:
                summary_df = pd.DataFrame([
                    {
                        'segment_id': r['segment_id'],
                        'MAE_baseline': r['metrics'].loc[0, 'MAE'],
                        'MAE_ensemble': r['metrics'].loc[1, 'MAE'],
                        'RMSE_baseline': r['metrics'].loc[0, 'RMSE'],
                        'RMSE_ensemble': r['metrics'].loc[1, 'RMSE'],
                        'R2_baseline': r['metrics'].loc[0, 'R²'],
                        'R2_ensemble': r['metrics'].loc[1, 'R²'],
                        'total_historical_weeks': len(r['future_predictions']) + 36,  # just for context
                        '36w_forecast_total_crashes': r['future_predictions']['predicted_mean'].sum()
                    }
                    for r in successful
                ])

                summary_path = os.path.join(BASE_OUTPUT_DIR, 'ALL_SEGMENTS_SUMMARY.csv')
                summary_df.to_csv(summary_path, index=False)
                print(f"\nCombined performance summary saved to: {summary_path}")

                # Optional: Print top performers
                print("\n" + "="*60)
                print("TOP 5 SEGMENTS BY MODEL IMPROVEMENT (MAE reduction)")
                print("="*60)
                summary_df['MAE_improvement'] = summary_df['MAE_baseline'] - summary_df['MAE_ensemble']
                top5 = summary_df.sort_values('MAE_improvement', ascending=False).head(5)
                print(top5[['segment_id', 'MAE_baseline', 'MAE_ensemble', 'MAE_improvement', '36w_forecast_total_crashes']]
                .round(2).to_string(index=False))

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"All outputs are in: {os.path.abspath(BASE_OUTPUT_DIR)}")
        print("="*60)

        return results


# ============================================================================
# RUN THE FULL PIPELINE
# ============================================================================
if __name__ == "__main__":
    # This allows running with python script.py
    final_results = main()
    
    # Optional: Save full detailed results as pickle for later analysis
    import pickle
    with open(os.path.join(BASE_OUTPUT_DIR, 'full_pipeline_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    print(f"\nDetailed results backup saved as pickle.")