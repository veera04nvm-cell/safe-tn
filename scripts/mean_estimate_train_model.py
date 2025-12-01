"""
Traffic Crash Prediction Model Training Script - FIXED VERSION
===============================================
This version properly handles long-term forecasting (36+ weeks)
with hybrid ML + seasonal approach to prevent prediction convergence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = 'data/traffic_crash_merged.csv'
OUTPUT_DIR = 'data/'
IMAGES_DIR = 'images/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

def load_and_prepare_data(filepath):
    """Load hourly crash data and prepare for analysis"""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✓ Data loaded: {len(df)} hourly records")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

# ============================================================================
# STEP 2: ADVANCED FEATURE ENGINEERING
# ============================================================================

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
        'NPMRDS2 2021': ['mean', 'std'],
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

# ============================================================================
# STEP 3: LAG FEATURES
# ============================================================================

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

# ============================================================================
# STEP 4: MACHINE LEARNING MODELS
# ============================================================================

def prepare_ml_features(df, feature_cols=None):
    """Prepare features for ML models"""
    if feature_cols is None:
        exclude_cols = ['week_start', 'total_crashes', 'year']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['total_crashes']
    
    return X, y, feature_cols

def train_ensemble_models(X_train, y_train, X_test, y_test):
    """Train multiple ML models and ensemble them"""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            min_samples_split=5, random_state=42
        ),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    predictions = {}
    trained_models = {}
    
    print("\n" + "="*60)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*60)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)
        predictions[name] = y_pred
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    
    ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
    ensemble_pred = np.maximum(ensemble_pred, 0)
    
    mae_ens = mean_absolute_error(y_test, ensemble_pred)
    rmse_ens = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    r2_ens = r2_score(y_test, ensemble_pred)
    
    print(f"\n{'ENSEMBLE (Average)':}")
    print(f"  MAE: {mae_ens:.2f} | RMSE: {rmse_ens:.2f} | R²: {r2_ens:.3f}")
    
    return trained_models, ensemble_pred, predictions

# ============================================================================
# STEP 5: FEATURE IMPORTANCE
# ============================================================================

def analyze_feature_importance(model, feature_names, top_n=15):
    """Analyze and visualize feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title('Top Feature Importances')
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_path = os.path.join(IMAGES_DIR, 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance plot saved to: {output_path}")
        plt.close()
        
        print("\n" + "="*60)
        print("TOP 15 MOST IMPORTANT FEATURES")
        print("="*60)
        for i in indices:
            print(f"{feature_names[i]:.<45} {importances[i]:.4f}")

# ============================================================================
# STEP 6: COMPREHENSIVE EVALUATION
# ============================================================================

def comprehensive_evaluation(df, y_test, y_pred_baseline, y_pred_ml):
    """Compare baseline vs ML models"""
    test_df = df.iloc[-len(y_test):].copy()
    test_df['pred_baseline'] = y_pred_baseline
    test_df['pred_ml'] = y_pred_ml
    
    metrics = pd.DataFrame({
        'Model': ['Baseline (Seasonal+Rolling)', 'ML Ensemble'],
        'MAE': [
            mean_absolute_error(y_test, y_pred_baseline),
            mean_absolute_error(y_test, y_pred_ml)
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(y_test, y_pred_baseline)),
            np.sqrt(mean_squared_error(y_test, y_pred_ml))
        ],
        'R²': [
            r2_score(y_test, y_pred_baseline),
            r2_score(y_test, y_pred_ml)
        ],
        'MAPE': [
            np.mean(np.abs((y_test - y_pred_baseline) / y_test.replace(0, 1))) * 100,
            np.mean(np.abs((y_test - y_pred_ml) / y_test.replace(0, 1))) * 100
        ]
    })
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(metrics.to_string(index=False))
    
    improvement_r2 = ((metrics.loc[1, 'R²'] - metrics.loc[0, 'R²']) / 
                      abs(metrics.loc[0, 'R²']) * 100)
    improvement_mae = ((metrics.loc[0, 'MAE'] - metrics.loc[1, 'MAE']) / 
                       metrics.loc[0, 'MAE'] * 100)
    
    print(f"\n✓ Improvements with ML:")
    print(f"  R² improvement: {improvement_r2:+.1f}%")
    print(f"  MAE improvement: {improvement_mae:+.1f}%")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    axes[0].plot(test_df['week_start'], y_test.values, 
                 label='Actual', color='black', linewidth=2, alpha=0.8)
    axes[0].plot(test_df['week_start'], y_pred_baseline, 
                 label='Baseline Model', color='blue', linestyle='--', alpha=0.7)
    axes[0].plot(test_df['week_start'], y_pred_ml, 
                 label='ML Ensemble', color='red', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Weekly Crashes')
    axes[0].set_title('Model Comparison: Actual vs Predictions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test, y_pred_baseline, alpha=0.5, label='Baseline', s=50)
    axes[1].scatter(y_test, y_pred_ml, alpha=0.5, label='ML Ensemble', s=50)
    max_val = max(y_test.max(), y_pred_ml.max())
    axes[1].plot([0, max_val], [0, max_val], 'k--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Weekly Crashes')
    axes[1].set_ylabel('Predicted Weekly Crashes')
    axes[1].set_title('Prediction Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Model comparison plot saved to: {output_path}")
    plt.close()
    
    return metrics

# ============================================================================
# STEP 7: FUTURE PREDICTIONS - HYBRID APPROACH (FIXED!)
# ============================================================================

def predict_future_hybrid(df, models, feature_cols, n_weeks=36):
    """
    FIXED: Hybrid ML + Seasonal approach prevents prediction convergence
    """
    extended_df = df.copy()
    last_date = df['week_start'].max()
    
    # Extract historical seasonal patterns
    print("\n" + "="*60)
    print(f"ANALYZING HISTORICAL SEASONAL PATTERNS")
    print("="*60)
    
    seasonal_stats = df.groupby('week_of_year')['total_crashes'].agg(['mean', 'std', 'min', 'max'])
    overall_mean = df['total_crashes'].mean()
    overall_std = df['total_crashes'].std()
    
    print(f"Overall average: {overall_mean:.2f} crashes/week")
    print(f"Seasonal variation: {seasonal_stats['mean'].min():.2f} to {seasonal_stats['mean'].max():.2f}")
    
    print("\n" + "="*60)
    print(f"GENERATING {n_weeks}-WEEK HYBRID FORECAST")
    print("="*60)
    print("Method: ML (weeks 1-12) → Hybrid (weeks 13-24) → Seasonal (weeks 25+)")
    print("="*60)
    
    future_predictions = []
    
    for i in range(n_weeks):
        next_date = last_date + pd.Timedelta(weeks=i+1)
        week_num = next_date.isocalendar().week
        
        # Create new row
        new_row = pd.Series(index=extended_df.columns, dtype='float64')
        new_row['week_start'] = next_date
        new_row['month'] = next_date.month
        new_row['quarter'] = next_date.quarter
        new_row['week_of_year'] = week_num
        new_row['week_sin'] = np.sin(2 * np.pi * week_num / 52)
        new_row['week_cos'] = np.cos(2 * np.pi * week_num / 52)
        new_row['month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
        
        # Copy static features
        last_row = extended_df.iloc[-1]
        for col in ['avg_speed', 'std_speed', 'min_speed', 'max_speed', 'avg_npmrds', 
                    'std_npmrds', 'avg_speed_deviation', 'max_speed_deviation', 
                    'avg_speed_ratio', 'min_speed_ratio', 'total_congestion_hours',
                    'total_weekend_hours', 'total_rush_hours', 'total_night_hours',
                    'total_peak_crash_hours', 'speed_variability', 'speed_range',
                    'pct_congested', 'pct_rush_hour']:
            if col in extended_df.columns:
                new_row[col] = last_row[col]
        
        # Update lag features
        for lag in [1, 2, 3, 4, 8, 12]:
            col_name = f'crashes_lag_{lag}w'
            if col_name in feature_cols and lag <= len(extended_df):
                new_row[col_name] = extended_df.iloc[-lag]['total_crashes']
            elif col_name in feature_cols:
                new_row[col_name] = overall_mean
        
        # Update rolling features
        for window in [2, 4, 8, 12]:
            if window <= len(extended_df):
                recent = extended_df.tail(window)['total_crashes']
                new_row[f'crashes_rolling_mean_{window}w'] = recent.mean()
                new_row[f'crashes_rolling_std_{window}w'] = recent.std() if len(recent) > 1 else 0
                new_row[f'crashes_rolling_max_{window}w'] = recent.max()
        
        # Update change features
        if len(extended_df) >= 2:
            new_row['crash_change_1w'] = extended_df.iloc[-1]['total_crashes'] - extended_df.iloc[-2]['total_crashes']
        if len(extended_df) >= 4:
            new_row['crash_change_4w'] = extended_df.iloc[-1]['total_crashes'] - extended_df.iloc[-4]['total_crashes']
        
        # Fill any remaining NaNs
        new_row = new_row.fillna(0)
        
        # ====================================================================
        # GET ML PREDICTION
        # ====================================================================
        X_new = new_row[feature_cols].values.reshape(1, -1)
        X_new = np.nan_to_num(X_new, nan=0.0)
        
        ml_preds = [model.predict(X_new)[0] for model in models.values()]
        ml_pred = np.mean(ml_preds)
        ml_pred = max(ml_pred, 0)
        
        # ====================================================================
        # GET SEASONAL PREDICTION
        # ====================================================================
        if week_num in seasonal_stats.index:
            seasonal_mean = seasonal_stats.loc[week_num, 'mean']
            seasonal_std = seasonal_stats.loc[week_num, 'std']
        else:
            seasonal_mean = overall_mean
            seasonal_std = overall_std
        
        # ====================================================================
        # HYBRID BLENDING - THIS IS THE KEY FIX!
        # ====================================================================
        if i < 4:
            # Weeks 1-4: Pure ML (100% confidence)
            final_pred = ml_pred
            method = "ML"
        elif i < 12:
            # Weeks 5-12: 80-100% ML, 0-20% seasonal
            weight = (i - 4) / 8 * 0.2  # 0% → 20%
            final_pred = ml_pred * (1 - weight) + seasonal_mean * weight
            method = "ML+Season"
        elif i < 24:
            # Weeks 13-24: 50-80% ML, 20-50% seasonal
            weight = 0.2 + (i - 12) / 12 * 0.3  # 20% → 50%
            final_pred = ml_pred * (1 - weight) + seasonal_mean * weight
            method = "Hybrid"
            # Add noise to prevent convergence
            noise = np.random.normal(0, seasonal_std * 0.15)
            final_pred += noise
        else:
            # Weeks 25+: 20-50% ML, 50-80% seasonal
            weight = 0.5 + min((i - 24) / 12 * 0.3, 0.3)  # 50% → 80%
            final_pred = ml_pred * (1 - weight) + seasonal_mean * weight
            method = "Seasonal"
            # More noise for long-term
            noise = np.random.normal(0, seasonal_std * 0.2)
            final_pred += noise
        
        final_pred = max(final_pred, 0)
        
        # ====================================================================
        # UPDATE AND STORE
        # ====================================================================
        new_row['total_crashes'] = final_pred
        extended_df = pd.concat([extended_df, pd.DataFrame([new_row])], ignore_index=True)
        
        future_predictions.append({
            'week_start': next_date,
            'week_of_year': week_num,
            'predicted_crashes': round(final_pred, 1),
            'month': next_date.strftime('%B'),
            'method': method
        })
        
        # Print with method indicator
        print(f"{next_date.strftime('%Y-%m-%d')} (Week {week_num:2d}): {final_pred:.1f} crashes [{method}]")
    
    future_df = pd.DataFrame(future_predictions)
    
    print("\n" + "="*60)
    print("FORECAST SUMMARY")
    print("="*60)
    print(f"Total predicted: {future_df['predicted_crashes'].sum():.1f} crashes")
    print(f"Weekly average: {future_df['predicted_crashes'].mean():.1f} crashes")
    print(f"Range: {future_df['predicted_crashes'].min():.1f} - {future_df['predicted_crashes'].max():.1f}")
    print(f"Variation (std): {future_df['predicted_crashes'].std():.2f}")
    print("="*60)
    
    return future_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("TRAFFIC CRASH PREDICTION MODEL TRAINING")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # Load data
        print("\n[1/8] Loading and preprocessing data...")
        df = load_and_prepare_data(DATA_PATH)
        
        # Feature engineering
        print("\n[2/8] Creating enhanced features...")
        df = create_enhanced_features(df)
        
        # Weekly aggregation
        print("\n[3/8] Aggregating to weekly...")
        weekly_df = aggregate_to_weekly_advanced(df)
        
        # Add lag features
        print("\n[4/8] Adding lag and rolling features...")
        weekly_df = add_lag_features(weekly_df)
        
        # Prepare for modeling
        split_idx = int(len(weekly_df) * 0.75)
        train_df = weekly_df.iloc[:split_idx].copy()
        test_df = weekly_df.iloc[split_idx:].copy()
        
        seasonal_avg = train_df.groupby('week_of_year')['total_crashes'].mean()
        test_df['baseline_pred'] = test_df['week_of_year'].map(seasonal_avg)
        
        # Prepare ML features
        X, y, feature_cols = prepare_ml_features(weekly_df)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        
        # Train ML models
        print("\n[5/8] Training machine learning models...")
        models, ensemble_pred, all_preds = train_ensemble_models(
            X_train, y_train, X_test, y_test
        )
        
        # Feature importance
        print("\n[6/8] Analyzing feature importance...")
        analyze_feature_importance(models['Random Forest'], feature_cols)
        
        # Comprehensive evaluation
        print("\n[7/8] Comprehensive evaluation...")
        metrics = comprehensive_evaluation(
            weekly_df, y_test, test_df['baseline_pred'].values, ensemble_pred
        )
        
        # Future predictions - USING FIXED HYBRID METHOD
        print("\n[8/8] Predicting future weeks...")
        future_predictions = predict_future_hybrid(weekly_df, models, feature_cols, n_weeks=36)
        
        # Save results
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        weekly_path = os.path.join(OUTPUT_DIR, 'weekly_crashes_enhanced.csv')
        weekly_df.to_csv(weekly_path, index=False)
        print(f"✓ Saved: {weekly_path}")
        
        future_path = os.path.join(OUTPUT_DIR, 'future_predictions_ml.csv')
        future_predictions.to_csv(future_path, index=False)
        print(f"✓ Saved: {future_path}")
        
        metrics_path = os.path.join(OUTPUT_DIR, 'model_performance_comparison.csv')
        metrics.to_csv(metrics_path, index=False)
        print(f"✓ Saved: {metrics_path}")
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nNext step: Run the Streamlit dashboard")
        print("Command: streamlit run app/streamlit_app.py")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Data file not found!")
        print(f"Expected location: {DATA_PATH}")
        print(f"Please ensure your data file is in the correct location.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()