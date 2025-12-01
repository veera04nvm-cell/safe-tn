"""
Traffic Crash Prediction Model Training Script
===============================================
This script trains the ML models and generates predictions.
Run this script whenever you have new data or want to update predictions.

Usage:
    python scripts/train_model.py
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

# File paths (relative to project root)
DATA_PATH = 'data/traffic_crash_merged.csv'
OUTPUT_DIR = 'data/'
IMAGES_DIR = 'images/'

# Ensure output directories exist
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
    """
    Create comprehensive features from hourly data before weekly aggregation
    """
    print("Creating enhanced features...")
    df = df.copy()
    
    # Speed-related features
    df['speed_deviation'] = np.abs(df['speed'] - df['historical_average_speed'])
    df['speed_ratio'] = df['speed'] / df['reference_speed'].replace(0, 1)
    df['congestion_indicator'] = (df['speed'] < df['reference_speed'] * 0.7).astype(int)
    
    # Time-based risk indicators
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df['is_peak_crash_hour'] = df['hour'].isin([15, 16, 17, 18]).astype(int)
    
    print(f"✓ Created {len([c for c in df.columns if c not in ['timestamp', 'crash_count']])} features")
    
    return df

def aggregate_to_weekly_advanced(df):
    """
    Aggregate to weekly with rich feature set
    """
    print("Aggregating to weekly data...")
    df = df.set_index('timestamp')
    
    # Comprehensive aggregation
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
    
    # Flatten column names
    weekly_df.columns = ['week_start', 'total_crashes',
                         'avg_speed', 'std_speed', 'min_speed', 'max_speed',
                         'avg_npmrds', 'std_npmrds',
                         'avg_speed_deviation', 'max_speed_deviation',
                         'avg_speed_ratio', 'min_speed_ratio',
                         'total_congestion_hours', 'total_weekend_hours',
                         'total_rush_hours', 'total_night_hours', 
                         'total_peak_crash_hours']
    
    # Derived features
    weekly_df['speed_variability'] = weekly_df['std_speed'] / weekly_df['avg_speed'].replace(0, 1)
    weekly_df['speed_range'] = weekly_df['max_speed'] - weekly_df['min_speed']
    weekly_df['pct_congested'] = weekly_df['total_congestion_hours'] / 168  # 168 hours/week
    weekly_df['pct_rush_hour'] = weekly_df['total_rush_hours'] / 168
    
    # Temporal features
    weekly_df['year'] = weekly_df['week_start'].dt.year
    weekly_df['month'] = weekly_df['week_start'].dt.month
    weekly_df['quarter'] = weekly_df['week_start'].dt.quarter
    weekly_df['week_of_year'] = weekly_df['week_start'].dt.isocalendar().week
    
    # Cyclical encoding for seasonality
    weekly_df['week_sin'] = np.sin(2 * np.pi * weekly_df['week_of_year'] / 52)
    weekly_df['week_cos'] = np.cos(2 * np.pi * weekly_df['week_of_year'] / 52)
    weekly_df['month_sin'] = np.sin(2 * np.pi * weekly_df['month'] / 12)
    weekly_df['month_cos'] = np.cos(2 * np.pi * weekly_df['month'] / 12)
    
    print(f"✓ Created {len(weekly_df)} weekly records")
    
    return weekly_df

# ============================================================================
# STEP 3: LAG FEATURES (HISTORICAL PATTERNS)
# ============================================================================

def add_lag_features(df, lag_weeks=[1, 2, 3, 4, 8, 12]):
    """
    Add lagged crash counts and rolling statistics
    """
    print("Adding lag and rolling features...")
    df = df.copy()
    
    # Lag features
    for lag in lag_weeks:
        df[f'crashes_lag_{lag}w'] = df['total_crashes'].shift(lag)
    
    # Rolling window features
    for window in [2, 4, 8, 12]:
        df[f'crashes_rolling_mean_{window}w'] = df['total_crashes'].rolling(
            window=window, min_periods=1).mean()
        df[f'crashes_rolling_std_{window}w'] = df['total_crashes'].rolling(
            window=window, min_periods=1).std()
        df[f'crashes_rolling_max_{window}w'] = df['total_crashes'].rolling(
            window=window, min_periods=1).max()
    
    # Rate of change
    df['crash_change_1w'] = df['total_crashes'].diff(1)
    df['crash_change_4w'] = df['total_crashes'].diff(4)
    
    # Fill NaN values from lagging with 0
    df = df.fillna(0)
    
    print(f"✓ Total features: {len(df.columns)}")
    
    return df

# ============================================================================
# STEP 4: MACHINE LEARNING MODELS
# ============================================================================

def prepare_ml_features(df, feature_cols=None):
    """
    Prepare features for ML models
    """
    if feature_cols is None:
        # Exclude date and target columns
        exclude_cols = ['week_start', 'total_crashes', 'year']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['total_crashes']
    
    return X, y, feature_cols

def train_ensemble_models(X_train, y_train, X_test, y_test):
    """
    Train multiple ML models and ensemble them
    """
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
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
        predictions[name] = y_pred
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
    
    # Ensemble prediction (average of all models)
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
    """
    Analyze and visualize feature importance
    """
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
    """
    Compare baseline vs ML models
    """
    test_df = df.iloc[-len(y_test):].copy()
    test_df['pred_baseline'] = y_pred_baseline
    test_df['pred_ml'] = y_pred_ml
    
    # Metrics comparison
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
    
    # Improvement calculation
    improvement_r2 = ((metrics.loc[1, 'R²'] - metrics.loc[0, 'R²']) / 
                      abs(metrics.loc[0, 'R²']) * 100)
    improvement_mae = ((metrics.loc[0, 'MAE'] - metrics.loc[1, 'MAE']) / 
                       metrics.loc[0, 'MAE'] * 100)
    
    print(f"\n✓ Improvements with ML:")
    print(f"  R² improvement: {improvement_r2:+.1f}%")
    print(f"  MAE improvement: {improvement_mae:+.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series comparison
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
    
    # Scatter plots
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
# STEP 7: FUTURE PREDICTIONS WITH ML (IMPROVED HYBRID APPROACH)
# ============================================================================

def predict_future_hybrid(df, models, feature_cols, n_weeks=36):
    """
    Hybrid approach: Use ML for short-term + Seasonal patterns for long-term
    This prevents prediction convergence and maintains realistic variation
    """
    extended_df = df.copy()
    last_date = df['week_start'].max()
    
    # Calculate seasonal patterns from historical data
    seasonal_patterns = df.groupby('week_of_year').agg({
        'total_crashes': ['mean', 'std']
    }).reset_index()
    seasonal_patterns.columns = ['week_of_year', 'seasonal_mean', 'seasonal_std']
    
    overall_mean = df['total_crashes'].mean()
    overall_std = df['total_crashes'].std()
    
    print("\n" + "="*60)
    print(f"HYBRID FORECAST - ML + SEASONAL (Next {n_weeks} weeks)")
    print("="*60)
    
    future_predictions = []
    
    for i in range(n_weeks):
        next_date = last_date + pd.Timedelta(weeks=i+1)
        
        # Create new row with temporal features
        new_row = pd.Series(index=extended_df.columns)
        new_row['week_start'] = next_date
        new_row['month'] = next_date.month
        new_row['quarter'] = next_date.quarter
        new_row['week_of_year'] = next_date.isocalendar().week
        new_row['week_sin'] = np.sin(2 * np.pi * new_row['week_of_year'] / 52)
        new_row['week_cos'] = np.cos(2 * np.pi * new_row['week_of_year'] / 52)
        new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
        
        # Copy non-temporal features
        last_row = extended_df.iloc[-1]
        for col in extended_df.columns:
            if col not in new_row or pd.isna(new_row[col]):
                if col not in ['week_start', 'total_crashes', 'month', 'quarter', 
                              'week_of_year', 'week_sin', 'week_cos', 'month_sin', 
                              'month_cos'] and not col.startswith('crashes_'):
                    new_row[col] = last_row[col]
        
        # Update lag features
        for lag in [1, 2, 3, 4, 8, 12]:
            if f'crashes_lag_{lag}w' in feature_cols:
                if lag <= len(extended_df):
                    new_row[f'crashes_lag_{lag}w'] = extended_df.iloc[-lag]['total_crashes']
                else:
                    new_row[f'crashes_lag_{lag}w'] = 0
        
        # Update rolling statistics
        for window in [2, 4, 8, 12]:
            if f'crashes_rolling_mean_{window}w' in feature_cols:
                recent_crashes = extended_df.tail(window)['total_crashes']
                new_row[f'crashes_rolling_mean_{window}w'] = recent_crashes.mean()
                new_row[f'crashes_rolling_std_{window}w'] = recent_crashes.std()
                new_row[f'crashes_rolling_max_{window}w'] = recent_crashes.max()
        
        # Update change features
        if 'crash_change_1w' in feature_cols:
            new_row['crash_change_1w'] = (extended_df.iloc[-1]['total_crashes'] - 
                                          extended_df.iloc[-2]['total_crashes'])
        if 'crash_change_4w' in feature_cols and len(extended_df) >= 4:
            new_row['crash_change_4w'] = (extended_df.iloc[-1]['total_crashes'] - 
                                          extended_df.iloc[-4]['total_crashes'])
        
        # Get ML predictions
        X_new = new_row[feature_cols].values.reshape(1, -1)
        X_new = np.nan_to_num(X_new, nan=0.0)
        
        ml_predictions = []
        for name, model in models.items():
            pred = model.predict(X_new)[0]
            ml_predictions.append(pred)
        
        ml_pred = np.mean(ml_predictions)
        ml_pred = max(ml_pred, 0)
        
        # Get seasonal prediction
        week_num = int(new_row['week_of_year'])
        seasonal_row = seasonal_patterns[seasonal_patterns['week_of_year'] == week_num]
        
        if len(seasonal_row) > 0:
            seasonal_pred = seasonal_row['seasonal_mean'].values[0]
            seasonal_std = seasonal_row['seasonal_std'].values[0]
        else:
            seasonal_pred = overall_mean
            seasonal_std = overall_std
        
        # Hybrid blending based on forecast horizon
        if i < 4:  # Weeks 1-4: Pure ML
            final_pred = ml_pred
        elif i < 12:  # Weeks 5-12: Mostly ML
            blend_weight = (i - 4) / 8 * 0.2
            final_pred = ml_pred * (1 - blend_weight) + seasonal_pred * blend_weight
        elif i < 24:  # Weeks 13-24: Balanced
            blend_weight = 0.2 + (i - 12) / 12 * 0.3
            final_pred = ml_pred * (1 - blend_weight) + seasonal_pred * blend_weight
        else:  # Weeks 25+: Mostly seasonal
            blend_weight = 0.5 + (i - 24) / 12 * 0.3
            blend_weight = min(blend_weight, 0.8)
            final_pred = ml_pred * (1 - blend_weight) + seasonal_pred * blend_weight
        
        # Add variation for long-term forecasts
        if i >= 12:
            noise = np.random.normal(0, seasonal_std * 0.1)
            final_pred = final_pred + noise
        
        final_pred = max(final_pred, 0)
        
        # Update the row
        new_row['total_crashes'] = final_pred
        extended_df = pd.concat([extended_df, new_row.to_frame().T], ignore_index=True)
        
        # Store prediction
        future_predictions.append({
            'week_start': next_date,
            'week_of_year': week_num,
            'predicted_crashes': round(final_pred, 1),
            'month': next_date.strftime('%B')
        })
        
        print(f"{next_date.strftime('%Y-%m-%d')} (Week {week_num:2d}): {final_pred:.1f} crashes")
    
    future_df = pd.DataFrame(future_predictions)
    print(f"\n✓ Total: {future_df['predicted_crashes'].sum():.1f} | "
          f"Avg: {future_df['predicted_crashes'].mean():.1f}")
    
    return future_df

def predict_future_with_ml(df, models, feature_cols, n_weeks=12):
    """
    Predict future weeks using trained ML models with iterative lag updates
    Includes seasonal adjustment for long-term forecasts
    """
    # Create a copy to avoid modifying original
    extended_df = df.copy()
    
    last_date = df['week_start'].max()
    
    # Calculate historical seasonal patterns for adjustment
    historical_weekly_avg = df.groupby('week_of_year')['total_crashes'].mean()
    overall_avg = df['total_crashes'].mean()
    
    print("\n" + "="*60)
    print(f"FUTURE PREDICTIONS - ML ENSEMBLE (Next {n_weeks} weeks)")
    print("="*60)
    
    future_predictions = []
    
    for i in range(n_weeks):
        # Calculate next week's date
        next_date = last_date + pd.Timedelta(weeks=i+1)
        
        # Create new row with temporal features
        new_row = pd.Series(index=extended_df.columns)
        new_row['week_start'] = next_date
        new_row['month'] = next_date.month
        new_row['quarter'] = next_date.quarter
        new_row['week_of_year'] = next_date.isocalendar().week
        new_row['week_sin'] = np.sin(2 * np.pi * new_row['week_of_year'] / 52)
        new_row['week_cos'] = np.cos(2 * np.pi * new_row['week_of_year'] / 52)
        new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
        
        # Copy non-temporal features from last known row
        last_row = extended_df.iloc[-1]
        for col in extended_df.columns:
            if col not in new_row or pd.isna(new_row[col]):
                if col not in ['week_start', 'total_crashes', 'month', 'quarter', 
                              'week_of_year', 'week_sin', 'week_cos', 'month_sin', 
                              'month_cos'] and not col.startswith('crashes_'):
                    new_row[col] = last_row[col]
        
        # Update lag features based on recent history
        for lag in [1, 2, 3, 4, 8, 12]:
            if f'crashes_lag_{lag}w' in feature_cols:
                if lag <= len(extended_df):
                    new_row[f'crashes_lag_{lag}w'] = extended_df.iloc[-lag]['total_crashes']
                else:
                    new_row[f'crashes_lag_{lag}w'] = 0
        
        # Update rolling statistics
        for window in [2, 4, 8, 12]:
            if f'crashes_rolling_mean_{window}w' in feature_cols:
                recent_crashes = extended_df.tail(window)['total_crashes']
                new_row[f'crashes_rolling_mean_{window}w'] = recent_crashes.mean()
                new_row[f'crashes_rolling_std_{window}w'] = recent_crashes.std()
                new_row[f'crashes_rolling_max_{window}w'] = recent_crashes.max()
        
        # Update change features
        if 'crash_change_1w' in feature_cols:
            new_row['crash_change_1w'] = (extended_df.iloc[-1]['total_crashes'] - 
                                          extended_df.iloc[-2]['total_crashes'])
        if 'crash_change_4w' in feature_cols and len(extended_df) >= 4:
            new_row['crash_change_4w'] = (extended_df.iloc[-1]['total_crashes'] - 
                                          extended_df.iloc[-4]['total_crashes'])
        
        # Prepare features for prediction
        X_new = new_row[feature_cols].values.reshape(1, -1)
        
        # Handle any remaining NaN values
        X_new = np.nan_to_num(X_new, nan=0.0)
        
        # Get predictions from all models
        predictions = []
        for name, model in models.items():
            pred = model.predict(X_new)[0]
            predictions.append(pred)
        
        # Ensemble average
        ensemble_pred = np.mean(predictions)
        ensemble_pred = max(ensemble_pred, 0)  # Ensure non-negative
        
        # Apply seasonal adjustment for longer forecasts (improves variation)
        if i >= 12:  # After 12 weeks, add seasonal variation
            week_num = int(new_row['week_of_year'])
            seasonal_factor = historical_weekly_avg.get(week_num, overall_avg) / overall_avg
            # Blend model prediction with seasonal pattern
            blend_weight = min((i - 12) / 24, 0.3)  # Gradually increase seasonal influence
            ensemble_pred = ensemble_pred * (1 - blend_weight) + (ensemble_pred * seasonal_factor * blend_weight)
            ensemble_pred = max(ensemble_pred, 0)
        
        # Update the row with prediction
        new_row['total_crashes'] = ensemble_pred
        
        # Append to extended dataframe for next iteration
        extended_df = pd.concat([extended_df, new_row.to_frame().T], ignore_index=True)
        
        # Store prediction
        future_predictions.append({
            'week_start': next_date,
            'week_of_year': new_row['week_of_year'],
            'predicted_crashes': round(ensemble_pred, 1),
            'month': next_date.strftime('%B')
        })
        
        print(f"{next_date.strftime('%Y-%m-%d')} (Week {int(new_row['week_of_year']):2d}): "
              f"{ensemble_pred:.1f} crashes")
    
    future_df = pd.DataFrame(future_predictions)
    print(f"\n✓ Total predicted crashes: {future_df['predicted_crashes'].sum():.1f}")
    print(f"✓ Average weekly crashes: {future_df['predicted_crashes'].mean():.1f}")
    
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
        
        # Weekly aggregation with rich features
        print("\n[3/8] Aggregating to weekly with advanced features...")
        weekly_df = aggregate_to_weekly_advanced(df)
        
        # Add lag features
        print("\n[4/8] Adding lag and rolling features...")
        weekly_df = add_lag_features(weekly_df)
        
        # Prepare for modeling
        split_idx = int(len(weekly_df) * 0.75)
        train_df = weekly_df.iloc[:split_idx].copy()
        test_df = weekly_df.iloc[split_idx:].copy()
        
        # Get baseline predictions
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
        
        # Future predictions
        print("\n[8/8] Predicting future weeks...")
        future_predictions = predict_future_with_ml(weekly_df, models, feature_cols, n_weeks=36)
        
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