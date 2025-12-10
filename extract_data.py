"""
Generate predictions from ALL models with EXACT feature sets
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATE PREDICTIONS FROM ALL MODELS (EXACT FEATURES)")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'data_file': r'D:\Course_stuff_M\Machine_Learning\Project\Data\bluebikes_ml_ready.csv',  # Has all features needed
    'linear_regression': r'D:\Course_stuff_M\Machine_Learning\Project\results\configs_weigths\optimized_lr_poly_model.pkl',  # or baseline_lr_poly_model.pkl
    'random_forest': r'D:\Course_stuff_M\Machine_Learning\Project\results\configs_weigths\optimized_random_forest.pkl',
    'neural_network_config': r'D:\Course_stuff_M\Machine_Learning\Project\results\configs_weigths\dnn_best_config.pkl',
    'neural_network_weights': r'D:\Course_stuff_M\Machine_Learning\Project\results\configs_weigths\dnn_best_weights.pkl',

    'test_size': 0.15
}

# =============================================================================
# EXACT FEATURE SETS (from model inspection)
# =============================================================================

# Random Forest: 33 features (base features only, NO weather one-hot)
RF_FEATURES = [
    'hour_of_day_sin', 'hour_of_day_cos', 'day_of_week_sin', 'day_of_week_cos',
    'month_sin', 'month_cos', 'is_weekend', 'is_peak_hour', 'is_holiday',
    'special_event_flag', 'station_latitude', 'station_longitude',
    'station_capacity', 'neighborhood_cluster_id', 'demand_t_minus_1',
    'demand_t_minus_24', 'demand_t_minus_168', 'rolling_mean_7d', 'rolling_std_7d',
    'same_hour_previous_week', 'month_to_date_average', 'day_of_week_average_4w',
    'trend_coefficient_7d', 'temperature', 'feels_like_temperature',
    'precipitation_mm', 'wind_speed_mph', 'weather_severity_score',
    'subscriber_ratio', 'average_trip_duration', 'return_trip_probability',
    'average_age_bracket', 'weather_severity_score.1'
]

# Linear Regression: 38 features (base 33 + weather one-hot 5)
LR_FEATURES = RF_FEATURES + [
    'weather_clear', 'weather_heavy_rain', 'weather_hot', 'weather_rain', 'weather_windy'
]

# Neural Network: 41 features (LR 38 + encoded 3)
NN_FEATURES = LR_FEATURES + [
    'station_id_encoded', 'station_type_encoded', 'weather_category_encoded'
]

print(f"\nFeature counts:")
print(f"  Random Forest: {len(RF_FEATURES)} features")
print(f"  Linear Regression: {len(LR_FEATURES)} features")
print(f"  Neural Network: {len(NN_FEATURES)} features")

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

df = pd.read_csv(CONFIG['data_file'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"✓ Loaded: {df.shape}")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Verify all features exist
all_features_needed = set(RF_FEATURES + LR_FEATURES + NN_FEATURES)
missing = [f for f in all_features_needed if f not in df.columns]

if missing:
    print(f"\n⚠️  Missing features in data: {missing}")
    print("  These will be filled with 0")
    for feat in missing:
        df[feat] = 0

# =============================================================================
# STEP 2: CREATE TEST SPLIT
# =============================================================================
print("\n" + "="*80)
print("STEP 2: CREATING TEST SPLIT")
print("="*80)

split_idx = int(len(df) * (1 - CONFIG['test_size']))
test_df = df.iloc[split_idx:].reset_index(drop=True)

print(f"  ✓ Test set: {len(test_df):,} samples")

y_test = test_df['demand'].values

# =============================================================================
# STEP 3: LOAD MODELS
# =============================================================================
print("\n" + "="*80)
print("STEP 3: LOADING MODELS")
print("="*80)

models = {}

# Linear Regression
try:
    with open(CONFIG['linear_regression'], 'rb') as f:
        models['linear_regression'] = pickle.load(f)
    print(f"  ✓ Linear Regression")
except Exception as e:
    print(f"  ✗ Linear Regression: {e}")

# Random Forest
try:
    with open(CONFIG['random_forest'], 'rb') as f:
        models['random_forest'] = pickle.load(f)
    print(f"  ✓ Random Forest")
except Exception as e:
    print(f"  ✗ Random Forest: {e}")

# Neural Network
try:
    with open(CONFIG['neural_network_config'], 'rb') as f:
        nn_config = pickle.load(f)
    
    scaler_mean = np.array(nn_config['scaler_mean'])
    scaler_scale = np.array(nn_config['scaler_scale'])
    arch = nn_config['architecture']
    
    class DNNModel(nn.Module):
        def __init__(self, input_dim, layers_config, dropout_rate=0.2):
            super(DNNModel, self).__init__()
            self.layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            
            prev_size = input_dim
            for size in layers_config:
                self.layers.append(nn.Linear(prev_size, size))
                self.batch_norms.append(nn.BatchNorm1d(size))
                self.dropouts.append(nn.Dropout(dropout_rate))
                prev_size = size
            
            self.output = nn.Linear(prev_size, 1)
        
        def forward(self, x):
            for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
                x = layer(x)
                x = bn(x)
                x = torch.relu(x)
                x = dropout(x)
            x = self.output(x)
            return x
    
    nn_model = DNNModel(41, arch['layers_config'], arch['dropout_rate'])
    weights = torch.load(CONFIG['neural_network_weights'], map_location='cpu')
    nn_model.load_state_dict(weights)
    nn_model.eval()
    
    models['neural_network'] = (nn_model, scaler_mean, scaler_scale)
    print(f"  ✓ Neural Network")
except Exception as e:
    print(f"  ✗ Neural Network: {e}")

print(f"\n✓ Loaded {len(models)} model(s)")

# =============================================================================
# STEP 4: GENERATE PREDICTIONS
# =============================================================================
print("\n" + "="*80)
print("STEP 4: GENERATING PREDICTIONS")
print("="*80)

predictions = {}

# Linear Regression (38 features)
if 'linear_regression' in models:
    print(f"\n  Linear Regression...")
    try:
        X_test_lr = test_df[LR_FEATURES].fillna(0).astype('float32')
        preds = models['linear_regression'].predict(X_test_lr)
        preds = np.maximum(preds, 0)
        predictions['linear_regression_prediction'] = preds
        print(f"    ✓ Generated {len(preds):,} predictions")
        print(f"    Range: {preds.min():.2f} to {preds.max():.2f}, Mean: {preds.mean():.2f}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

# Random Forest (33 features)
if 'random_forest' in models:
    print(f"\n  Random Forest...")
    try:
        X_test_rf = test_df[RF_FEATURES].fillna(0).astype('float32')
        preds = models['random_forest'].predict(X_test_rf)
        preds = np.maximum(preds, 0)
        predictions['random_forest_prediction'] = preds
        print(f"    ✓ Generated {len(preds):,} predictions")
        print(f"    Range: {preds.min():.2f} to {preds.max():.2f}, Mean: {preds.mean():.2f}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

# Neural Network (41 features)
if 'neural_network' in models:
    print(f"\n  Neural Network...")
    try:
        nn_model, scaler_mean, scaler_scale = models['neural_network']
        X_test_nn = test_df[NN_FEATURES].fillna(0).astype('float').values
        X_scaled = (X_test_nn - scaler_mean) / scaler_scale
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            preds = nn_model(X_tensor).numpy().flatten()
        
        preds = np.maximum(preds, 0)
        predictions['neural_network_prediction'] = preds
        print(f"    ✓ Generated {len(preds):,} predictions")
        print(f"    Range: {preds.min():.2f} to {preds.max():.2f}, Mean: {preds.mean():.2f}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

print(f"\n✓ Generated predictions from {len(predictions)} model(s)")

# =============================================================================
# STEP 5: CALCULATE METRICS
# =============================================================================
print("\n" + "="*80)
print("STEP 5: CALCULATING METRICS")
print("="*80)

metrics_list = []

for pred_col, preds in predictions.items():
    model_name = pred_col.replace('_prediction', '').replace('_', ' ').title()
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    metrics_list.append({
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_score': r2
    })
    
    print(f"\n  {model_name}:")
    print(f"    RMSE: {rmse:.3f}")
    print(f"    MAE:  {mae:.3f}")
    print(f"    R²:   {r2:.3f}")

# =============================================================================
# STEP 6: CREATE DASHBOARD FILE
# =============================================================================
print("\n" + "="*80)
print("STEP 6: CREATING DASHBOARD FILE")
print("="*80)

# Extract hour from cyclical feature
hour = np.round(np.arctan2(test_df['hour_of_day_sin'], test_df['hour_of_day_cos']) * 24 / (2*np.pi)) % 24

dashboard_df = pd.DataFrame({
    'station_id': test_df['station_id'],
    'hour': hour.astype(int),
    'actual_demand': y_test,
    'start_station_latitude': test_df['station_latitude'],
    'start_station_longitude': test_df['station_longitude']
})

# Add all predictions
for pred_col, preds in predictions.items():
    dashboard_df[pred_col] = preds

# Try to add station names
if 'station_name' in test_df.columns:
    dashboard_df['start_station_name'] = test_df['station_name']

print(f"  ✓ Dashboard DataFrame: {dashboard_df.shape}")
print(f"  Columns: {list(dashboard_df.columns)}")

# =============================================================================
# STEP 7: SAVE FILES
# =============================================================================
print("\n" + "="*80)
print("STEP 7: SAVING FILES")
print("="*80)

dashboard_df.to_csv('predictions_for_dashboard.csv', index=False)
print(f"  ✓ predictions_for_dashboard.csv")
print(f"    - {len(dashboard_df):,} rows")
print(f"    - {dashboard_df['station_id'].nunique()} stations")
print(f"    - Models: {list(predictions.keys())}")

station_meta = dashboard_df[['station_id', 'start_station_latitude', 'start_station_longitude']].drop_duplicates()
if 'start_station_name' in dashboard_df.columns:
    station_meta = dashboard_df[['station_id', 'start_station_name', 'start_station_latitude', 'start_station_longitude']].drop_duplicates()
station_meta = station_meta.rename(columns={
    'start_station_name': 'name',
    'start_station_latitude': 'latitude',
    'start_station_longitude': 'longitude'
})
station_meta.to_csv('station_metadata_for_dashboard.csv', index=False)
print(f"  ✓ station_metadata_for_dashboard.csv ({len(station_meta)} stations)")

# =============================================================================
# VERIFICATION
# =============================================================================
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

print("\nSample predictions (first 5 rows):")
print(dashboard_df.head().to_string(index=False))

print("\n\nModel Performance:")
print(metrics_df.to_string(index=False))

print("\n" + "="*80)
print("✅ SUCCESS! ALL 3 MODELS READY")
print("="*80)
print(f"\nGenerated predictions from {len(predictions)} models:")
for pred in predictions.keys():
    print(f"  ✓ {pred.replace('_prediction', '').replace('_', ' ').title()}")
print("\nUpload predictions_for_dashboard.csv to your dashboard!")
print("All 3 models will switch dynamically!")
print("="*80)