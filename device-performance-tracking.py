import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

# Set up joblib for caching
if not os.path.exists('plots_xgboost_optimized_2000000_track'):
    os.makedirs('plots_xgboost_optimized_2000000_track')
    print("'plots_xgboost_optimized_2000000_track' folder created.")

# Function to load and preprocess data for a specific device
def load_device_data(device_id):
    file_path = f'csv/measures_24h_before_events_device_{device_id}.csv'
    data = pd.read_csv(file_path, parse_dates=['measure_date'])
    
    # Convert 'value' column to numeric, forcing errors to NaN
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    
    # Feature engineering
    data['hour'] = data['measure_date'].dt.hour
    data['day_of_week'] = data['measure_date'].dt.dayofweek
    data['month'] = data['measure_date'].dt.month
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Lag features and rolling statistics
    data = data.sort_values('measure_date')
    data['value_lag_1'] = data.groupby('event_variable')['value'].shift(1)
    data['value_rolling_mean'] = data.groupby('event_variable')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
    
    # Drop rows with NaN values in the features or target
    data = data.dropna(subset=['value'] + ['value_lag_1', 'value_rolling_mean'])
    
    feature_columns = ['event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend', 
                       'value_lag_1', 'value_rolling_mean']
    
    # Prepare data for modeling
    X = data[feature_columns]
    y = (data['value'] > data['value'].quantile(0.75)).astype(int)
    
    return X, y

# Hyperparameter search space
xgb_param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'scale_pos_weight': [1, 3, 5, 7, 9],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [0, 0.1, 1, 10]
}

device_ids = ['18', '19', '63','123', ' 147']  # Lista dei device ID

# Simuliamo più iterazioni (o epoche) per ogni dispositivo
iterations = list(range(1, 11))  # Supponiamo 10 iterazioni

# Metrics collection
performance_metrics = {
    'device_id': [],
    'iteration': [],
    'accuracy': [],
    'f1_score': [],
    'roc_auc_score': []
}

for device_id in device_ids:
    for iteration in iterations:
        X, y = load_device_data(device_id)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1, 100), stratify=y)

        # Scale the data
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # XGBoost model and hyperparameter search
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=4)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_distributions, 
                                           n_iter=30, cv=cv, n_jobs=4, scoring='f1_macro', random_state=42)
        random_search.fit(X_train_scaled, y_train)

        best_model = random_search.best_estimator_

        # Model evaluation
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        # Collect metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        performance_metrics['device_id'].append(device_id)
        performance_metrics['iteration'].append(iteration)
        performance_metrics['accuracy'].append(accuracy)
        performance_metrics['f1_score'].append(f1)
        performance_metrics['roc_auc_score'].append(roc_auc)
        
        print(f"Device {device_id}, Iteration {iteration} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {roc_auc:.4f}")

# Convert to DataFrame
metrics_df = pd.DataFrame(performance_metrics)
metrics_df.to_csv('metrics_results.csv', index=False)  # Salva i risultati in un CSV

# Plotting results for each device
for device_id in device_ids:
    device_data = metrics_df[metrics_df['device_id'] == device_id]
    
    plt.figure(figsize=(10, 6))

    # Plot con linee continue
    plt.plot(device_data['iteration'], device_data['accuracy'], marker='o', linestyle='-', color='blue', label='Accuracy')
    plt.plot(device_data['iteration'], device_data['f1_score'], marker='s', linestyle='-', color='green', label='F1 Score')
    plt.plot(device_data['iteration'], device_data['roc_auc_score'], marker='^', linestyle='-', color='red', label='ROC-AUC')

    plt.title(f"Performance del Modello per Dispositivo {device_id}")
    plt.xlabel("Iterazione del Modello")
    plt.ylabel("Metriche di Performance")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Salva il grafico per ogni dispositivo
    plt.savefig(f'plots_xgboost_optimized_2000000/smoothed_performance_chart_device_{device_id}.png')
    plt.show()
