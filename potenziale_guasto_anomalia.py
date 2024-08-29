import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb

# Set up joblib for caching
if not os.path.exists('plots_xgboost_optimized_fault'):
    os.makedirs('plots_xgboost_optimized_fault')
    print("'plots_xgboost_optimized_fault' folder created.")

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
    data['value_rolling_std'] = data.groupby('event_variable')['value'].rolling(window=24, min_periods=1).std().reset_index(0, drop=True)

    # Drop rows with NaN values in the features or target
    data = data.dropna(subset=['value'] + ['value_lag_1', 'value_rolling_mean', 'value_rolling_std'])

    feature_columns = ['event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend', 
                       'value_lag_1', 'value_rolling_mean', 'value_rolling_std']

    # Prepare data for modeling
    X = data[feature_columns]
    
    # Adjust the threshold for anomaly detection (e.g., 95th percentile instead of 75th)
    y = (data['value'] > data['value'].quantile(0.95)).astype(int)

    return X, y

# Hyperparameter search space
xgb_param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

device_ids = ['18', '19', '63', '123', '147']
iterations = list(range(1, 11))  # Reduced number of iterations for faster results

# Metrics collection
performance_metrics = {
    'device_id': [],
    'iteration': [],
    'accuracy': [],
    'f1_score': [],
    'roc_auc_score': [],
    'anomaly_classification': [],
    'fault_classification': [],
    'true_positives': [],
    'false_positives': [],
    'false_negatives': [],
    'true_negatives': []
}

fault_threshold = 0.8  # Increased threshold for potential fault detection

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
        xgb_model = xgb.XGBClassifier(random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_distributions, 
                                           n_iter=20, cv=cv, n_jobs=-1, scoring='f1', random_state=42)
        random_search.fit(X_train_scaled, y_train)

        best_model = random_search.best_estimator_

        # Model evaluation
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Classifica come "anomalia" se y_pred è 1
        anomaly_classification = "Anomaly Detected" if 1 in y_pred else "No Anomaly"

        # Classifica come "guasto" se la probabilità supera la soglia
        fault_classification = "Potential Fault Detected" if any(y_pred_proba >= fault_threshold) else "No Fault"

        performance_metrics['device_id'].append(device_id)
        performance_metrics['iteration'].append(iteration)
        performance_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        performance_metrics['f1_score'].append(f1_score(y_test, y_pred))
        performance_metrics['roc_auc_score'].append(roc_auc_score(y_test, y_pred_proba))
        performance_metrics['anomaly_classification'].append(anomaly_classification)
        performance_metrics['fault_classification'].append(fault_classification)
        performance_metrics['true_positives'].append(tp)
        performance_metrics['false_positives'].append(fp)
        performance_metrics['false_negatives'].append(fn)
        performance_metrics['true_negatives'].append(tn)

        print(f"Device {device_id}, Iteration {iteration} - {anomaly_classification}, {fault_classification}")
        print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}, True Negatives: {tn}")

# Convert to DataFrame
metrics_df = pd.DataFrame(performance_metrics)
metrics_df.to_csv('metrics_results.csv', index=False)

# Plotting results for each device
for device_id in device_ids:
    device_data = metrics_df[metrics_df['device_id'] == device_id]

    plt.figure(figsize=(12, 8))

    # Plot con linee continue
    plt.plot(device_data['iteration'], device_data['accuracy'], marker='o', linestyle='-', color='blue', label='Accuracy')
    plt.plot(device_data['iteration'], device_data['f1_score'], marker='s', linestyle='-', color='green', label='F1 Score')
    plt.plot(device_data['iteration'], device_data['roc_auc_score'], marker='^', linestyle='-', color='red', label='ROC-AUC')

    # Plot confusion matrix metrics
    plt.plot(device_data['iteration'], device_data['true_positives'] / (device_data['true_positives'] + device_data['false_negatives']), 
             marker='D', linestyle='-', color='purple', label='True Positive Rate')
    plt.plot(device_data['iteration'], device_data['false_positives'] / (device_data['false_positives'] + device_data['true_negatives']), 
             marker='x', linestyle='-', color='orange', label='False Positive Rate')

    plt.title(f"Model Performance for Device {device_id}")
    plt.xlabel("Model Iteration")
    plt.ylabel("Performance Metrics")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot for each device
    plt.savefig(f'plots_xgboost_optimized_fault/performance_chart_device_{device_id}.png')
    plt.close()

print("Analysis complete. Check the 'plots_xgboost_optimized_fault' folder for performance charts.")
