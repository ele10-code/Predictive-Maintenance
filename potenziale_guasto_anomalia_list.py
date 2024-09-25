import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from joblib import Memory
import glob

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.covariance")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn")

# Set up joblib for caching
memory = Memory(location='.joblib_cache', verbose=0)

# Create 'plots' folder if it doesn't exist
for device_id in range(93, 118):
    plot_folder = f'plots_xgboost_device_{device_id}'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
        print(f"'{plot_folder}' folder created.")

# Data cleaning function
def clean_data(df):
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'event_reference_value' in df.columns:
        df['event_reference_value'] = pd.to_numeric(df['event_reference_value'], errors='coerce')
    df = df.dropna(subset=['value', 'event_reference_value'])
    if 'measure_date' in df.columns:
        df['measure_date'] = pd.to_datetime(df['measure_date'], errors='coerce')
    return df

# Load and preprocess data
@memory.cache
def load_and_preprocess_data():
    print("Loading and cleaning data...")
    file_list = glob.glob('csv/measures_72h_before_events_device_*.csv')
    data_list = [clean_data(pd.read_csv(file, parse_dates=['measure_date'], low_memory=False)) for file in file_list]
    data = pd.concat(data_list, ignore_index=True)

    # Reduce dataset size
    sample_size = min(2000000, len(data))
    data_sampled = data.sample(n=sample_size, random_state=42)

    print("Creating target column...")
    quantiles = data_sampled['value'].quantile([0.33, 0.66])
    data_sampled['target'] = pd.cut(data_sampled['value'], bins=[-np.inf, quantiles[0.33], quantiles[0.66], np.inf], labels=[0, 1, 2])

    print("Target variable distribution:")
    print(data_sampled['target'].value_counts(normalize=True))

    # Feature engineering
    data_sampled['hour'] = data_sampled['measure_date'].dt.hour
    data_sampled['day_of_week'] = data_sampled['measure_date'].dt.dayofweek
    data_sampled['month'] = data_sampled['measure_date'].dt.month
    data_sampled['is_weekend'] = data_sampled['day_of_week'].isin([5, 6]).astype(int)

    # Lag features and rolling statistics
    data_sampled = data_sampled.sort_values('measure_date')
    data_sampled['value_lag_1'] = data_sampled.groupby('event_variable')['value'].shift(1)
    data_sampled['value_rolling_mean'] = data_sampled.groupby('event_variable')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)

    feature_columns = ['event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend', 
                       'value_lag_1', 'value_rolling_mean']

    # Prepare data for modeling
    X = data_sampled[feature_columns].dropna()
    y = data_sampled['target'].loc[X.index]

    return X, y, feature_columns

X, y, feature_columns = load_and_preprocess_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Outlier detection
print("Performing outlier detection...")
outlier_fraction = 0.01

# Isolation Forest
iso_forest = IsolationForest(contamination=outlier_fraction, random_state=42, n_jobs=4)
iso_forest_labels = iso_forest.fit_predict(X_train_scaled)

# Elliptic Envelope
ee = EllipticEnvelope(contamination=outlier_fraction, random_state=42, support_fraction=0.7)
ee_labels = ee.fit_predict(X_train_scaled)

# One-Class SVM
oc_svm = OneClassSVM(nu=outlier_fraction)
oc_svm_labels = oc_svm.fit_predict(X_train_scaled)

# Combine outlier detection results
outlier_mask = (iso_forest_labels == -1) | (ee_labels == -1) | (oc_svm_labels == -1)

# Remove outliers from training data
X_train_clean = X_train_scaled[~outlier_mask]
y_train_clean = y_train[~outlier_mask]

print(f"Removed {sum(outlier_mask)} outliers from training data.")

# Advanced resampling techniques
print("Applying advanced resampling techniques...")

from imblearn.over_sampling import SMOTE

nn = NearestNeighbors(n_jobs=4)
smote = SMOTE(random_state=42, k_neighbors=nn)
smoteenn = SMOTEENN(random_state=42, smote=smote)
X_resampled_smoteenn, y_resampled_smoteenn = smoteenn.fit_resample(X_train_clean, y_train_clean)

smotetomek = SMOTETomek(random_state=42, n_jobs=4)
X_resampled_smotetomek, y_resampled_smotetomek = smotetomek.fit_resample(X_train_clean, y_train_clean)

adasyn = ADASYN(random_state=42, n_jobs=4)
X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(X_train_clean, y_train_clean)

# XGBoost parameters
xgb_param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'max_depth': [3, 5, 7, 9, 11],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [0, 0.1, 1, 10]
}

# Train and evaluate XGBoost
best_model = None
best_score = float("inf")
best_name = ""

for name, X_resampled, y_resampled in [("SMOTEENN", X_resampled_smoteenn, y_resampled_smoteenn),
                                       ("SMOTETomek", X_resampled_smotetomek, y_resampled_smotetomek),
                                       ("ADASYN", X_resampled_adasyn, y_resampled_adasyn)]:
    print(f"\nTraining XGBoost with {name}...")
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=4, num_class=3, objective='multi:softprob')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_distributions, 
                                       n_iter=30, cv=cv, n_jobs=4, scoring='neg_log_loss', random_state=42)
    random_search.fit(X_resampled, y_resampled)

    print(f"Best parameters for XGBoost with {name}:", random_search.best_params_)
    current_best_model = random_search.best_estimator_

    y_pred = current_best_model.predict(X_test_scaled)

    log_loss_score = -f1_score(y_test, y_pred, average='weighted')
    print(f"\nLog Loss for XGBoost with {name}: {log_loss_score}")

    if log_loss_score < best_score:
        best_score = log_loss_score
        best_model = current_best_model
        best_name = name

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - XGBoost with {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'plots_xgboost_device_lists/confusion_matrix_XGBoost_{name}.png')
    plt.close()

    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': current_best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance - XGBoost with {name}')
    plt.tight_layout()
    plt.savefig(f'plots_xgboost_device_lists/feature_importance_XGBoost_{name}.png')
    plt.close()

print(f"\nBest model was trained with {best_name} resampling technique with Log Loss: {best_score}")

# Function to evaluate model over 100 iterations
def evaluate_model_over_iterations(X, y, model, iterations=100):
    f1_scores = []
    for i in range(iterations):
        X_train_iter, X_val_iter, y_train_iter, y_val_iter = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
        model.fit(X_train_iter, y_train_iter)
        y_pred_iter = model.predict(X_val_iter)
        f1 = f1_score(y_val_iter, y_pred_iter, average='weighted')
        f1_scores.append(f1)
        print(f"Iteration {i+1}/{iterations}: F1 Score = {f1:.4f}")

    return f1_scores

# Analyze multiple devices
def analyze_multiple_devices(device_ids, best_model, scaler, feature_columns):
    for device_id in device_ids:
        print(f"\nAnalyzing device {device_id}...")

        # Load device file
        file_pattern = f'csv/measures_72h_before_events_device_{device_id}_*.csv'
        device_file = glob.glob(file_pattern)

        if not device_file:
            print(f"No file found for device {device_id}.")
            continue

        df = pd.read_csv(device_file[0], parse_dates=['measure_date'], low_memory=False)
        df = clean_data(df)

        df['hour'] = df['measure_date'].dt.hour
        df['day_of_week'] = df['measure_date'].dt.dayofweek
        df['month'] = df['measure_date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        df['value_lag_1'] = df.groupby('event_variable')['value'].shift(1)
        df['value_rolling_mean'] = df.groupby('event_variable')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)

        X_device = df[feature_columns].dropna()
        X_device_scaled = scaler.transform(X_device)
        y_pred_device = best_model.predict(X_device_scaled)

        print(f"\nPrediction results for device {device_id}:")
        print(pd.DataFrame(y_pred_device, columns=['predicted_class']))

        f1_scores = evaluate_model_over_iterations(X_train_clean, y_train_clean, best_model)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 101), f1_scores, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('F1 Score (weighted)')
        plt.title(f'Model Performance Over 100 Iterations - Device {device_id}')
        plt.savefig(f'plots_xgboost_device_{device_id}/model_performance_iterations.png')
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test_scaled)), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - XGBoost - Device {device_id}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'plots_xgboost_device_{device_id}/confusion_matrix_XGBoost_device_{device_id}.png')
        plt.close()

        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(f'Feature Importance - XGBoost - Device {device_id}')
        plt.tight_layout()
        plt.savefig(f'plots_xgboost_device_{device_id}/feature_importance_XGBoost_device_{device_id}.png')
        plt.close()

# List of devices to analyze
device_ids = [114, 114, 93]

# Call the function to analyze multiple devices and generate the plots
analyze_multiple_devices(device_ids, best_model, scaler, feature_columns)
