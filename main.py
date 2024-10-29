import pandas as pd
import numpy as np
import os
import glob
import warnings
import joblib  # For saving the model

# Machine learning imports
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
import xgboost as xgb

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from load_process_data.py
from load_process_data import clean_data  # Adjust the import based on your actual file name

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.covariance")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=FutureWarning, module="imblearn")

# Create 'plots' folder if it doesn't exist
plot_folder = 'plots_xgboost_device_lists'
os.makedirs(plot_folder, exist_ok=True)

# Load and preprocess data
def load_and_preprocess_data():
    print("Loading and cleaning data...")
    # Load all CSV files from all devices and months
    file_list = glob.glob('csv/device_*/measures_device_*.csv')
    data_list = []
    for file in file_list:
        df = pd.read_csv(file, parse_dates=['measure_date'], low_memory=False)
        df = clean_data(df)
        data_list.append(df)
    data = pd.concat(data_list, ignore_index=True)
    
    # Use all data
    data_sampled = data
    
    print("Creating target column...")
    # Define target variable
    quantiles = data_sampled['value'].quantile([0.33, 0.66])
    data_sampled['target'] = pd.cut(
        data_sampled['value'],
        bins=[-np.inf, quantiles[0.33], quantiles[0.66], np.inf],
        labels=[0, 1, 2]
    )
    data_sampled['target'] = data_sampled['target'].astype(int)
    
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
    
    # Drop rows with NaN values in the features
    data_sampled = data_sampled.dropna(subset=['value_lag_1', 'value_rolling_mean'])
    
    # Define feature columns
    feature_columns = [
        'event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend',
        'value_lag_1', 'value_rolling_mean'
    ]
    
    # Prepare data for modeling
    X = data_sampled[feature_columns]
    y = data_sampled['target']
    
    return X, y, feature_columns, data_sampled['device_id'].unique()

X, y, feature_columns, all_device_ids = load_and_preprocess_data()

# Split the data into training+validation and test sets (80% train+val, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Further split the training+validation set into training and validation sets (75% train, 25% val)
# This results in 60% training, 20% validation, 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# Use RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Outlier detection on the training set
print("Performing outlier detection...")
outlier_fraction = 0.01

# Isolation Forest
iso_forest = IsolationForest(
    contamination=outlier_fraction, random_state=42, n_jobs=-1
)
iso_forest_labels = iso_forest.fit_predict(X_train_scaled)

# Elliptic Envelope
ee = EllipticEnvelope(
    contamination=outlier_fraction, random_state=42, support_fraction=0.7
)
ee_labels = ee.fit_predict(X_train_scaled)

# One-Class SVM
oc_svm = OneClassSVM(nu=outlier_fraction)
oc_svm_labels = oc_svm.fit_predict(X_train_scaled)

# Combine outlier detection results
outlier_mask = (
    (iso_forest_labels == -1) | (ee_labels == -1) | (oc_svm_labels == -1)
)

# Remove outliers from training data
X_train_clean = X_train_scaled[~outlier_mask]
y_train_clean = y_train.reset_index(drop=True)[~outlier_mask]

print(f"Removed {sum(outlier_mask)} outliers from training data.")

# Advanced resampling techniques on the training set
print("Applying advanced resampling techniques...")

smote = SMOTE(random_state=42, n_jobs=-1)
smoteenn = SMOTEENN(random_state=42, smote=smote, n_jobs=-1)
X_resampled_smoteenn, y_resampled_smoteenn = smoteenn.fit_resample(
    X_train_clean, y_train_clean
)

smotetomek = SMOTETomek(random_state=42, n_jobs=-1)
X_resampled_smotetomek, y_resampled_smotetomek = smotetomek.fit_resample(
    X_train_clean, y_train_clean
)

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

# Train and evaluate XGBoost using the validation set for hyperparameter tuning
best_model = None
best_score = float("-inf")  # We will maximize F1 score
best_name = ""

for name, X_resampled, y_resampled in [
    ("SMOTEENN", X_resampled_smoteenn, y_resampled_smoteenn),
    ("SMOTETomek", X_resampled_smotetomek, y_resampled_smotetomek)
]:
    print(f"\nTraining XGBoost with {name}...")
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1,
        num_class=3,
        objective='multi:softprob'
    )
    
    # Hyperparameter tuning using RandomizedSearchCV on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=xgb_param_distributions,
        n_iter=30,
        cv=cv,
        n_jobs=-1,
        scoring='f1_weighted',
        random_state=42
    )
    random_search.fit(X_resampled, y_resampled)
    
    print(f"Best parameters for XGBoost with {name}:", random_search.best_params_)
    current_best_model = random_search.best_estimator_

    # Evaluate on the validation set
    y_val_pred = current_best_model.predict(X_val_scaled)
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    print(f"Validation F1 Score for XGBoost with {name}: {f1}")

    if f1 > best_score:
        best_score = f1
        best_model = current_best_model
        best_name = name

    # Plot confusion matrix for validation set
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_val, y_val_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Validation) - XGBoost with {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{plot_folder}/confusion_matrix_XGBoost_{name}_validation.png')
    plt.close()

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': current_best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance - XGBoost with {name}')
    plt.tight_layout()
    plt.savefig(f'{plot_folder}/feature_importance_XGBoost_{name}.png')
    plt.close()

print(f"\nBest model was trained with {best_name} resampling technique with Validation F1 Score: {best_score}")

# Evaluate the best model on the test set
y_test_pred = best_model.predict(X_test_scaled)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
print(f"\nTest F1 Score for the best model: {test_f1}")

# Plot confusion matrix for test set
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Test) - Best XGBoost Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(f'{plot_folder}/confusion_matrix_XGBoost_best_model_test.png')
plt.close()

# Save the best model using joblib
model_filename = f'best_xgboost_model_{best_name}.joblib'
joblib.dump(best_model, model_filename)
print(f"Best model saved to {model_filename}")

# Save the scaler as well
scaler_filename = 'scaler.joblib'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")

# Function to analyze multiple devices
def analyze_multiple_devices(all_device_ids, best_model, scaler, feature_columns):
    for device_id in all_device_ids:
        print(f"\nAnalyzing device {device_id}...")

        # Load device files
        file_pattern = f'csv/device_{device_id}/measures_device_{device_id}_*.csv'
        device_files = glob.glob(file_pattern)

        if not device_files:
            print(f"No files found for device {device_id}.")
            continue

        # Load and concatenate all monthly CSV files for the device
        df_list = []
        for file in device_files:
            df = pd.read_csv(file, parse_dates=['measure_date'], low_memory=False)
            df = clean_data(df)
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)

        # Feature engineering
        df['hour'] = df['measure_date'].dt.hour
        df['day_of_week'] = df['measure_date'].dt.dayofweek
        df['month'] = df['measure_date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        df = df.sort_values('measure_date')
        df['value_lag_1'] = df.groupby('event_variable')['value'].shift(1)
        df['value_rolling_mean'] = df.groupby('event_variable')['value'].rolling(
            window=24, min_periods=1
        ).mean().reset_index(0, drop=True)

        # Drop rows with NaN values in the features
        df = df.dropna(subset=['value_lag_1', 'value_rolling_mean'])

        X_device = df[feature_columns]
        X_device_scaled = scaler.transform(X_device)
        y_pred_device = best_model.predict(X_device_scaled)

        print(f"\nPrediction results for device {device_id}:")
        print(pd.DataFrame(y_pred_device, columns=['predicted_class']))

        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Ensure the plot directory exists
        device_plot_folder = f'plots_xgboost_device_{device_id}'
        os.makedirs(device_plot_folder, exist_ok=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(f'Feature Importance - XGBoost - Device {device_id}')
        plt.tight_layout()
        plt.savefig(f'{device_plot_folder}/feature_importance_XGBoost_device_{device_id}.png')
        plt.close()

# Call the function to analyze all devices
print("\nAnalyzing all devices...")
analyze_multiple_devices(all_device_ids, best_model, scaler, feature_columns)

print("\nAnalysis complete. Check the plot folders for visualizations.")