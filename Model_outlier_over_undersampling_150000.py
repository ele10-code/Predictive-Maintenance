import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.covariance")

# Create 'plots' folder if it doesn't exist
if not os.path.exists('plots_RFM_advanced_resampling'):
    os.makedirs('plots_RFM_advanced_resampling')
    print("'plots_RFM_advanced_resampling' folder created.")

# Load cleaned data
print("Loading cleaned data...")
data = pd.read_csv('csv/cleaned_data.csv', parse_dates=['measure_date'])

# Reduce dataset size
sample_size = min(150000, len(data))
data_sampled = data.sample(n=sample_size, random_state=42)

print("Creating target column...")
threshold = data_sampled['value'].quantile(0.75)
data_sampled['target'] = (data_sampled['value'] > threshold).astype(int)

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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Outlier detection
print("Performing outlier detection...")
outlier_fraction = 0.01
n_outliers = int(outlier_fraction * len(X_train_scaled))

# Isolation Forest
iso_forest = IsolationForest(contamination=outlier_fraction, random_state=42)
iso_forest_labels = iso_forest.fit_predict(X_train_scaled)

# Elliptic Envelope with increased support_fraction
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

# SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_resampled_smoteenn, y_resampled_smoteenn = smoteenn.fit_resample(X_train_clean, y_train_clean)

# SMOTETomek
smotetomek = SMOTETomek(random_state=42)
X_resampled_smotetomek, y_resampled_smotetomek = smotetomek.fit_resample(X_train_clean, y_train_clean)

print("Class distribution after SMOTEENN:")
print(pd.Series(y_resampled_smoteenn).value_counts(normalize=True))
print("Class distribution after SMOTETomek:")
print(pd.Series(y_resampled_smotetomek).value_counts(normalize=True))

# Hyperparameter optimization with RandomizedSearchCV
param_distributions = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

# Train and evaluate models for both resampling techniques
for name, X_resampled, y_resampled in [("SMOTEENN", X_resampled_smoteenn, y_resampled_smoteenn),
                                       ("SMOTETomek", X_resampled_smotetomek, y_resampled_smotetomek)]:
    print(f"\nTraining model with {name}...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, 
                                       n_iter=30, cv=5, n_jobs=-1, scoring='roc_auc', random_state=42)
    random_search.fit(X_resampled, y_resampled)

    print(f"Best parameters for {name}:", random_search.best_params_)
    best_model = random_search.best_estimator_

    # Model evaluation
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))

    print(f"\nAUC-ROC Score for {name}:")
    print(roc_auc_score(y_test, y_pred_proba))

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'plots_RFM_advanced_resampling/confusion_matrix_{name}.png')
    plt.close()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance - {name}')
    plt.tight_layout()
    plt.savefig(f'plots_RFM_advanced_resampling/feature_importance_{name}.png')
    plt.close()

print("\nVisualizations have been saved in the 'plots_RFM_advanced_resampling' folder.")
print("\nModeling completed.")
