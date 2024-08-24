import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer
from imblearn.combine import SMOTEENN, SMOTETomek
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Create 'plots' folder
if not os.path.exists('plots_ensemble_2000000'):
    os.makedirs('plots_ensemble_2000000')
    print("'plots_ensemble_2000000' folder created.")

# Custom scorer
def safe_roc_auc_score(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5

safe_roc_auc_scorer = make_scorer(safe_roc_auc_score, response_method='predict_proba')

# Load and prepare data
print("Loading and preparing data...")
data = pd.read_csv('csv/cleaned_data.csv', parse_dates=['measure_date'])
sample_size = min(2000000, len(data))
data_sampled = data.sample(n=sample_size, random_state=42)

threshold = data_sampled['value'].quantile(0.75)
data_sampled['target'] = (data_sampled['value'] > threshold).astype(int)

print("Target variable distribution:")
print(data_sampled['target'].value_counts(normalize=True))

# Feature engineering
data_sampled['hour'] = data_sampled['measure_date'].dt.hour
data_sampled['day_of_week'] = data_sampled['measure_date'].dt.dayofweek
data_sampled['month'] = data_sampled['measure_date'].dt.month
data_sampled['is_weekend'] = data_sampled['day_of_week'].isin([5, 6]).astype(int)
data_sampled['value_lag_1'] = data_sampled.groupby('event_variable')['value'].shift(1)
data_sampled['value_rolling_mean'] = data_sampled.groupby('event_variable')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)

feature_columns = ['event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend', 
                   'value_lag_1', 'value_rolling_mean']

X = data_sampled[feature_columns].dropna()
y = data_sampled['target'].loc[X.index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Outlier detection
print("Performing outlier detection...")
outlier_fraction = 0.01
iso_forest = IsolationForest(contamination=outlier_fraction, random_state=42)
ee = EllipticEnvelope(contamination=outlier_fraction, random_state=42, support_fraction=0.7)
oc_svm = OneClassSVM(nu=outlier_fraction)

iso_forest_labels = iso_forest.fit_predict(X_train_scaled)
ee_labels = ee.fit_predict(X_train_scaled)
oc_svm_labels = oc_svm.fit_predict(X_train_scaled)

outlier_mask = (iso_forest_labels == -1) | (ee_labels == -1) | (oc_svm_labels == -1)
X_train_clean = X_train_scaled[~outlier_mask]
y_train_clean = y_train[~outlier_mask]

print(f"Removed {sum(outlier_mask)} outliers from training data.")

# Resampling
print("Applying resampling techniques...")
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X_train_clean, y_train_clean)

print("Class distribution after resampling:")
print(pd.Series(y_resampled).value_counts(normalize=True))

# Define models and parameters
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)

xgb_param_distributions = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [1, 3, 5]
}

lgb_param_distributions = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 50],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5],
    'min_split_gain': [0, 0.1, 0.2],
    'scale_pos_weight': [1, 3, 5]
}

# Optimize individual models
print("Optimizing XGBoost...")
xgb_random_search = RandomizedSearchCV(xgb_model, xgb_param_distributions, n_iter=30, cv=5, 
                                       scoring=safe_roc_auc_scorer, n_jobs=-1, random_state=42)
xgb_random_search.fit(X_resampled, y_resampled)
best_xgb = xgb_random_search.best_estimator_

print("Optimizing LightGBM...")
lgb_random_search = RandomizedSearchCV(lgb_model, lgb_param_distributions, n_iter=30, cv=5, 
                                       scoring=safe_roc_auc_scorer, n_jobs=-1, random_state=42)
lgb_random_search.fit(X_resampled, y_resampled)
best_lgb = lgb_random_search.best_estimator_

# Create and train ensemble
print("Training ensemble model...")
ensemble = VotingClassifier(
    estimators=[('xgb', best_xgb), ('lgb', best_lgb)],
    voting='soft'
)
ensemble.fit(X_resampled, y_resampled)

# Evaluate ensemble
y_pred = ensemble.predict(X_test_scaled)
y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report for Ensemble:")
print(classification_report(y_test, y_pred))

print("\nAUC-ROC Score for Ensemble:")
print(roc_auc_score(y_test, y_pred_proba))

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Ensemble')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('plots_ensemble_2000000/confusion_matrix_ensemble.png')
plt.close()

# Feature Importance (average of both models)
xgb_importance = pd.DataFrame({'feature': feature_columns, 'importance': best_xgb.feature_importances_})
lgb_importance = pd.DataFrame({'feature': feature_columns, 'importance': best_lgb.feature_importances_})
ensemble_importance = pd.merge(xgb_importance, lgb_importance, on='feature', suffixes=('_xgb', '_lgb'))
ensemble_importance['avg_importance'] = (ensemble_importance['importance_xgb'] + ensemble_importance['importance_lgb']) / 2
ensemble_importance = ensemble_importance.sort_values('avg_importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='avg_importance', y='feature', data=ensemble_importance)
plt.title('Average Feature Importance - Ensemble')
plt.tight_layout()
plt.savefig('plots_ensemble_2000000/feature_importance_ensemble.png')
plt.close()

print("\nEnsemble modeling completed. Visualizations saved in 'plots_ensemble_2000000' folder.")
