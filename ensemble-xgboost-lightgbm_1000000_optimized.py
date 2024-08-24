import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest, VotingClassifier, BaggingClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
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
if not os.path.exists('plots_ensemble_1000000_optimized'):
    os.makedirs('plots_ensemble_1000000_optimized')
    print("'plots_ensemble_1000000_optimized' folder created.")

# Custom scorer
def safe_roc_auc_score(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5

safe_roc_auc_scorer = make_scorer(safe_roc_auc_score, response_method='predict_proba')

# Function to replace inf and nan values
def replace_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()

# Function to check for inf and nan values
def check_inf_nan(X, y, stage):
    if np.isinf(X).any().any() or np.isnan(X).any().any() or np.isinf(y).any() or np.isnan(y).any():
        print(f"Trovati valori infiniti o NaN in {stage}")
        X = replace_inf_nan(pd.DataFrame(X))
        y = y.loc[X.index]
    return X, y

# Load and prepare data
print("Loading and preparing data...")
data = pd.read_csv('csv/cleaned_data.csv', parse_dates=['measure_date'])
sample_size = min(1000000, len(data))
data_sampled = data.sample(n=sample_size, random_state=42)

threshold = data_sampled['value'].quantile(0.75)
data_sampled['target'] = (data_sampled['value'] > threshold).astype(int)

print("Target variable distribution:")
print(data_sampled['target'].value_counts(normalize=True))

# Enhanced Feature engineering
data_sampled['hour'] = data_sampled['measure_date'].dt.hour
data_sampled['day_of_week'] = data_sampled['measure_date'].dt.dayofweek
data_sampled['month'] = data_sampled['measure_date'].dt.month
data_sampled['is_weekend'] = data_sampled['day_of_week'].isin([5, 6]).astype(int)
data_sampled['value_lag_1'] = data_sampled.groupby('event_variable')['value'].shift(1)
data_sampled['value_rolling_mean'] = data_sampled.groupby('event_variable')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
data_sampled['value_rolling_std'] = data_sampled.groupby('event_variable')['value'].rolling(window=24, min_periods=1).std().reset_index(0, drop=True)
data_sampled['value_diff'] = data_sampled.groupby('event_variable')['value'].diff()
data_sampled['value_pct_change'] = data_sampled.groupby('event_variable')['value'].pct_change()

# More complex features
data_sampled['hour_sin'] = np.sin(2 * np.pi * data_sampled['hour'] / 24)
data_sampled['hour_cos'] = np.cos(2 * np.pi * data_sampled['hour'] / 24)
data_sampled['day_of_week_sin'] = np.sin(2 * np.pi * data_sampled['day_of_week'] / 7)
data_sampled['day_of_week_cos'] = np.cos(2 * np.pi * data_sampled['day_of_week'] / 7)
data_sampled['month_sin'] = np.sin(2 * np.pi * data_sampled['month'] / 12)
data_sampled['month_cos'] = np.cos(2 * np.pi * data_sampled['month'] / 12)

feature_columns = ['event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend', 
                   'value_lag_1', 'value_rolling_mean', 'value_rolling_std', 'value_diff', 
                   'value_pct_change', 'hour_sin', 'hour_cos', 'day_of_week_sin', 
                   'day_of_week_cos', 'month_sin', 'month_cos']

X = data_sampled[feature_columns].dropna()
y = data_sampled['target'].loc[X.index]

# Replace inf and nan values
X = replace_inf_nan(X)
y = y.loc[X.index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to preserve column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

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

# Check for inf and nan values before resampling
X_train_clean, y_train_clean = check_inf_nan(X_train_clean, y_train_clean, "prima del resampling")

# Advanced resampling
print("Applying advanced resampling techniques...")
smoteenn = SMOTEENN(random_state=42)
smote_tomek = SMOTETomek(random_state=42)
adasyn = ADASYN(random_state=42)
tomek = TomekLinks()

X_resampled_smoteenn, y_resampled_smoteenn = smoteenn.fit_resample(X_train_clean, y_train_clean)
X_resampled_smote_tomek, y_resampled_smote_tomek = smote_tomek.fit_resample(X_train_clean, y_train_clean)
X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(X_train_clean, y_train_clean)
X_resampled_tomek, y_resampled_tomek = tomek.fit_resample(X_train_clean, y_train_clean)

# Check for inf and nan values after resampling
X_resampled_smoteenn, y_resampled_smoteenn = check_inf_nan(X_resampled_smoteenn, y_resampled_smoteenn, "dopo SMOTEENN")
X_resampled_smote_tomek, y_resampled_smote_tomek = check_inf_nan(X_resampled_smote_tomek, y_resampled_smote_tomek, "dopo SMOTETomek")
X_resampled_adasyn, y_resampled_adasyn = check_inf_nan(X_resampled_adasyn, y_resampled_adasyn, "dopo ADASYN")
X_resampled_tomek, y_resampled_tomek = check_inf_nan(X_resampled_tomek, y_resampled_tomek, "dopo Tomek Links")

print("Class distribution after resampling:")
print("SMOTEENN:", pd.Series(y_resampled_smoteenn).value_counts(normalize=True))
print("SMOTETomek:", pd.Series(y_resampled_smote_tomek).value_counts(normalize=True))
print("ADASYN:", pd.Series(y_resampled_adasyn).value_counts(normalize=True))
print("Tomek Links:", pd.Series(y_resampled_tomek).value_counts(normalize=True))

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

# Feature selection for each model
print("Performing feature selection...")
xgb_selector = SelectFromModel(xgb.XGBClassifier(random_state=42), threshold='median')
lgb_selector = SelectFromModel(lgb.LGBMClassifier(random_state=42), threshold='median')

X_xgb_selected = xgb_selector.fit_transform(X_resampled_smoteenn, y_resampled_smoteenn)
X_lgb_selected = lgb_selector.fit_transform(X_resampled_smoteenn, y_resampled_smoteenn)

# Optimize individual models with selected features
print("Optimizing XGBoost...")
xgb_random_search = RandomizedSearchCV(xgb_model, xgb_param_distributions, n_iter=30, cv=StratifiedKFold(n_splits=5), 
                                       scoring=safe_roc_auc_scorer, n_jobs=-1, random_state=42)
xgb_random_search.fit(X_xgb_selected, y_resampled_smoteenn)
best_xgb = xgb_random_search.best_estimator_

print("Optimizing LightGBM...")
lgb_random_search = RandomizedSearchCV(lgb_model, lgb_param_distributions, n_iter=30, cv=StratifiedKFold(n_splits=5), 
                                       scoring=safe_roc_auc_scorer, n_jobs=-1, random_state=42)
lgb_random_search.fit(X_lgb_selected, y_resampled_smoteenn)
best_lgb = lgb_random_search.best_estimator_

# Bagging for each model
print("Applying Bagging...")
bagged_xgb = BaggingClassifier(estimator=best_xgb, n_estimators=10, random_state=42)
bagged_lgb = BaggingClassifier(estimator=best_lgb, n_estimators=10, random_state=42)

bagged_xgb.fit(X_xgb_selected, y_resampled_smoteenn)
bagged_lgb.fit(X_lgb_selected, y_resampled_smoteenn)

# Calibrate probabilities
print("Calibrating probabilities...")
calibrated_xgb = CalibratedClassifierCV(bagged_xgb, method='isotonic', cv=5)
calibrated_lgb = CalibratedClassifierCV(bagged_lgb, method='isotonic', cv=5)

calibrated_xgb.fit(X_xgb_selected, y_resampled_smoteenn)
calibrated_lgb.fit(X_lgb_selected, y_resampled_smoteenn)

# Create and optimize ensemble
print("Creating and optimizing ensemble...")
ensemble = VotingClassifier(
    estimators=[('xgb', calibrated_xgb), ('lgb', calibrated_lgb)],
    voting='soft'
)

# Optimize ensemble weights
param_grid = {'weights': [[1, 1], [1, 2], [2, 1], [1, 3], [3, 1], [2, 2]]}
ensemble_grid_search = GridSearchCV(ensemble, param_grid, cv=StratifiedKFold(n_splits=5), 
                                    scoring=safe_roc_auc_scorer, n_jobs=-1)
ensemble_grid_search.fit(X_resampled_smoteenn, y_resampled_smoteenn)
best_ensemble = ensemble_grid_search.best_estimator_

# Evaluate ensemble
X_test_xgb = xgb_selector.transform(X_test_scaled)
X_test_lgb = lgb_selector.transform(X_test_scaled)
y_pred = best_ensemble.predict(X_test_scaled)
y_pred_proba = best_ensemble.predict_proba(X_test_scaled)[:, 1]

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
plt.savefig('plots_ensemble_1000000_optimized/confusion_matrix_ensemble.png')
plt.close()

# Feature Importance (average of both models)
xgb_feature_names = xgb_selector.get_feature_names_out(feature_columns)
lgb_feature_names = lgb_selector.get_feature_names_out(feature_columns)

xgb_importance = pd.DataFrame({'feature': xgb_feature_names, 'importance': best_xgb.feature_importances_})
lgb_importance = pd.DataFrame({'feature': lgb_feature_names, 'importance': best_lgb.feature_importances_})

# Merge le importanze delle feature
ensemble_importance = pd.merge(xgb_importance, lgb_importance, on='feature', suffixes=('_xgb', '_lgb'), how='outer')
ensemble_importance = ensemble_importance.fillna(0)  # Riempi i NaN con 0 per le feature non presenti in entrambi i modelli
ensemble_importance['avg_importance'] = (ensemble_importance['importance_xgb'] + ensemble_importance['importance_lgb']) / 2
ensemble_importance = ensemble_importance.sort_values('avg_importance', ascending=False)

# Visualizzazione delle importanze delle feature
plt.figure(figsize=(10, 6))
sns.barplot(x='avg_importance', y='feature', data=ensemble_importance.head(20))  # Mostra solo le top 20 feature
plt.title('Average Feature Importance - Ensemble (Top 20)')
plt.tight_layout()
plt.savefig('plots_ensemble_1000000_optimized/feature_importance_ensemble.png')
plt.close()

print("\nEnsemble modeling completed. Visualizations saved in 'plots_ensemble_1000000_optimized' folder.")