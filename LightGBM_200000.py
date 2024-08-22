import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer
from imblearn.combine import SMOTEENN, SMOTETomek
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.covariance")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# Create 'plots' folder if it doesn't exist
if not os.path.exists('plots_lightgbm_200000'):
    os.makedirs('plots_lightgbm_200000')
    print("'plots_lightgbm_200000' folder created.")

# Define a custom scorer that handles the case when only one class is present
def safe_roc_auc_score(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5  # Return 0.5 (random guessing) when ROC AUC is undefined

# safe_roc_auc_scorer = make_scorer(safe_roc_auc_score, needs_proba=True)
# Updated scorer definition
safe_roc_auc_scorer = make_scorer(safe_roc_auc_score, response_method='predict_proba')

# Load cleaned data
print("Loading cleaned data...")
data = pd.read_csv('csv/cleaned_data.csv', parse_dates=['measure_date'])

# Reduce dataset size
sample_size = min(200000, len(data))
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

# Analyze feature distributions
print("\nAnalyzing feature distributions...")
for feature in feature_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(X[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.savefig(f'plots_lightgbm_200000/distribution_{feature}.png')
    plt.close()

# Analyze feature correlations
print("\nAnalyzing feature correlations...")
correlation_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Feature Correlation Matrix')
plt.savefig('plots_lightgbm_200000/feature_correlation_matrix.png')
plt.close()

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

# Train and evaluate LightGBM for both resampling techniques
for name, X_resampled, y_resampled in [("SMOTEENN", X_resampled_smoteenn, y_resampled_smoteenn),
                                       ("SMOTETomek", X_resampled_smotetomek, y_resampled_smotetomek)]:
    print(f"\nTraining LightGBM with {name}...")
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    
    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    

    # In the RandomizedSearchCV part:
    random_search = RandomizedSearchCV(estimator=lgb_model, 
                                   param_distributions=lgb_param_distributions, 
                                   n_iter=30, 
                                   cv=cv, 
                                   n_jobs=-1, 
                                   scoring=safe_roc_auc_scorer,  # Use the updated custom scorer
                                   random_state=42, 
                                   verbose=0)
    
    random_search.fit(X_resampled, y_resampled)

    print(f"Best parameters for LightGBM with {name}:", random_search.best_params_)
    best_model = random_search.best_estimator_

    # Model evaluation
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    print(f"\nClassification Report for LightGBM with {name}:")
    print(classification_report(y_test, y_pred))

    print(f"\nSafe AUC-ROC Score for LightGBM with {name}:")
    print(safe_roc_auc_score(y_test, y_pred_proba))

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - LightGBM with {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'plots_lightgbm_200000/confusion_matrix_LightGBM_{name}.png')
    plt.close()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance - LightGBM with {name}')
    plt.tight_layout()
    plt.savefig(f'plots_lightgbm_200000/feature_importance_LightGBM_{name}.png')
    plt.close()

    # Learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_resampled, y_resampled, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), scoring=safe_roc_auc_scorer
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Number of training examples')
    plt.ylabel('ROC AUC Score')
    plt.title(f'Learning Curves - LightGBM with {name}')
    plt.legend(loc='best')
    plt.savefig(f'plots_lightgbm_200000/learning_curves_LightGBM_{name}.png')
    plt.close()

print("\nVisualizations have been saved in the 'plots_lightgbm_200000' folder.")
print("\nModeling completed.")