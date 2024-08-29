import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.combine import SMOTETomek
import xgboost as xgb
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Create 'plots' folder if it doesn't exist
if not os.path.exists('plots_transmitter_model'):
    os.makedirs('plots_transmitter_model')
    print("'plots_transmitter_model' folder created.")

def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")
    data = pd.read_csv(file_path, parse_dates=['measure_date'])
    
    # Feature engineering based on domain knowledge
    data['delta_25'] = data.groupby('event_variable')['value'].transform(lambda x: x - x.rolling(window=25, min_periods=1).mean())
    data['reflected_power_high'] = ((data['event_variable'] == 'reflected_power') & (data['value'] > 5000)).astype(int)
    data['current_too_low'] = ((data['event_variable'] == 'current') & (data['value'] < data['value'].quantile(0.1))).astype(int)
    
    # Temperature difference (assuming we have inlet and outlet temperature)
    if 'inlet_temp' in data.columns and 'outlet_temp' in data.columns:
        data['temp_difference'] = data['outlet_temp'] - data['inlet_temp']
    
    # PSU difference check (assuming PSU values are in separate columns)
    psu_columns = [col for col in data.columns if col.startswith('PSU')]
    if len(psu_columns) >= 3:
        data['psu_difference'] = data[psu_columns].max(axis=1) - data[psu_columns].min(axis=1)
        data['psu_problem'] = (data['psu_difference'] > 20).astype(int)
    
    # Define anomaly based on multiple conditions
    data['anomaly'] = (
        (data['delta_25'].abs() > data['delta_25'].abs().quantile(0.95)) |
        (data['reflected_power_high'] == 1) |
        (data['current_too_low'] == 1) |
        (data['psu_problem'] == 1 if 'psu_problem' in data.columns else False)
    ).astype(int)
    
    # Additional features
    data['hour'] = data['measure_date'].dt.hour
    data['day_of_week'] = data['measure_date'].dt.dayofweek
    data['month'] = data['measure_date'].dt.month
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['value_rolling_mean'] = data.groupby('event_variable')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
    
    feature_columns = ['event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend', 
                       'value_rolling_mean', 'delta_25', 'reflected_power_high', 'current_too_low']
    
    if 'temp_difference' in data.columns:
        feature_columns.append('temp_difference')
    if 'psu_problem' in data.columns:
        feature_columns.append('psu_problem')
    
    X = data[feature_columns]
    y = data['anomaly']
    
    return X, y, feature_columns

def train_and_evaluate_model(X, y, feature_columns):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTETomek for balanced resampling
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)
    
    # XGBoost model and hyperparameter search
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

    # Hyperparameter optimization with RandomizedSearchCV
    param_distributions = {
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
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_distributions, 
                                       n_iter=30, cv=cv, scoring='f1', random_state=42, n_jobs=-1)
    random_search.fit(X_resampled, y_resampled)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nROC-AUC Score:")
    print(roc_auc_score(y_test, y_pred_proba))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - XGBoost Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots_transmitter_model/confusion_matrix.png')
    plt.close()
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance - XGBoost Model')
    plt.tight_layout()
    plt.savefig('plots_transmitter_model/feature_importance.png')
    plt.close()
    
    return best_model, feature_importance

def main():
    file_path = 'csv/cleaned_data.csv'  # Replace with your actual file path
    X, y, feature_columns = load_and_preprocess_data(file_path)
    best_model, feature_importance = train_and_evaluate_model(X, y, feature_columns)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    print("\nModel training and evaluation completed.")
    print("Visualizations have been saved in the 'plots_transmitter_model' folder.")

if __name__ == "__main__":
    main()
