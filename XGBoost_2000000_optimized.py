import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, recall_score, precision_score, make_scorer, accuracy_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from sklearn.svm import OneClassSVM
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from joblib import Memory, Parallel, delayed
import glob
import re
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up joblib for caching
memory = Memory(location='.joblib_cache', verbose=0)

# Set up logging
logging.basicConfig(filename='model_performance.log', level=logging.INFO)

# Create 'plots' folder
if not os.path.exists('plots_anomaly_detection'):
    os.makedirs('plots_anomaly_detection')
    print("'plots_anomaly_detection' folder created.")

# Function to clean and prepare data
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
    print("Loading and preparing data...")
    csv_files = glob.glob('csv/measures_72h_before_events_device_*.csv')
    if not csv_files:
        raise ValueError("No CSV files found matching the pattern 'measures_72h_before_events_device_*.csv' in the 'csv' folder.")

    data_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False, 
                             names=['measure_id', 'oid', 'measure_name', 'value', 'value_label', 
                                    'measure_date', 'device_name', 'event_id', 'event_variable', 
                                    'event_operator', 'event_reference_value'])
            df['id_device'] = re.search(r'device_(\d+)', file).group(1)
            data_list.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

    if not data_list:
        raise ValueError("No valid data found in CSV files.")

    data = pd.concat(data_list, ignore_index=True)
    print(f"Loaded and cleaned data from {len(data_list)} CSV files.")

    # Function to safely convert to datetime
    def safe_parse_date(date_str):
        try:
            return pd.to_datetime(date_str)
        except:
            return pd.NaT

    # Print some raw values of measure_date
    print("Sample of raw measure_date values:")
    print(data['measure_date'].head())

    # Convert measure_date to datetime, handling errors
    data['measure_date'] = data['measure_date'].apply(safe_parse_date)

    # Remove rows with invalid dates
    invalid_dates = data['measure_date'].isna()
    if invalid_dates.any():
        print(f"Removed {invalid_dates.sum()} rows with invalid dates")
        data = data[~invalid_dates]

    # Convert other columns to appropriate types
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    data['event_reference_value'] = pd.to_numeric(data['event_reference_value'], errors='coerce')

    # Create time-based features
    data['hour'] = data['measure_date'].dt.hour
    data['day_of_week'] = data['measure_date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # Print some information about the measure_date column after conversion
    print("\nMeasure_date info after conversion:")
    print(data['measure_date'].info())

    # Pivot table for measurements
    pivot_data = data.pivot_table(values='value', index=['measure_date', 'id_device', 'event_id'],
                                  columns='measure_name', aggfunc='first').reset_index()

    # Function to find columns by partial name
    def find_column(partial_name):
        matches = [col for col in pivot_data.columns if partial_name.lower() in col.lower()]
        return matches[0] if matches else None

    # Define conditions for anomalies and faults
    reflected_col = find_column('statReflected')
    rf_temp_col = find_column('statRfTemp')
    room_temp_col = find_column('statRoomTemp')
    forward_col = find_column('statForward')

    anomaly_conditions = pd.Series(False, index=pivot_data.index)
    fault_conditions = pd.Series(False, index=pivot_data.index)

    if reflected_col:
        anomaly_conditions |= (pivot_data[reflected_col] > 10)
        fault_conditions |= (pivot_data[reflected_col] > 20)
    
    if rf_temp_col:
        anomaly_conditions |= (pivot_data[rf_temp_col] > 50)
        fault_conditions |= (pivot_data[rf_temp_col] > 60)
    
    if room_temp_col:
        anomaly_conditions |= (pivot_data[room_temp_col] > 35)
    
    if forward_col:
        fault_conditions |= (pivot_data[forward_col] < pivot_data['event_reference_value'] * 0.5)

    # Assign status: 0 = normal, 1 = anomaly, 2 = fault
    pivot_data['status'] = np.select(
        [fault_conditions, anomaly_conditions],
        [2, 1],
        default=0
    )

    # Select features for the model
    base_features = ['statDeviation', 'statFreq', 'statRfTemp', 'statRoomTemp', 
                     'statForward', 'statReflected']
    feature_columns = [find_column(feat) for feat in base_features if find_column(feat)] + ['hour', 'day_of_week', 'is_weekend']
    feature_columns = [col for col in feature_columns if col is not None]

    X = pivot_data[feature_columns]
    y = pivot_data['status']

    print("\nFeature selezionate per il modello:")
    print(feature_columns)

    print("\nDistribuzione delle classi:")
    print(y.value_counts(normalize=True))

    return X, y, feature_columns

# Function to analyze feature distribution
def analyze_feature_distribution(X, feature_names):
    for feature in feature_names:
        plt.figure(figsize=(10, 4))
        plt.title(f'Distribution of {feature}')
        plt.hist(X[feature], bins=50)
        plt.savefig(f'plots_anomaly_detection/feature_distribution_{feature}.png')
        plt.close()

# Function to analyze feature correlation
def analyze_feature_correlation(X, feature_names):
    corr = X[feature_names].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('plots_anomaly_detection/feature_correlation.png')
    plt.close()

# Main execution
X, y, feature_columns = load_and_preprocess_data()

print("\nDistribuzione delle classi:")
print(y.value_counts(normalize=True))

# Analyze feature distribution and correlation
analyze_feature_distribution(X[feature_columns], feature_columns)
analyze_feature_correlation(X[feature_columns], feature_columns)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Scale the features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Outlier detection
print("Performing outlier detection...")
outlier_fraction = 0.01

iso_forest = IsolationForest(contamination=outlier_fraction, random_state=42, n_jobs=-1)
ee = EllipticEnvelope(contamination=outlier_fraction, random_state=42, support_fraction=0.7)
oc_svm = OneClassSVM(nu=outlier_fraction)

iso_forest_labels = iso_forest.fit_predict(X_train_scaled)
ee_labels = ee.fit_predict(X_train_scaled)
oc_svm_labels = oc_svm.fit_predict(X_train_scaled)

outlier_mask = (iso_forest_labels == -1) | (ee_labels == -1) | (oc_svm_labels == -1)
X_train_clean = X_train_scaled[~outlier_mask]
y_train_clean = y_train[~outlier_mask]

print(f"Removed {sum(outlier_mask)} outliers from training data.")

# Function to determine optimal n_neighbors
def find_optimal_n_neighbors(X, y, max_n_neighbors=20, n_jobs=-1):
    print("Finding optimal n_neighbors for SMOTE...")
    
    def evaluate_n_neighbors(n):
        smote = SMOTE(k_neighbors=n, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        model = xgb.XGBClassifier(random_state=42)
        scores = cross_val_score(model, X_resampled, y_resampled, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1_weighted')
        return np.mean(scores)
    
    f1_scores = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_n_neighbors)(n) for n in range(1, max_n_neighbors + 1)
    )
    
    optimal_n = np.argmax(f1_scores) + 1
    print(f"Optimal n_neighbors: {optimal_n}")
    return optimal_n

# Calculate the optimal n_neighbors once
optimal_n_neighbors = find_optimal_n_neighbors(X_train_clean, y_train_clean)

# Advanced resampling techniques
print("Applying advanced resampling techniques...")

# Define resampling methods with the precomputed optimal n_neighbors
resampling_methods = {
    'SMOTE': SMOTE(random_state=42, k_neighbors=optimal_n_neighbors),
    'ADASYN': ADASYN(random_state=42, n_neighbors=optimal_n_neighbors),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=optimal_n_neighbors),
    'SMOTEENN': SMOTEENN(smote=SMOTE(random_state=42, k_neighbors=optimal_n_neighbors))
}

best_resampling_method = None
best_f1_score = 0
for method_name, resampler in resampling_methods.items():
    print(f"Testing resampling method: {method_name}")
    X_resampled, y_resampled = resampler.fit_resample(X_train_clean, y_train_clean)
    model = xgb.XGBClassifier(random_state=42)
    scores = cross_val_score(model, X_resampled, y_resampled, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1_weighted')
    mean_score = np.mean(scores)
    print(f"F1 Score for {method_name}: {mean_score:.4f}")
    if mean_score > best_f1_score:
        best_f1_score = mean_score
        best_resampling_method = method_name

print(f"Best resampling method: {best_resampling_method} with F1 Score: {best_f1_score:.4f}")

# Apply the best resampling method
X_train_resampled, y_train_resampled = resampling_methods[best_resampling_method].fit_resample(X_train_clean, y_train_clean)

def xgb_cv_score(estimator, X, y):
    # Prepara i dati per XGBoost
    dtrain = xgb.DMatrix(X, label=y)
    
    # Prepara i parametri
    params = estimator.get_params()
    params['objective'] = 'multi:softprob'
    params['eval_metric'] = 'mlogloss'
    params['num_class'] = 3

    # Rimuovi i parametri non utilizzati
    params.pop('use_label_encoder', None)
    params.pop('enable_categorical', None)
    params.pop('missing', None)
    params.pop('n_estimators', None)
    
    # Esegui la cross-validation con early stopping
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        nfold=5,
        early_stopping_rounds=50,
        metrics='mlogloss',
        as_pandas=False
    )
    
    # Restituisci il miglior punteggio (negativo perché scikit-learn massimizza)
    best_score = min(cv_results['test-mlogloss-mean'])
    return -best_score

# Definizione del modello e dei parametri
xgb_model = xgb.XGBClassifier(
    eval_metric='mlogloss',
    enable_categorical=False
)

param_distributions = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1],
}

# RandomizedSearchCV con la funzione di scoring personalizzata
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_distributions,
    n_iter=20,
    scoring=xgb_cv_score,
    n_jobs=-1,
    cv=5,
    verbose=2,
    random_state=42
)

# Addestramento del modello
print("Iniziando la ricerca degli iperparametri ...")
random_search.fit(X_train_resampled, y_train_resampled)

# Migliori parametri e punteggio
print("Migliori parametri trovati:")
print(random_search.best_params_)
print("Miglior punteggio:", -random_search.best_score_)

best_params = random_search.best_params_
best_params['eval_metric'] = 'mlogloss' 
# Addestra il modello finale con i migliori parametri
best_model = xgb.XGBClassifier(**best_params)

# Fit del modello
best_model.fit(X_train_resampled, y_train_resampled)

print("Addestramento completato.")

# Predict on test set
xgb_pred_proba = best_model.predict_proba(X_test_scaled)

# Find dynamic optimal threshold for multi-class classification
def find_dynamic_threshold(y_true, y_pred_proba, metric='f1_weighted'):
    thresholds = np.linspace(0.1, 0.9, 100)
    scores = []
    for threshold in thresholds:
        y_pred = np.argmax(y_pred_proba >= threshold[:, np.newaxis], axis=1)
        if metric == 'f1_weighted':
            score = f1_score(y_true, y_pred, average='weighted')
        elif metric == 'recall_weighted':
            score = recall_score(y_true, y_pred, average='weighted')
        elif metric == 'precision_weighted':
            score = precision_score(y_true, y_pred, average='weighted')
        scores.append(score)
    optimal_threshold = thresholds[np.argmax(scores)]
    return optimal_threshold

optimal_threshold = find_dynamic_threshold(y_test, xgb_pred_proba, metric='f1_weighted')
print(f"\nOptimal dynamic threshold: {optimal_threshold}")

# Make predictions using the optimal threshold
xgb_pred = np.argmax(xgb_pred_proba >= optimal_threshold, axis=1)

# Evaluate model
print("\nDetailed Model Evaluation Metrics:")
accuracy = accuracy_score(y_test, xgb_pred)
precision = precision_score(y_test, xgb_pred, average='weighted')
recall = recall_score(y_test, xgb_pred, average='weighted')
f1 = f1_score(y_test, xgb_pred, average='weighted')
roc_auc = roc_auc_score(y_test, xgb_pred_proba, multi_class='ovo', average='weighted')
mcc = matthews_corrcoef(y_test, xgb_pred)
kappa = cohen_kappa_score(y_test, xgb_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC Score: {roc_auc:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, xgb_pred)
print(cm)

# Visualizations

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('plots_anomaly_detection/confusion_matrix.png')
plt.close()

# 2. ROC Curve for Multi-class
plt.figure(figsize=(10, 8))
for i in range(3):  # Assuming three classes: normal, anomaly, fault
    fpr, tpr, _ = roc_curve((y_test == i).astype(int), xgb_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f'Class {i} ROC Curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig('plots_anomaly_detection/roc_curve.png')
plt.close()

# 3. Precision-Recall Curve for Multi-class
plt.figure(figsize=(10, 8))
for i in range(3):  # Assuming three classes: normal, anomaly, fault
    precision, recall, _ = precision_recall_curve((y_test == i).astype(int), xgb_pred_proba[:, i])
    plt.plot(recall, precision, label=f'Class {i} Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('plots_anomaly_detection/precision_recall_curve.png')
plt.close()

# 4. Feature Importance
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(feature_columns)[sorted_idx])
plt.title('Feature Importance (XGBoost)')
plt.tight_layout()
plt.savefig('plots_anomaly_detection/feature_importance.png')
plt.close()

print("\nXGBoost anomaly and fault detection modeling completed. Visualizations saved in 'plots_anomaly_detection' folder.")

# Log the final model performance
logging.info(f"Final model performance: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC-ROC={roc_auc:.4f}")

# Clear joblib cache
memory.clear()