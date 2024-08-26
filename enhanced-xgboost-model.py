import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from joblib import Memory, dump, load
import shap
import gc
import time
import logging

# Configurazione del logging
logging.basicConfig(filename='model_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up joblib for caching
memory = Memory(location='.joblib_cache', verbose=0)

# Create 'plots' folder if it doesn't exist
if not os.path.exists('plots_xgboost_enhanced'):
    os.makedirs('plots_xgboost_enhanced')
    print("'plots_xgboost_enhanced' folder created.")

def validate_input(X, feature_columns):
    """Valida l'input e rimuove valori anomali."""
    if not all(col in X.columns for col in feature_columns):
        raise ValueError("Input mancante di alcune feature attese")
    
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    is_inlier = iso_forest.fit_predict(X) == 1
    
    return X[is_inlier]

def apply_data_quality_filters(df, feature_columns):
    """Applica filtri per assicurare la qualità dei dati."""
    df = df.drop_duplicates()
    
    for col in feature_columns:
        if df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    if 'hour' in df.columns:
        df = df[df['hour'].between(0, 23)]
    if 'day_of_week' in df.columns:
        df = df[df['day_of_week'].between(0, 6)]
    if 'month' in df.columns:
        df = df[df['month'].between(1, 12)]
    
    return df

def incremental_train(model, X_new, y_new, num_boost_round=100):
    """Addestra incrementalmente il modello XGBoost con nuovi dati."""
    # Crea un DMatrix con i nuovi dati
    dtrain = xgb.DMatrix(X_new, label=y_new)
    
    # Ottieni i parametri correnti del modello
    params = model.get_xgb_params()
    
    # Aggiorna i parametri per l'addestramento incrementale
    params['process_type'] = 'update'
    params['updater'] = 'refresh,prune'  
    
    # Addestra il modello con i nuovi dati, mantenendo il booster esistente
    model.get_booster().update(dtrain, iteration=num_boost_round)
    
    return model


# Load cleaned data and preprocess
@memory.cache
def load_and_preprocess_data():
    print("Loading cleaned data...")
    data = pd.read_csv('csv/cleaned_data.csv', parse_dates=['measure_date'])
    
    sample_size = min(2000000, len(data))
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
    
    data_sampled = data_sampled.sort_values('measure_date')
    data_sampled['value_lag_1'] = data_sampled.groupby('event_variable')['value'].shift(1)
    data_sampled['value_rolling_mean'] = data_sampled.groupby('event_variable')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
    
    feature_columns = ['event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend', 
                       'value_lag_1', 'value_rolling_mean']
    
    # Applica i filtri di qualità dei dati
    data_sampled = apply_data_quality_filters(data_sampled, feature_columns)
    
    X = data_sampled[feature_columns].dropna()
    
    # Qui, allineiamo `y` a `X`
    y = data_sampled['target'].loc[X.index]
    
    # Valida l'input
    X = validate_input(X, feature_columns)
    
    # Ancora una volta, allineiamo `y` a `X` dopo la validazione
    y = y.loc[X.index]
    
    return X, y, feature_columns


print("Starting data loading and preprocessing...")
X, y, feature_columns = load_and_preprocess_data()
print("Data loading and preprocessing completed.")

# Split the data
print("Splitting the data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use RobustScaler
print("Scaling the data...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Outlier detection
print("Performing outlier detection...")
outlier_fraction = 0.01

iso_forest = IsolationForest(contamination=outlier_fraction, random_state=42, n_jobs=-1)
iso_forest_labels = iso_forest.fit_predict(X_train_scaled)

oc_svm = OneClassSVM(nu=outlier_fraction)
oc_svm_labels = oc_svm.fit_predict(X_train_scaled)

outlier_mask = (iso_forest_labels == -1) | (oc_svm_labels == -1)

X_train_clean = X_train_scaled[~outlier_mask]
y_train_clean = y_train[~outlier_mask]

print(f"Removed {sum(outlier_mask)} outliers from training data.")

# Advanced resampling techniques
print("Applying advanced resampling techniques...")


resampling_techniques = [
    ("SMOTEENN", SMOTEENN(random_state=42, n_jobs=-1)),
    ("SMOTETomek", SMOTETomek(random_state=42, n_jobs=-1)),
    ("ADASYN", ADASYN(random_state=42, n_jobs=-1, n_neighbors=NearestNeighbors(n_neighbors=5, n_jobs=-1)))
]


resampled_data = {}

for name, technique in resampling_techniques:
    print(f"Applying {name}...")
    X_resampled, y_resampled = technique.fit_resample(X_train_clean, y_train_clean)
    resampled_data[name] = (X_resampled, y_resampled)
    print(f"Class distribution after {name}:")
    print(pd.Series(y_resampled).value_counts(normalize=True))
    gc.collect()  # Force garbage collection

# Hyperparameter optimization with RandomizedSearchCV
xgb_param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    thresholds = np.arange(0.1, 1.0, 0.01)
    scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        scores.append(score)
    optimal_threshold = thresholds[np.argmax(scores)]
    return optimal_threshold

# Train and evaluate XGBoost for all resampling techniques
for name, (X_resampled, y_resampled) in resampled_data.items():
    print(f"\nTraining XGBoost with {name}...")
    start_time = time.time()
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_distributions, 
                                       n_iter=40, cv=cv, n_jobs=-1, scoring='f1_macro', random_state=42, verbose=2)
    random_search.fit(X_resampled, y_resampled)

    print(f"Best parameters for XGBoost with {name}:", random_search.best_params_)
    best_model = random_search.best_estimator_

    # Salva il modello
    dump(best_model, f'xgboost_model_{name}.joblib')

    # Model evaluation
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    print(f"\nClassification Report for XGBoost with {name}:")
    print(classification_report(y_test, y_pred))

    print(f"\nAUC-ROC Score for XGBoost with {name}:")
    print(roc_auc_score(y_test, y_pred_proba))

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, metric='recall')
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

    print(f"\nOptimal threshold for {name}:", optimal_threshold)
    print(f"\nClassification Report with optimal threshold for {name}:")
    print(classification_report(y_test, y_pred_optimal))

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred_optimal), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - XGBoost with {name} (Optimal Threshold)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'plots_xgboost_enhanced/confusion_matrix_XGBoost_{name}_optimal.png')
    plt.close()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance - XGBoost with {name}')
    plt.tight_layout()
    plt.savefig(f'plots_xgboost_enhanced/feature_importance_XGBoost_{name}.png')
    plt.close()

    # SHAP values for model interpretability
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", feature_names=feature_columns)
    plt.title(f'SHAP Feature Importance - XGBoost with {name}')
    plt.tight_layout()
    plt.savefig(f'plots_xgboost_enhanced/shap_importance_XGBoost_{name}.png')
    plt.close()

    # Esempio di addestramento incrementale
    print("\nPerforming incremental training...")
    try:
        # Simula nuovi dati
        X_new = X_test_scaled[:1000]
        y_new = y_test[:1000]
        
        # Carica il modello salvato
        loaded_model = load(f'xgboost_model_{name}.joblib')
        
        # Addestra incrementalmente
        updated_model = incremental_train(loaded_model, X_new, y_new)
        print(f"Incremental training completed for {name}")
        
        # Salva il modello aggiornato
        dump(updated_model, f'xgboost_model_{name}_updated.joblib')
    except Exception as e:
        print(f"Error during incremental training: {str(e)}")
        logging.error(f"Incremental training error for {name}: {str(e)}")
    
    end_time = time.time()
    print(f"Total time for {name}: {end_time - start_time:.2f} seconds")

    gc.collect()  # Force garbage collection

print("\nVisualizations have been saved in the 'plots_xgboost_enhanced' folder.")
print("\nModeling completed.")

# Clear joblib cache
memory.clear()

print("Script execution completed. Check 'model_log.txt' for any logged information.")
