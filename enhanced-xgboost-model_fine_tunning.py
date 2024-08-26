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
import re
from multiprocessing import cpu_count
import psutil
import json
from datetime import datetime

# Gestione condizionale di ONNX
use_onnx = True
try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    print("ONNX libraries not found. ONNX functionality will be disabled.")
    use_onnx = False

# Configurazione del logging
logging.basicConfig(filename='model_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up joblib for caching
memory = Memory(location='.joblib_cache', verbose=0)

# Create 'ml_model_visualizations' folder if it doesn't exist
if not os.path.exists('ml_model_visualizations'):
    os.makedirs('ml_model_visualizations')
    print("'ml_model_visualizations' folder created.")

def sanitize_input(data):
    """Sanitizza l'input rimuovendo caratteri potenzialmente pericolosi."""
    if isinstance(data, str):
        return re.sub(r'[^\w\s-]', '', data)
    return data

def validate_input(X, feature_columns):
    """Valida l'input e rimuove valori anomali."""
    if not all(col in X.columns for col in feature_columns):
        raise ValueError("Input mancante di alcune feature attese")
    
    for col in feature_columns:
        if X[col].dtype not in ['int64', 'float64']:
            try:
                X[col] = X[col].astype(float)
                print(f"Colonna {col} convertita in float.")
            except ValueError:
                print(f"Impossibile convertire la colonna {col} in un tipo numerico.")
                continue
        
        if X[col].min() < -1e6 or X[col].max() > 1e6:
            print(f"Attenzione: Valori fuori range nella colonna {col}")
            X[col] = X[col].clip(-1e6, 1e6)
    
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    is_inlier = iso_forest.fit_predict(X) == 1
    
    return X[is_inlier]
# applica filtri avanzati per assicurare la qualità dei dati 
def apply_data_quality_filters(df, feature_columns):
    """Applica filtri avanzati per assicurare la qualità dei dati."""
    df = df.drop_duplicates()
    
    for col in feature_columns:
        if df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    low_variance_cols = []
    for col in feature_columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].var() < 0.1:
                if len(df[col].unique()) <= 2:
                    print(f"Nota: La colonna {col} è binaria con bassa varianza. "
                          f"Distribuzione: {df[col].value_counts(normalize=True)}")
                else:
                    low_variance_cols.append(col)
    
    if low_variance_cols:
        print(f"Attenzione: le seguenti colonne non binarie hanno bassa varianza: {low_variance_cols}")
    
    corr_matrix = df[feature_columns].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    if high_corr_cols:
        print(f"Attenzione: le seguenti colonne sono altamente correlate: {high_corr_cols}")
    
    if 'hour' in df.columns:
        df = df[df['hour'].between(0, 23)]
    if 'day_of_week' in df.columns:
        df = df[df['day_of_week'].between(0, 6)]
    if 'month' in df.columns:
        df = df[df['month'].between(1, 12)]
    
    return df

# controlla l'uso della memoria
def check_memory_usage():
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 90:
        print("Attenzione: uso della memoria elevato. Considerare l'ottimizzazione.")

# Addestramento incrementale su nuovi dati
def incremental_train(model, X_new, y_new, num_boost_round=100):
    """Addestra incrementalmente il modello XGBoost con nuovi dati."""
    dtrain = xgb.DMatrix(X_new, label=y_new)
    params = model.get_xgb_params()
    params['process_type'] = 'update'
    params['updater'] = 'refresh,prune'  
    model.get_booster().update(dtrain, iteration=num_boost_round)
    return model

# Fine-tuning del modello con nuovi dati 
def fine_tune_model(model, X_new, y_new, learning_rate=0.001, num_boost_round=10):
    """
    Esegue il fine-tuning del modello XGBoost con nuovi dati.
    """
    dtrain = xgb.DMatrix(X_new, label=y_new)
    params = model.get_xgb_params()
    params['learning_rate'] = learning_rate
    model.get_booster().update(dtrain, num_boost_round)
    return model

# Addestramento continuo del modello 
def continuous_training(model, X_train, y_train, X_new, y_new, method='incremental', **kwargs):
    """
    Esegue l'addestramento continuo del modello utilizzando il metodo specificato.
    """
    if method == 'incremental':
        return incremental_train(model, X_new, y_new, **kwargs)
    elif method == 'fine_tuning':
        return fine_tune_model(model, X_new, y_new, **kwargs)
    else:
        raise ValueError("Metodo di addestramento non supportato. Usa 'incremental' o 'fine_tuning'.")


def engineer_features(df):
    df['hour'] = df['measure_date'].dt.hour.astype(int)
    df['day_of_week'] = df['measure_date'].dt.dayofweek.astype(int)
    df['month'] = df['measure_date'].dt.month.astype(int)
    
    df['day_sin'] = np.sin(df['day_of_week'] * (2. * np.pi / 7))
    df['day_cos'] = np.cos(df['day_of_week'] * (2. * np.pi / 7))
    
    df['value_lag_1'] = df.groupby('event_variable')['value'].shift(1)
    df['value_rolling_mean'] = df.groupby('event_variable')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
    
    return df
# Caricamento e pre-elaborazione dei dati
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
    
    data_sampled = engineer_features(data_sampled)
    
    feature_columns = ['event_reference_value', 'hour', 'day_sin', 'day_cos', 'month', 
                       'value_lag_1', 'value_rolling_mean']
    
    data_sampled = apply_data_quality_filters(data_sampled, feature_columns)
    
    X = data_sampled[feature_columns].dropna()
    y = data_sampled['target'].loc[X.index]
    
    X = sanitize_input(X)
    X = validate_input(X, feature_columns)
    y = y.loc[X.index]
    
    return X, y, feature_columns

# Trova la soglia ottimale per la classificazione binaria 
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

# Esporta il modello in formato ONNX
def export_model_to_onnx(model, feature_columns):
    """Esporta il modello XGBoost in formato ONNX."""
    if not use_onnx:
        print("ONNX export not available. Skipping...")
        return
    
    initial_type = [('input', FloatTensorType([None, len(feature_columns)]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    
    with open("xgboost_model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    
    print("Modello esportato in formato ONNX.")

# Carica un modello ONNX 
def load_onnx_model(model_path):
    """Carica un modello ONNX."""
    if not use_onnx:
        print("ONNX loading not available. Skipping...")
        return None
    
    session = ort.InferenceSession(model_path)
    return session

# Effettua predizioni utilizzando il modello ONNX
def predict_with_onnx(session, input_data):
    """Effettua predizioni utilizzando il modello ONNX."""
    if not use_onnx or session is None:
        print("ONNX prediction not available. Skipping...")
        return None
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: input_data.astype(np.float32)})[0]

def collect_data(data, user_id, device_id):
    """Raccoglie dati dall'applicazione."""
    timestamp = datetime.now().isoformat()
    data_point = {
        "user_id": user_id,
        "device_id": device_id,
        "timestamp": timestamp,
        "data": data
    }
    
    with open('collected_data.json', 'a') as f:
        json.dump(data_point, f)
        f.write('\n')
    
    print(f"Dati raccolti per l'utente {user_id} dal dispositivo {device_id}")

# Main execution
if __name__ == "__main__":
    print("Starting data loading and preprocessing...")
    X, y, feature_columns = load_and_preprocess_data()
    print("Data loading and preprocessing completed.")

    print("Splitting the data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Scaling the data...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
        gc.collect()

    xgb_param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'min_child_weight': [1, 3],
        'scale_pos_weight': [1, 3],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1]
    }

    n_jobs = max(1, cpu_count() // 2)

    for name, (X_resampled, y_resampled) in resampled_data.items():
        print(f"\nTraining XGBoost with {name}...")
        start_time = time.time()
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=n_jobs)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_distributions, 
                                           n_iter=20, cv=cv, n_jobs=n_jobs, scoring='f1_macro', random_state=42, verbose=2)
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
        plt.savefig(f'ml_model_visualizations/confusion_matrix_XGBoost_{name}_optimal.png')
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
        plt.savefig(f'ml_model_visualizations/feature_importance_XGBoost_{name}.png')
        plt.close()

        # SHAP values for model interpretability
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", feature_names=feature_columns)
        plt.title(f'SHAP Feature Importance - XGBoost with {name}')
        plt.tight_layout()
        plt.savefig(f'ml_model_visualizations/shap_importance_XGBoost_{name}.png')
        plt.close()

        end_time = time.time()
        print(f"Total time for {name}: {end_time - start_time:.2f} seconds")

        # Controlla l'uso della memoria
        check_memory_usage()

        gc.collect()  # Force garbage collection

    print("\nVisualizations have been saved in the 'ml_model_visualizations' folder.")
    print("\nModeling completed.")

    print("\nSimulazione dell'addestramento continuo...")
    # Simula nuovi dati (in un'applicazione reale, questi sarebbero nuovi dati raccolti)
    X_new = X_test_scaled[:1000]
    y_new = y_test[:1000]

    for name in resampled_data.keys():
        print(f"\nAddestramento continuo per il modello {name}...")
        
        # Carica il modello salvato
        loaded_model = load(f'xgboost_model_{name}.joblib')
        
        # Esegui l'addestramento incrementale
        updated_model_incremental = continuous_training(loaded_model, X_train, y_train, X_new, y_new, method='incremental')
        print(f"Addestramento incrementale completato per {name}")
        
        # Valuta il modello aggiornato con addestramento incrementale
        y_pred_incremental = updated_model_incremental.predict(X_test_scaled)
        print("\nReport di classificazione dopo l'addestramento incrementale:")
        print(classification_report(y_test, y_pred_incremental))
        
        # Esegui il fine-tuning
        updated_model_fine_tuned = continuous_training(loaded_model, X_train, y_train, X_new, y_new, method='fine_tuning', learning_rate=0.001)
        print(f"Fine-tuning completato per {name}")
        
        # Valuta il modello aggiornato con fine-tuning
        y_pred_fine_tuned = updated_model_fine_tuned.predict(X_test_scaled)
        print("\nReport di classificazione dopo il fine-tuning:")
        print(classification_report(y_test, y_pred_fine_tuned))
        
        # Salva i modelli aggiornati
        dump(updated_model_incremental, f'xgboost_model_{name}_incremental.joblib')
        dump(updated_model_fine_tuned, f'xgboost_model_{name}_fine_tuned.joblib')

    print("\nAddestramento continuo completato.")

    # Esportazione del modello in formato ONNX
    if use_onnx:
        print("\nEsportazione del modello in formato ONNX...")
        best_model = random_search.best_estimator_
        export_model_to_onnx(best_model, feature_columns)

        print("\nTest del modello ONNX...")
        onnx_session = load_onnx_model("xgboost_model.onnx")
        if onnx_session:
            onnx_predictions = predict_with_onnx(onnx_session, X_test_scaled[:5])
            print("Predizioni ONNX (prime 5):", onnx_predictions)
    else:
        print("\nONNX functionality is disabled. Skipping ONNX export and testing.")

    print("\nSimulazione della raccolta dati...")
    sample_data = X_test[:1].to_dict('records')[0]  # Prendi un campione di dati
    collect_data(sample_data, user_id="user123", device_id="device456")

    print("\nIntegrazione del modello e raccolta dati completate.")

    # Clear joblib cache
    memory.clear()

    print("Script execution completed. Check 'model_log.txt' for any logged information.")