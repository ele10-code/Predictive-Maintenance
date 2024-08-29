import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import logging
import os, json
from joblib import dump, load

from xgboost_model_federated_learning import (
    encrypt_data,
    decrypt_data,
    sanitize_input,
    validate_input,
    apply_data_quality_filters,
    create_base_model,
    train_local_model,
    federated_averaging,  
    find_optimal_threshold,
    federated_averaging,
    simulate_client_data,
    upload_to_s3,
    download_from_s3,
    lambda_handler,
    local_training_handler,
    global_aggregation_handler,
    S3_BUCKET,
    S3_CLIENT,
    parse_tree,
    average_weights,
    update_tree,
    federated_learning_round,
    run_federated_learning
)

# Configurazione del logging
logging.basicConfig(filename='federated_learning.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Creazione cartelle necessarie
for folder in ['plots_federated_learning', 'models', 'data']:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"'{folder}' folder created.")

def load_device_data(device_id: str):
    """
    Carica i dati per un dispositivo specifico.
    Se il file dei dati del dispositivo esiste, lo carica.
    Altrimenti, genera dati simulati per il dispositivo.
    """
    file_path = f'data/device_{device_id}.csv'
    if os.path.exists(file_path):
        logger.info(f"Loading data for device {device_id} from file")
        data = pd.read_csv(file_path)
        X = data.drop('target', axis=1)
        y = data['target']
    else:
        logger.info(f"Generating simulated data for device {device_id}")
        np.random.seed(int(device_id.split('_')[-1]))
        X = pd.DataFrame(np.random.rand(1000, 5), columns=['feature_' + str(i) for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 1000))
        
        # Salva i dati simulati per uso futuro
        data = pd.concat([X, y.rename('target')], axis=1)
        data.to_csv(file_path, index=False)
        logger.info(f"Simulated data for device {device_id} saved to file")

    return X, y

def create_base_model():
    return xgb.XGBClassifier(random_state=42, eval_metric='logloss')

def train_local_model(X, y, base_model):
    model = base_model.fit(X, y)
    return model

def federated_averaging(models: List[xgb.XGBClassifier]) -> xgb.XGBClassifier:
    # Inizializza il modello globale con i parametri del primo modello locale
    global_model = xgb.XGBClassifier()
    global_model.fit(np.zeros((1, models[0].n_features_in_)), [0])
    
    # Calcola la media dei parametri
    param_keys = models[0].get_params().keys()
    averaged_params = {}
    
    for key in param_keys:
        if key in ['n_estimators', 'max_depth', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
            averaged_params[key] = np.mean([model.get_params()[key] for model in models])
    
    # Imposta i parametri medi nel modello globale
    global_model.set_params(**averaged_params)
    
    # Media dei pesi degli alberi
    for i in range(global_model.n_estimators):
        tree_weights = []
        for model in models:
            tree = model.get_booster()[i]
            tree_weights.append(tree.get_dump())
        
        # Calcola la media dei pesi degli alberi
        averaged_tree = json.loads(tree_weights[0][0])
        for j in range(1, len(tree_weights)):
            tree_j = json.loads(tree_weights[j][0])
            average_tree_recursive(averaged_tree, tree_j)
        
        # Imposta l'albero medio nel modello globale
        global_model.get_booster().set_param({'updater': 'refresh'})
        global_model.get_booster().load_model(json.dumps([averaged_tree]))
    
    return global_model

def average_tree_recursive(tree1, tree2):
    """Calcola la media ricorsiva di due alberi."""
    if 'leaf' in tree1 and 'leaf' in tree2:
        tree1['leaf'] = (tree1['leaf'] + tree2['leaf']) / 2
    elif 'children' in tree1 and 'children' in tree2:
        for child1, child2 in zip(tree1['children'], tree2['children']):
            average_tree_recursive(child1, child2)
    # Se le strutture degli alberi sono diverse, manteniamo il primo albero

def federated_learning_round(devices, global_model):
    local_models = []
    for device_id in devices:
        X, y = load_device_data(device_id)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        local_model = train_local_model(X_train, y_train, global_model)
        local_models.append(local_model)

    new_global_model = federated_averaging(local_models)
    return new_global_model

def federated_learning_round(devices, global_model):
    local_models = []
    for device_id in devices:
        X, y = load_device_data(device_id)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        local_model = train_local_model(X_train, y_train, global_model)
        local_models.append(local_model)

    new_global_model = federated_averaging(local_models)
    return new_global_model

def track_device_performance(device_id: str, rounds: List[Dict], X_test: pd.DataFrame, y_test: pd.Series):
    accuracies = []
    f1_scores = []
    auc_scores = []

    for round_data in rounds:
        model = round_data['model']
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))

    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = range(1, len(accuracies) + 1)

    ax.plot(rounds, accuracies, label='Accuracy', marker='o')
    ax.plot(rounds, f1_scores, label='F1 Score', marker='s')
    ax.plot(rounds, auc_scores, label='AUC', marker='^')

    ax.set_xlabel('Round di Federated Learning')
    ax.set_ylabel('Score')
    ax.set_title(f'Performance del modello per il dispositivo {device_id}')
    ax.legend()
    ax.grid(True)

    return fig

def run_federated_learning_with_tracking(num_rounds=10, devices=None):
    if devices is None:
        devices = [f'device_{i:03d}' for i in range(5)]  # Default to 5 devices
    
    global_model = create_base_model()
    rounds = []

    for round in range(num_rounds):
        logger.info(f"Starting Federated Learning round {round + 1}/{num_rounds}")

        global_model = federated_learning_round(devices, global_model)

        rounds.append({
            'round': round + 1,
            'model': global_model,
        })

        # Evaluate global model on all devices
        total_accuracy = 0
        for device_id in devices:
            X, y = load_device_data(device_id)
            y_pred = global_model.predict(X)
            accuracy = (y_pred == y).mean()
            total_accuracy += accuracy
            logger.info(f"Round {round + 1}, Device {device_id} accuracy: {accuracy:.4f}")
        
        avg_accuracy = total_accuracy / len(devices)
        logger.info(f"Round {round + 1} completed. Average global model accuracy: {avg_accuracy:.4f}")

    logger.info("Federated Learning process completed")
    return global_model, rounds

if __name__ == "__main__":
    logger.info("Starting Federated Learning process with performance tracking")
    
    # Specifica i dispositivi che vuoi includere nel processo di Federated Learning
    devices = ['device_001', 'device_002', 'device_003']
    
    final_model, rounds = run_federated_learning_with_tracking(num_rounds=10, devices=devices)

    # Traccia le performance per ogni dispositivo
    for device_id in devices:
        X, y = load_device_data(device_id)
        X_test, y_test = X.iloc[-200:], y.iloc[-200:]  # Use last 200 samples as test set

        fig = track_device_performance(device_id, rounds, X_test, y_test)
        fig.savefig(f'plots_federated_learning/device_{device_id}_performance.png')
        logger.info(f"Performance plot for device {device_id} saved")

    dump(final_model, 'models/final_federated_model.joblib')
    logger.info("Final model saved locally")

    logger.info("Federated Learning process with performance tracking completed successfully")