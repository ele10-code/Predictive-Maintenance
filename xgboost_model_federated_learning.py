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
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
import hashlib
from cryptography.fernet import Fernet
import unittest
import diffprivlib.models as dp

# Configurazione del logging avanzata
logging.basicConfig(filename='federated_learning.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up joblib for caching
memory = Memory(location='.joblib_cache', verbose=0)

# Creazione cartelle necessarie
for folder in ['plots_federated_learning', 'models', 'data']:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"'{folder}' folder created.")

# Configurazione AWS
S3_BUCKET = 'your-s3-bucket-name'
S3_CLIENT = boto3.client('s3')

# Chiave per la crittografia
ENCRYPTION_KEY = Fernet.generate_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

def encrypt_data(data):
    """Crittografa i dati."""
    return cipher_suite.encrypt(json.dumps(data).encode())

def decrypt_data(encrypted_data):
    """Decrittografa i dati."""
    return json.loads(cipher_suite.decrypt(encrypted_data).decode())

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
                logger.info(f"Colonna {col} convertita in float.")
            except ValueError:
                logger.error(f"Impossibile convertire la colonna {col} in un tipo numerico.")
                raise ValueError(f"Tipo di dati non valido per la colonna {col}")

        if X[col].min() < -1e6 or X[col].max() > 1e6:
            logger.warning(f"Valori fuori range nella colonna {col}")
            X[col] = X[col].clip(-1e6, 1e6)

    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    is_inlier = iso_forest.fit_predict(X) == 1

    return X[is_inlier]

def simulate_client_data(X, y):
    """Simula la divisione dei dati per un client."""
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.8, random_state=42)
    return X_train, y_train

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

    low_variance_cols = [col for col in feature_columns if df[col].var() < 0.1]
    if low_variance_cols:
        logger.warning(f"Colonne con bassa varianza: {low_variance_cols}")

    corr_matrix = df[feature_columns].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    if high_corr_cols:
        logger.warning(f"Colonne altamente correlate: {high_corr_cols}")

    if 'hour' in df.columns:
        df = df[df['hour'].between(0, 23)]
    if 'day_of_week' in df.columns:
        df = df[df['day_of_week'].between(0, 6)]
    if 'month' in df.columns:
        df = df[df['month'].between(1, 12)]

    return df

def check_memory_usage():
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 90:
        logger.warning("Uso della memoria elevato. Considerare l'ottimizzazione.")
        gc.collect()

def incremental_train(model, X_new, y_new, num_boost_round=100):
    """Addestra incrementalmente il modello XGBoost con nuovi dati."""
    dtrain = xgb.DMatrix(X_new, label=y_new)
    params = model.get_xgb_params()
    params['process_type'] = 'update'
    params['updater'] = 'refresh,prune'  
    model.get_booster().update(dtrain, iteration=num_boost_round)
    return model

@memory.cache
def load_and_preprocess_data():
    logger.info("Loading and preprocessing data...")
    data = pd.read_csv('data/cleaned_data.csv', parse_dates=['measure_date'])

    sample_size = min(2000000, len(data))
    data_sampled = data.sample(n=sample_size, random_state=42)

    threshold = data_sampled['value'].quantile(0.75)
    data_sampled['target'] = (data_sampled['value'] > threshold).astype(int)

    logger.info(f"Target variable distribution:\n{data_sampled['target'].value_counts(normalize=True)}")

    data_sampled['hour'] = data_sampled['measure_date'].dt.hour
    data_sampled['day_of_week'] = data_sampled['measure_date'].dt.dayofweek
    data_sampled['month'] = data_sampled['measure_date'].dt.month
    data_sampled['is_weekend'] = data_sampled['day_of_week'].isin([5, 6]).astype(int)

    data_sampled = data_sampled.sort_values('measure_date')
    data_sampled['value_lag_1'] = data_sampled.groupby('event_variable')['value'].shift(1)
    data_sampled['value_rolling_mean'] = data_sampled.groupby('event_variable')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)

    feature_columns = ['event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend', 
                       'value_lag_1', 'value_rolling_mean']

    data_sampled = apply_data_quality_filters(data_sampled, feature_columns)

    X = data_sampled[feature_columns].dropna()
    y = data_sampled['target'].loc[X.index]

    X = sanitize_input(X)
    X = validate_input(X, feature_columns)
    y = y.loc[X.index]

    return X, y, feature_columns

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

def create_base_model():
    """Crea un modello XGBoost base."""
    return xgb.XGBClassifier(random_state=42, eval_metric='logloss')

def train_local_model(X, y, base_model):
    """Addestra un modello locale con i dati forniti."""
    model = base_model.fit(X, y)
    return model

def aggregate_models(models):
    """Aggrega i modelli locali in un modello globale."""
    global_model = create_base_model()
    for param in global_model.get_params():
        if param in models[0].get_params():
            param_values = [model.get_params()[param] for model in models]
            if all(isinstance(val, (int, float)) for val in param_values):
                avg_param = np.mean(param_values)
                global_model.set_params(**{param: avg_param})
    return global_model

def upload_to_s3(file_name, bucket, object_name=None):
    """Carica un file su Amazon S3."""
    if object_name is None:
        object_name = file_name
    try:
        S3_CLIENT.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logger.error(e)
        return False
    return True

def download_from_s3(bucket, object_name, file_name):
    """Scarica un file da Amazon S3."""
    try:
        S3_CLIENT.download_file(bucket, object_name, file_name)
    except ClientError as e:
        logger.error(e)
        return False
    return True

def lambda_handler(event, context):
    """Handler per AWS Lambda."""
    try:
        if event['task'] == 'local_training':
            return local_training_handler(event, context)
        elif event['task'] == 'global_aggregation':
            return global_aggregation_handler(event, context)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps('Invalid task specified')
            }
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps('Internal server error')
        }

def local_training_handler(event, context):
    """Gestisce l'addestramento locale del modello."""
    logger.info("Starting local model training")

    try:
        # Carica il modello globale da S3
        download_from_s3(S3_BUCKET, 'global_model.joblib', '/tmp/global_model.joblib')
        global_model = load('/tmp/global_model.joblib')

        # Decrittografa e carica i dati dal payload dell'evento
        encrypted_data = event['encrypted_data']
        decrypted_data = decrypt_data(encrypted_data)
        X = np.array(decrypted_data['X'])
        y = np.array(decrypted_data['y'])

        # Addestra il modello locale
        local_model = xgb.XGBClassifier()
        local_model.fit(X, y)

        # Carica il modello locale su S3
        dump(local_model, '/tmp/local_model.joblib')
        upload_to_s3('/tmp/local_model.joblib', S3_BUCKET, f'local_model_{event["client_id"]}.joblib')

        logger.info(f"Local model training completed for client {event['client_id']}")

        return {
            'statusCode': 200,
            'body': json.dumps(f"Local model trained and uploaded for client {event['client_id']}")
        }
    except Exception as e:
        logger.error(f"Error in local_training_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error during local training: {str(e)}')
        }

def global_aggregation_handler(event, context):
    """Gestisce l'aggregazione globale dei modelli."""
    logger.info("Starting global model aggregation")

    try:
        # Carica tutti i modelli locali
        local_models = []
        for client_id in event['client_ids']:
            download_from_s3(S3_BUCKET, f'local_model_{client_id}.joblib', f'/tmp/local_model_{client_id}.joblib')
            local_model = load(f'/tmp/local_model_{client_id}.joblib')
            local_models.append(local_model)

        # Aggrega i modelli
        global_model = aggregate_models(local_models)

        # Carica il nuovo modello globale su S3
        dump(global_model, '/tmp/global_model.joblib')
        upload_to_s3('/tmp/global_model.joblib', S3_BUCKET, 'global_model.joblib')

        # Versioning del modello
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_to_s3('/tmp/global_model.joblib', S3_BUCKET, f'global_model_v{model_version}.joblib')

        logger.info("Global model aggregation completed")

        return {
            'statusCode': 200,
            'body': json.dumps("Global model aggregated and uploaded successfully")
        }
    except Exception as e:
        logger.error(f"Error in global_aggregation_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps('Error during global aggregation')
        }

