import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

# Set up directory for plots
if not os.path.exists('plots_transmitter_monitoring'):
    os.makedirs('plots_transmitter_monitoring')
    print("'plots_transmitter_monitoring' folder created.")

def load_device_data(file_path):
    # Leggiamo prima le colonne del file CSV
    columns = pd.read_csv(file_path, nrows=0).columns

    # Inizializziamo il DataFrame
    data = pd.read_csv(file_path, low_memory=False)

    # Verifichiamo se 'measure_date' è presente nelle colonne
    if 'measure_date' not in columns:
        print(f"Warning: 'measure_date' column not found in {file_path}. Using default date.")
        data['measure_date'] = pd.Timestamp.now()  # Usa la data corrente come default
    else:
        data['measure_date'] = pd.to_datetime(data['measure_date'], errors='coerce')

    # Verifichiamo se 'value' è presente nelle colonne
    if 'value' not in columns:
        print(f"Warning: 'value' column not found in {file_path}. Using a default value.")
        data['value'] = 0  # Usa 0 come valore di default
    else:
        data['value'] = pd.to_numeric(data['value'], errors='coerce')

    # Feature engineering
    data['hour'] = data['measure_date'].dt.hour
    data['day_of_week'] = data['measure_date'].dt.dayofweek
    data['month'] = data['measure_date'].dt.month
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # Lag features and rolling statistics
    data = data.sort_values('measure_date')
    data['value_lag_1'] = data.groupby('measure_name')['value'].shift(1)
    data['value_rolling_mean'] = data.groupby('measure_name')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)

    # Implementing specific checks for the transmitter

    # 1. Airflow check (using txForwardPower as a proxy for airflow)
    if 'txForwardPower.0' in data['measure_name'].values:
        mask = data['measure_name'] == 'txForwardPower.0'
        data.loc[mask, 'tx_forward_power_rolling_mean'] = data.loc[mask, 'value'].rolling(window=25).mean()
        data.loc[mask, 'tx_forward_power_delta'] = data.loc[mask, 'value'].diff()

    # 2. Temperature check
    if 'statRoomTemp.0' in data['measure_name'].values and 'statRfTemp.0' in data['measure_name'].values:
        room_temp = data[data['measure_name'] == 'statRoomTemp.0'].set_index('measure_date')['value']
        rf_temp = data[data['measure_name'] == 'statRfTemp.0'].set_index('measure_date')['value']

        # Gestione dei valori duplicati
        room_temp = room_temp[~room_temp.index.duplicated(keep='first')]
        rf_temp = rf_temp[~rf_temp.index.duplicated(keep='first')]

        temp_diff = rf_temp.subtract(room_temp, fill_value=0)

        # Creazione di un dizionario per il mapping
        temp_diff_dict = temp_diff.to_dict()

        # Applicazione del mapping
        data['temp_difference'] = data['measure_date'].map(temp_diff_dict).fillna(0)

    # 3. Reflected power check
    if 'statReflected.0' in data['measure_name'].values:
        mask = data['measure_name'] == 'statReflected.0'
        data.loc[mask, 'reflected_power_issue'] = data.loc[mask, 'value'] > 5000
    else:
        data['reflected_power_issue'] = False

    # 4. Frequency and efficiency check
    if 'statFreq.0' in data['measure_name'].values:
        mask = data['measure_name'] == 'statFreq.0'
        data.loc[mask, 'freq_stability'] = data.loc[mask, 'value'].rolling(window=10).std()

    if 'rfEfficiency.0' in data['measure_name'].values:
        mask = data['measure_name'] == 'rfEfficiency.0'
        threshold = data.loc[mask, 'value'].quantile(0.1)
        data.loc[mask, 'efficiency_issue'] = data.loc[mask, 'value'] < threshold
    else:
        data['efficiency_issue'] = False

    # 5. Current check
    if 'psuCurrent.0' in data['measure_name'].values:
        mask = data['measure_name'] == 'psuCurrent.0'
        threshold = data.loc[mask, 'value'].quantile(0.1)
        data.loc[mask, 'low_current'] = data.loc[mask, 'value'] < threshold
    else:
        data['low_current'] = False

    # 6. PSU check
    psu_cols = ['psu1_out', 'psu2_out', 'psu3_out']
    if all(col in data['measure_name'].values for col in psu_cols):
        for i in range(1, 3):
            for j in range(i+1, 4):
                psu_i = data[data['measure_name'] == f'psu{i}_out'].set_index('measure_date')['value']
                psu_j = data[data['measure_name'] == f'psu{j}_out'].set_index('measure_date')['value']

                # Gestione dei valori duplicati
                psu_i = psu_i[~psu_i.index.duplicated(keep='first')]
                psu_j = psu_j[~psu_j.index.duplicated(keep='first')]

                diff = psu_i.subtract(psu_j, fill_value=0).abs()

                # Creazione di un dizionario per il mapping
                diff_dict = diff.to_dict()

                # Applicazione del mapping
                data[f'psu_diff_{i}_{j}'] = data['measure_date'].map(diff_dict).fillna(0)

        data['psu_issue'] = (data['psu_diff_1_2'] > 20) | (data['psu_diff_1_3'] > 20) | (data['psu_diff_2_3'] > 20)
    else:
        data['psu_issue'] = False

    return data

def prepare_data_for_modeling(data):
    feature_columns = ['hour', 'day_of_week', 'month', 'is_weekend', 'value_lag_1', 'value_rolling_mean']

    # Add new features if they exist
    for col in ['tx_forward_power_rolling_mean', 'tx_forward_power_delta', 'temp_difference', 'freq_stability']:
        if col in data.columns:
            feature_columns.append(col)

    # Seleziona solo le colonne che esistono nel DataFrame
    existing_columns = [col for col in feature_columns if col in data.columns]
    X = data[existing_columns].ffill().bfill()

    # Create target variable
    target_columns = ['reflected_power_issue', 'efficiency_issue', 'low_current', 'psu_issue']
    existing_target_columns = [col for col in target_columns if col in data.columns]

    if existing_target_columns:
        y = data[existing_target_columns].any(axis=1).astype(int)
    else:
        print("Warning: No target columns found. Creating a dummy target variable.")
        y = pd.Series(0, index=data.index)

    return X, y

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Creiamo un set di validazione per l'early stopping
    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    # Convertiamo i dati in DMatrix
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    # Parametri del modello
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    # Addestriamo il modello con early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # Facciamo le predizioni
    y_pred = (model.predict(dtest) > 0.5).astype(int)
    y_pred_proba = model.predict(dtest)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return model, accuracy, f1, roc_auc, y_pred_proba.max()

def main():
    # Leggi tutti i file CSV nella cartella
    file_paths = glob.glob('csv/measures_72h_before_events_device_*.csv')

    results = []
    devices_in_fault = []

    for file_path in file_paths:
        device_id = file_path.split('_')[-1].split('.')[0]
        print(f"Processing device {device_id}...")

        try:
            data = load_device_data(file_path)
            X, y = prepare_data_for_modeling(data)

            print(f"Features for device {device_id}: {X.columns.tolist()}")
            print(f"Number of positive cases: {y.sum()}")

            if y.sum() > 0 and len(X.columns) > 0:
                model, accuracy, f1, roc_auc, max_fault_prob = train_and_evaluate_model(X, y)

                results.append({
                    'device_id': device_id,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'roc_auc_score': roc_auc,
                    'max_fault_probability': max_fault_prob
                })

                # Plot feature importances
                plt.figure(figsize=(10, 6))
                xgb.plot_importance(model)
                plt.title(f'Feature Importances for Device {device_id}')
                plt.tight_layout()
                plt.savefig(f'plots_transmitter_monitoring/feature_importances_device_{device_id}.png')
                plt.close()

                # Aggiungi il dispositivo alla lista dei dispositivi in guasto se la probabilità massima supera 0.7
                if max_fault_prob > 0.7:
                    devices_in_fault.append((device_id, max_fault_prob))
            else:
                print(f"Insufficient data for device {device_id}. Skipping model training.")
        except Exception as e:
            print(f"Error processing device {device_id}: {str(e)}")
            continue

    if results:
        # Create summary DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('transmitter_monitoring_results.csv', index=False)

        # Plot performance metrics
        metrics = ['accuracy', 'f1_score', 'roc_auc_score']
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            plt.plot(results_df['device_id'], results_df[metric], marker='o', label=metric)
        plt.title('Model Performance Across Devices')
        plt.xlabel('Device ID')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots_transmitter_monitoring/performance_comparison.png')
        plt.close()

        # Plot devices predicted to be in fault
        if devices_in_fault:
            devices_in_fault.sort(key=lambda x: x[1], reverse=True)
            device_ids, fault_probs = zip(*devices_in_fault)

            plt.figure(figsize=(12, 6))
            plt.bar(device_ids, fault_probs)
            plt.title('Devices Predicted to be in Fault')
            plt.xlabel('Device ID')
            plt.ylabel('Maximum Fault Probability')
            plt.ylim(0, 1)
            for i, v in enumerate(fault_probs):
                plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
            plt.tight_layout()
            plt.savefig('plots_transmitter_monitoring/devices_in_fault.png')
            plt.close()

            print("Devices predicted to be in fault:")
            for device_id, prob in devices_in_fault:
                print(f"Device {device_id}: Maximum fault probability = {prob:.2f}")
        else:
            print("No devices were predicted to be in fault.")
    else:
        print("No models were trained due to lack of positive cases. No results to plot.")

if __name__ == "__main__":
    main()