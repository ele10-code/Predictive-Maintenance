import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Funzione per salvare i grafici come file immagine
def save_plot(fig, filename):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    fig.savefig(os.path.join('plots', filename))
    plt.close(fig)

# Carica i dati
data = pd.read_csv('csv/data_pre_processing.csv', 
                   na_values=['NOT APPLICABLE', 'NOT  APPLICABLE', 'NaN', ''], 
                   keep_default_na=True,
                   dtype={
                       'measure_name': str,
                       'value': object,
                       'measure_date': str,
                       'event_id': str,
                       'event_variable': str,
                       'event_operator': str,
                       'event_reference_value': object
                   })

# Converti le colonne numeriche
data['value'] = pd.to_numeric(data['value'], errors='coerce')
data['event_reference_value'] = pd.to_numeric(data['event_reference_value'], errors='coerce')

# Converti la colonna 'measure_date' in datetime
data['measure_date'] = pd.to_datetime(data['measure_date'], errors='coerce')

# Crea id_device se non esiste
if 'id_device' not in data.columns:
    print("ATTENZIONE: Nessuna colonna 'id_device' trovata. Creazione di un ID univoco per ogni dispositivo.")
    data['id_device'] = data.groupby(['measure_name', 'event_variable']).ngroup()

# Stampa informazioni sul DataFrame
print(data.info())
print("\nPrime righe del DataFrame:")
print(data.head())
print("\nStatistiche descrittive:")
print(data.describe())

# Ordina i dati per id_device e data
data = data.sort_values(['id_device', 'measure_date'])

# Funzione per creare features dalle misure precedenti
def create_features(group):
    group = group.sort_values('measure_date')
    group['target'] = group['event_id'].shift(-1).notna().astype(int)
    
    numeric_columns = group.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col not in ['id_device', 'event_id', 'target']:
            group[f'{col}_mean_24'] = group[col].rolling(window=24, min_periods=1).mean()
            group[f'{col}_std_24'] = group[col].rolling(window=24, min_periods=1).std()
            group[f'{col}_min_24'] = group[col].rolling(window=24, min_periods=1).min()
            group[f'{col}_max_24'] = group[col].rolling(window=24, min_periods=1).max()
    
    return group

# Applica la funzione create_features a ogni dispositivo
data = data.groupby('id_device', group_keys=False).apply(create_features).reset_index(drop=True)

# Gestione dei valori NaN
data = data.fillna(data.mean())

# Seleziona le colonne numeriche per il modello
feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = [col for col in feature_columns if col not in ['id_device', 'event_id', 'target']]

# Standardizza le features
scaler = StandardScaler()
X = scaler.fit_transform(data[feature_columns])
y = data['target']

print("\nDistribuzione della variabile target:")
print(y.value_counts(normalize=True))

# Distribuzione dei valori
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['value'], bins=50, kde=True, ax=ax)
ax.set_title('Distribuzione dei valori')
ax.set_xlabel('Valore')
ax.set_ylabel('Frequenza')
save_plot(fig, 'distribuzione_valori.png')

# Distribuzione dei valori per variabile evento
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='event_variable', y='value', data=data, ax=ax)
ax.set_title('Distribuzione dei valori per variabile evento')
ax.tick_params(axis='x', rotation=90)
save_plot(fig, 'distribuzione_valori_per_evento.png')

# Matrice di correlazione
correlation_matrix = data[feature_columns].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', ax=ax)
ax.set_title('Matrice di correlazione delle feature')
save_plot(fig, 'matrice_correlazione.png')

# Dividi i dati in set di training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of training set:", X_train.shape)
print("Shape of test set:", X_test.shape)
print("Number of positive samples in training set:", sum(y_train))
print("Number of positive samples in test set:", sum(y_test))

# Bilanciamento del dataset con SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Shape of resampled training set:", X_train_resampled.shape)
print("Number of positive samples in resampled training set:", sum(y_train_resampled))

# Inizializzazione del modello Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Definizione dei parametri per la ricerca a griglia
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Esecuzione della ricerca a griglia con validazione incrociata
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Stampare i migliori parametri trovati
print("Migliori parametri:", grid_search.best_params_)

# Utilizzare il modello migliore per fare predizioni sul set di test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Stampare il report di classificazione
print("\nReport di classificazione:")
print(classification_report(y_test, y_pred))

# Visualizzare la matrice di confusione
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', ax=ax)
ax.set_title('Matrice di Confusione')
ax.set_ylabel('Valore Reale')
ax.set_xlabel('Valore Predetto')
save_plot(fig, 'matrice_confusione.png')

# Visualizzare l'importanza delle feature
feature_importance = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_columns, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10,7))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), ax=ax)
ax.set_title('Top 20 Feature Importance')
save_plot(fig, 'feature_importance.png')

print("I grafici sono stati salvati nella cartella 'plots'.")

# Salva il modello
joblib.dump(best_model, 'predictive_maintenance_model.joblib')
print("Modello salvato come 'predictive_maintenance_model.joblib'")

# Funzione per fare previsioni su nuovi dati
def predict_maintenance(new_data):
    # Assicurati che new_data abbia le stesse colonne di feature_columns
    new_data = new_data[feature_columns]
    # Standardizza i nuovi dati
    new_data_scaled = scaler.transform(new_data)
    # Fai la previsione
    prediction = best_model.predict(new_data_scaled)
    return prediction

print("Esempio di utilizzo della funzione predict_maintenance:")
print("predict_maintenance(new_data)")
print("Dove new_data è un DataFrame con le stesse colonne di feature utilizzate per l'addestramento.")

# media del 25, misura del delta per comprendere se il flusso d'aria sta funzionando correttamente
# controllare la temperatura dell'aria in ingresso e in uscita
#refected power da tenere sotto controllo non deve crescere troppo (5000 W)
#frequenza e efficienza di macchina

# da 1000 a 5000 W 30grad
# a parità di tensione deve aumentare la corrnete

# se la corrente è troppo bassa, la macchina non sta funzionando correttamente
# PSU i out devno essere tutti e 3 gli alimentatori vicino, se c'è di differneza di 20pti, c'è un problema