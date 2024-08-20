""" import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Carica i dati puliti
print("Caricamento dei dati puliti...")
data = pd.read_csv('csv/cleaned_data.csv', parse_dates=['measure_date'])

# 1. Campionamento casuale
sample_size = 150000  # Scegli una dimensione appropriata
data_sampled = data.sample(n=sample_size, random_state=42)

# 2. Aggregazione temporale
data['date'] = data['measure_date'].dt.date
data_aggregated = data.groupby(['date', 'event_variable']).agg({
    'value': ['mean', 'max', 'min'],
    'event_reference_value': 'first'
}).reset_index()

# 3. Selezione delle feature
important_features = ['event_variable', 'value', 'event_reference_value', 'measure_date']
data_reduced_features = data[important_features]

# 4. Filtraggio per periodo di tempo
start_date = '2023-01-01'
end_date = '2024-12-31'
data_filtered = data[(data['measure_date'] >= start_date) & (data['measure_date'] <= end_date)]

# Scegli una delle strategie sopra e usa il dataset risultante per il tuo modello
# Ad esempio:
reduced_data = data_sampled  # o data_aggregated, data_reduced_features, data_filtered

# Salva il dataset ridotto
reduced_data.to_csv('csv/reduced_data.csv', index=False)
print(f"Dataset ridotto salvato. Nuove dimensioni: {reduced_data.shape}")

print("Creazione della colonna target...")
threshold = data['value'].quantile(0.75)
data['target'] = (data['value'] > threshold).astype(int)

print("Distribuzione della variabile target:")
print(data['target'].value_counts(normalize=True))

# Feature engineering
data['hour'] = data['measure_date'].dt.hour
data['day_of_week'] = data['measure_date'].dt.dayofweek
data['month'] = data['measure_date'].dt.month
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# Lag features
data = data.sort_values('measure_date')
data['value_lag_1'] = data.groupby('event_variable')['value'].shift(1)
data['value_lag_7'] = data.groupby('event_variable')['value'].shift(7)

# Rolling statistics
data['value_rolling_mean'] = data.groupby('event_variable')['value'].rolling(window=24).mean().reset_index(0, drop=True)
data['value_rolling_std'] = data.groupby('event_variable')['value'].rolling(window=24).std().reset_index(0, drop=True)

feature_columns = ['event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend', 
                   'value_lag_1', 'value_lag_7', 'value_rolling_mean', 'value_rolling_std']

# Prepara i dati per il modello
X = data[feature_columns].dropna()
y = data['target'].loc[X.index]

# Divisione temporale
train_date = data['measure_date'].quantile(0.8)
X_train = X[data.loc[X.index, 'measure_date'] <= train_date]
X_test = X[data.loc[X.index, 'measure_date'] > train_date]
y_train = y[data.loc[y.index, 'measure_date'] <= train_date]
y_test = y[data.loc[y.index, 'measure_date'] > train_date]

# Standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bilanciamento delle classi
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Ottimizzazione degli iperparametri
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
grid_search.fit(X_train_resampled, y_train_resampled)

print("Migliori parametri:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Valutazione del modello
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nReport di classificazione:")
print(classification_report(y_test, y_pred))

print("\nAUC-ROC Score:")
print(roc_auc_score(y_test, y_pred_proba))

# Matrice di confusione
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione')
plt.ylabel('Valore Reale')
plt.xlabel('Valore Predetto')
plt.savefig('plots/confusion_matrix.png')
plt.close()

# Importanza delle feature
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Importanza delle Feature')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

print("\nLe visualizzazioni sono state salvate nella cartella 'plots'.")
print("\nModellazione completata.") """



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Carica i dati puliti
print("Caricamento dei dati puliti...")
data = pd.read_csv('csv/cleaned_data.csv', parse_dates=['measure_date'])

# Riduzione del dataset
sample_size = min(150000, len(data))  # Usa un campione più piccolo se necessario
data_sampled = data.sample(n=sample_size, random_state=42)

print("Creazione della colonna target...")
threshold = data_sampled['value'].quantile(0.75)
data_sampled['target'] = (data_sampled['value'] > threshold).astype(int)

print("Distribuzione della variabile target:")
print(data_sampled['target'].value_counts(normalize=True))

# Feature engineering più efficiente
data_sampled['hour'] = data_sampled['measure_date'].dt.hour
data_sampled['day_of_week'] = data_sampled['measure_date'].dt.dayofweek
data_sampled['month'] = data_sampled['measure_date'].dt.month
data_sampled['is_weekend'] = data_sampled['day_of_week'].isin([5, 6]).astype(int)

# Lag features e rolling statistics (su un subset più piccolo di dati)
data_sampled = data_sampled.sort_values('measure_date')
data_sampled['value_lag_1'] = data_sampled.groupby('event_variable')['value'].shift(1)
data_sampled['value_rolling_mean'] = data_sampled.groupby('event_variable')['value'].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)

feature_columns = ['event_reference_value', 'hour', 'day_of_week', 'month', 'is_weekend', 
                   'value_lag_1', 'value_rolling_mean']

# Prepara i dati per il modello
X = data_sampled[feature_columns].dropna()
y = data_sampled['target'].loc[X.index]

# Divisione dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bilanciamento delle classi (su un subset più piccolo se necessario)
if len(X_train) > 100000:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train_scaled, y_train, train_size=100000, stratify=y_train, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_subset, y_train_subset)
else:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Ottimizzazione degli iperparametri con RandomizedSearchCV invece di GridSearchCV
param_distributions = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, 
                                   n_iter=10, cv=3, n_jobs=-1, scoring='roc_auc', random_state=42)
random_search.fit(X_train_resampled, y_train_resampled)

print("Migliori parametri:", random_search.best_params_)
best_model = random_search.best_estimator_

# Valutazione del modello
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nReport di classificazione:")
print(classification_report(y_test, y_pred))

print("\nAUC-ROC Score:")
print(roc_auc_score(y_test, y_pred_proba))

# Matrice di confusione
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione')
plt.ylabel('Valore Reale')
plt.xlabel('Valore Predetto')
plt.savefig('plots/confusion_matrix.png')
plt.close()

# Importanza delle feature
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Importanza delle Feature')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

print("\nLe visualizzazioni sono state salvate nella cartella 'plots'.")
print("\nModellazione completata.")