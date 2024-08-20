import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging
import xgboost as xgb
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Configurazione del logging
logging.basicConfig(filename='model_performance.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Funzione per salvare i grafici
def save_plot(fig, filename):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    fig.savefig(os.path.join('plots', filename))
    plt.close(fig)

# Carica i dati puliti
print("Caricamento dei dati puliti...")
data = pd.read_csv('csv/cleaned_data.csv', 
                   parse_dates=['measure_date'])

print("Dati caricati. Informazioni sul DataFrame:")
print(data.info())

# Verifica se la colonna 'target' esiste
if 'target' not in data.columns:
    print("La colonna 'target' non esiste. Creazione della colonna target...")
    # Assumiamo che un evento sia indicato dalla presenza di un 'event_id'
    data['target'] = data['event_id'].notna().astype(int)
    print("Colonna 'target' creata.")

# Seleziona le colonne numeriche per il modello
feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = [col for col in feature_columns if col not in ['id_device', 'event_id', 'target']]

# Prepara i dati per il modello
X = data[feature_columns]
y = data['target']

print("Distribuzione della variabile target:")
print(y.value_counts(normalize=True))

# Standardizza le features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividi i dati in set di training, validazione e test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Shape of training set:", X_train.shape)
print("Shape of validation set:", X_val.shape)
print("Shape of test set:", X_test.shape)

# Il resto del codice rimane lo stesso...

# Funzione per addestrare e valutare modelli
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    # Applicazione di SMOTE solo al training set
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_val)
    score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    
    # Logging delle performance
    logging.info(f'{model_name} - AUC-ROC: {score}')
    
    return model, score

# Definizione dei modelli
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'Neural Network': Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
}

# Addestramento e valutazione dei modelli
model_scores = {}
for name, model in models.items():
    if name == 'Neural Network':
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['AUC'])
    model, score = train_and_evaluate_model(model, X_train, y_train, X_val, y_val, name)
    model_scores[name] = score

# Seleziona il miglior modello
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]

print(f"Il miglior modello è {best_model_name} con un AUC-ROC di {model_scores[best_model_name]}")

# Ottimizzazione degli iperparametri per il miglior modello
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Migliori parametri:", grid_search.best_params_)

# Valutazione finale sul set di test
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nReport di classificazione:")
print(classification_report(y_test, y_pred))

print("\nAUC-ROC:")
print(roc_auc_score(y_test, y_pred_proba))

print("\nAverage Precision Score:")
print(average_precision_score(y_test, y_pred_proba))

# Visualizzazione della matrice di confusione
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', ax=ax)
ax.set_title('Matrice di Confusione')
ax.set_ylabel('Valore Reale')
ax.set_xlabel('Valore Predetto')
save_plot(fig, 'matrice_confusione.png')

# Visualizzazione della curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(recall, precision, label='Precision-Recall curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
save_plot(fig, 'precision_recall_curve.png')

# Visualizzazione dell'importanza delle feature (se supportato dal modello)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_columns, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10,7))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), ax=ax)
    ax.set_title('Top 20 Feature Importance')
    save_plot(fig, 'feature_importance.png')

# Curve di apprendimento
train_sizes, train_scores, test_scores = learning_curve(
    estimator=best_model, X=X_scaled, y=y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(train_sizes, train_mean, label='Training score')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
ax.plot(train_sizes, test_mean, label='Cross-validation score')
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
ax.set_xlabel('Number of training examples')
ax.set_ylabel('Score')
ax.legend(loc='best')
ax.set_title('Learning Curves')
save_plot(fig, 'learning_curves.png')

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
    prediction_proba = best_model.predict_proba(new_data_scaled)[:, 1]
    
    # Log delle previsioni
    logging.info(f'Nuova previsione - Classe: {prediction}, Probabilità: {prediction_proba}')
    
    return prediction, prediction_proba

print("Esempio di utilizzo della funzione predict_maintenance:")
print("prediction, probability = predict_maintenance(new_data)")
print("Dove new_data è un DataFrame con le stesse colonne di feature utilizzate per l'addestramento.")

# Monitoraggio continuo (simulazione)
print("\nSimulazione di monitoraggio continuo:")
# Supponiamo di avere nuovi dati ogni giorno
for i in range(7):  # Simuliamo una settimana
    # In un caso reale, qui caricheresti nuovi dati
    new_data = X.sample(n=100, random_state=i)  # Simuliamo nuovi dati campionando dal dataset esistente
    predictions, probabilities = predict_maintenance(new_data)
    print(f"Giorno {i+1}: {sum(predictions)} manutenzioni previste")

print("\nNota: Implementa un sistema per caricare regolarmente nuovi dati e aggiornare il modello.")