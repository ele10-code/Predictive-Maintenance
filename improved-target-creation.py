import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Carica i dati puliti
print("Caricamento dei dati puliti...")
data = pd.read_csv('csv/cleaned_data.csv', parse_dates=['measure_date'])

print("Dati caricati. Informazioni sul DataFrame:")
print(data.info())

# Creazione di una variabile target più bilanciata
print("Creazione della colonna target...")
if 'value' in data.columns:
    # Usa il 75° percentile come soglia invece della mediana
    threshold = data['value'].quantile(0.75)
    data['target'] = (data['value'] > threshold).astype(int)
else:
    # Se 'value' non è disponibile, usa 'event_id'
    data['target'] = data['event_id'].notna().astype(int)

print("Distribuzione della variabile target:")
print(data['target'].value_counts(normalize=True))

# Verifica se abbiamo almeno due classi
if len(data['target'].unique()) < 2:
    print("ERRORE: La variabile target ha meno di due classi. Non è possibile procedere con la classificazione.")
    exit()

# Seleziona le colonne numeriche per il modello
feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = [col for col in feature_columns if col not in ['id_device', 'event_id', 'target']]

# Prepara i dati per il modello
X = data[feature_columns]
y = data['target']

# Standardizza le features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividi i dati in set di training e test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Shape of training set:", X_train.shape)
print("Shape of test set:", X_test.shape)

# Visualizzazione della distribuzione dei valori per ogni feature
for feature in feature_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=feature, hue='target', kde=True, palette='Set2')
    plt.title(f'Distribuzione di {feature} per classe target')
    plt.savefig(f'plots/distribution_{feature}.png')
    plt.close()

print("I grafici di distribuzione sono stati salvati nella cartella 'plots'.")

# Matrice di correlazione
correlation_matrix = data[feature_columns + ['target']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Matrice di Correlazione')
plt.savefig('plots/correlation_matrix.png')
plt.close()

print("La matrice di correlazione è stata salvata come 'correlation_matrix.png'.")

print("\nAnalisi esplorativa completata. Ora puoi procedere con la modellazione.")