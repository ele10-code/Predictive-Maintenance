import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Funzione per assicurarsi che una directory esista
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Funzione per creare la colonna target
def create_target(data, threshold_percentile=75):
    threshold = data['value'].quantile(threshold_percentile / 100)
    data['target'] = (data['value'] > threshold).astype(int)
    return data

# Analisi delle caratteristiche dei guasti
def analyze_fault_characteristics(data):
    fault_data = data[data['target'] == 1]
    non_fault_data = data[data['target'] == 0]

    print(f"Numero totale di osservazioni: {len(data)}")
    print(f"Numero di guasti rilevati: {len(fault_data)} ({len(fault_data)/len(data)*100:.2f}%)")

    # Seleziona solo le colonne numeriche per l'analisi statistica
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    fault_stats = fault_data[numeric_columns].describe()
    non_fault_stats = non_fault_data[numeric_columns].describe()

    # Confronto tra le medie delle feature durante i guasti e non guasti
    mean_comparison = pd.concat([fault_stats.loc['mean'], non_fault_stats.loc['mean']], axis=1)
    mean_comparison.columns = ['Fault Mean', 'Non-Fault Mean']
    mean_comparison['Difference'] = mean_comparison['Fault Mean'] - mean_comparison['Non-Fault Mean']
    
    # Calcola la differenza relativa evitando la divisione per zero
    mean_comparison['Relative Difference'] = np.where(
        mean_comparison['Non-Fault Mean'] != 0,
        mean_comparison['Difference'] / mean_comparison['Non-Fault Mean'],
        np.inf
    )
    
    print("\nConfronto delle medie delle feature durante i guasti e non guasti:")
    print(mean_comparison.sort_values('Relative Difference', ascending=False).head(10))

    # Visualizzazione della distribuzione delle feature più rilevanti
    top_features = mean_comparison.sort_values('Relative Difference', ascending=False).head(5).index

    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle('Distribuzione delle feature più rilevanti durante i guasti e non guasti')

    for i, feature in enumerate(top_features):
        ax = axes[i // 2, i % 2]
        sns.histplot(data=data, x=feature, hue='target', kde=True, ax=ax)
        ax.set_title(feature)

    plt.tight_layout()
    plt.savefig('plots_fault/fault_feature_distributions.png')
    plt.close()

    # Analisi delle colonne non numeriche
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_columns) > 0:
        print("\nAnalisi delle colonne non numeriche:")
        for col in non_numeric_columns:
            print(f"\nColonna: {col}")
            print(data[col].value_counts(normalize=True).head())

    return mean_comparison

# Analisi delle correlazioni tra le feature
def analyze_correlations(data):
    # Seleziona solo le colonne numeriche
    numeric_data = data.select_dtypes(include=[np.number])
    
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
    plt.title('Matrice di correlazione delle feature')
    plt.tight_layout()
    plt.savefig('plots_fault/correlation_matrix.png')
    plt.close()

    # Feature più correlate con il target
    target_correlations = correlation_matrix['target'].sort_values(key=abs, ascending=False)
    print("\nFeature più correlate con il target (guasto):")
    print(target_correlations.head(10))

    return target_correlations

# Analisi delle componenti principali (PCA)
def perform_pca(data):
    # Seleziona solo le colonne numeriche, escludendo 'target' e 'measure_date'
    features = data.select_dtypes(include=[np.number]).drop(['target'], axis=1, errors='ignore')
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA()
    pca_result = pca.fit_transform(scaled_features)

    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Numero di componenti')
    plt.ylabel('Varianza cumulativa spiegata')
    plt.title('Analisi delle Componenti Principali (PCA)')
    plt.savefig('plots_fault/pca_cumulative_variance.png')
    plt.close()

    print("\nNumero di componenti necessarie per spiegare il 95% della varianza:")
    print(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1)

    return pca

def main():
    # Assicurati che la directory per i plot esista
    ensure_dir('plots_fault')

    # Carica i dati con feature engineering
    print("Caricamento dei dati con feature engineering...")
    data_engineered = pd.read_csv('csv/data_with_interactions.csv', parse_dates=['measure_date'])

    # Crea la colonna target
    data_engineered = create_target(data_engineered)

    # Esegui le analisi
    fault_characteristics = analyze_fault_characteristics(data_engineered)
    feature_correlations = analyze_correlations(data_engineered)
    pca_results = perform_pca(data_engineered)

    print("\nAnalisi completata. Controlla le visualizzazioni generate nella cartella 'plots_fault'.")

if __name__ == "__main__":
    main()