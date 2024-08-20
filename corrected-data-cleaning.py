import pandas as pd
import numpy as np

# Carica i dati
print("Caricamento dei dati...")
data = pd.read_csv('csv/data_pre_processing.csv', 
                   na_values=['NOT APPLICABLE', 'NOT  APPLICABLE', 'NaN', ''], 
                   keep_default_na=True,
                   low_memory=False)

print("Dati caricati. Informazioni sul DataFrame originale:")
print(data.info())

# Converti le colonne nel tipo di dato appropriato
data['measure_name'] = data['measure_name'].astype('category')
data['value'] = pd.to_numeric(data['value'], errors='coerce')
data['measure_date'] = pd.to_datetime(data['measure_date'], errors='coerce')
data['event_id'] = pd.to_numeric(data['event_id'], errors='coerce').astype('Int64')  # Int64 permette NaN
data['event_variable'] = data['event_variable'].astype('category')
data['event_operator'] = data['event_operator'].astype('category')
data['event_reference_value'] = pd.to_numeric(data['event_reference_value'], errors='coerce')

# Rimuovi le righe con valori NaN
data_cleaned = data.dropna()

print("\nInformazioni sul DataFrame dopo la pulizia:")
print(data_cleaned.info())

# Salva il DataFrame pulito
output_file = 'csv/cleaned_data.csv'
data_cleaned.to_csv(output_file, index=False)
print(f"\nDati puliti salvati in {output_file}")

# Stampa alcune statistiche di base
print("\nStatistiche di base per le colonne numeriche:")
print(data_cleaned.describe())

print("\nConteggi per le colonne categoriche:")
for col in ['measure_name', 'event_variable', 'event_operator']:
    print(f"\n{col}:")
    print(data_cleaned[col].value_counts().head())

# Verifica dei valori unici nelle colonne categoriche
print("\nValori unici nelle colonne categoriche:")
for col in ['measure_name', 'event_variable', 'event_operator']:
    print(f"\n{col}:")
    print(data_cleaned[col].unique())

# Verifica della distribuzione temporale
print("\nDistribuzione temporale dei dati:")
print(data_cleaned['measure_date'].describe())

