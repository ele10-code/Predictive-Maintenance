import pandas as pd
import numpy as np

# Carica i dati
print("Caricamento dei dati...")
data = pd.read_csv('csv/data_pre_processing.csv', 
                   na_values=['NOT APPLICABLE', 'NOT  APPLICABLE', 'NaN', ''], 
                   keep_default_na=True)

print("Dati caricati. Informazioni sul DataFrame:")
print(data.info())

# Identifica le colonne problematiche
problematic_columns = []
for col in data.columns:
    try:
        pd.to_numeric(data[col], errors='raise')
    except ValueError:
        problematic_columns.append(col)

print("\nColonne problematiche:", problematic_columns)

# Per ogni colonna problematica
for col in problematic_columns:
    # Stampa alcuni valori unici per capire il contenuto
    print(f"\nValori unici in {col}:")
    print(data[col].unique()[:5])  # Stampa i primi 5 valori unici
    
    # Chiedi all'utente come gestire questa colonna
    action = input(f"Come vuoi gestire la colonna {col}? (skip/remove/replace): ")
    
    if action == 'skip':
        continue
    elif action == 'remove':
        data = data.drop(col, axis=1)
        print(f"Colonna {col} rimossa.")
    elif action == 'replace':
        # Sostituisci i valori non numerici con NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')
        print(f"Valori non numerici in {col} sostituiti con NaN.")
    else:
        print("Azione non valida. La colonna verrà saltata.")

# Dopo aver gestito tutte le colonne problematiche, prova di nuovo la conversione
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col], errors='coerce')

print("\nConversione completata. Verifica il DataFrame risultante.")
print(data.info())

# Salva il DataFrame pulito
output_file = 'csv/cleaned_data.csv'
data.to_csv(output_file, index=False)
print(f"\nDati puliti salvati in {output_file}")