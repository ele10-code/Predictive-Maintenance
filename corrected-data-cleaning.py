import pandas as pd
import numpy as np
import os
import glob

# Configurazione del logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cartella contenente i file CSV generati
input_folder = 'csv'

# File di output per i dati puliti
output_file = 'csv/cleaned_combined_data.csv'

def clean_and_combine_data():
    logging.info("Inizio del processo di pulizia e combinazione dei dati...")
    
    # Lista per memorizzare tutti i DataFrame
    all_dataframes = []
    
    # Leggi tutti i file CSV nella cartella di input
    for csv_file in glob.glob(os.path.join(input_folder, 'measures_72h_before_events_device_*.csv')):
        logging.info(f"Elaborazione del file: {csv_file}")
        
        # Estrai l'ID del dispositivo dal nome del file
        device_id = os.path.basename(csv_file).split('_')[5]
        
        # Leggi il file CSV
        df = pd.read_csv(csv_file, 
                         na_values=['NOT APPLICABLE', 'NOT  APPLICABLE', 'NaN', ''], 
                         keep_default_na=True,
                         low_memory=False)
        
        # Aggiungi una colonna per l'ID del dispositivo
        df['device_id'] = device_id
        
        # Aggiungi il DataFrame alla lista
        all_dataframes.append(df)
    
    # Combina tutti i DataFrame
    combined_data = pd.concat(all_dataframes, ignore_index=True)
    
    logging.info("Dati combinati. Inizio della pulizia...")
    
    # Converti le colonne nel tipo di dato appropriato
    combined_data['measure_name'] = combined_data['measure_name'].astype('category')
    combined_data['value'] = pd.to_numeric(combined_data['value'], errors='coerce')
    combined_data['measure_date'] = pd.to_datetime(combined_data['measure_date'], errors='coerce')
    combined_data['event_id'] = pd.to_numeric(combined_data['event_id'], errors='coerce').astype('Int64')  # Int64 permette NaN
    combined_data['event_variable'] = combined_data['event_variable'].astype('category')
    combined_data['event_operator'] = combined_data['event_operator'].astype('category')
    combined_data['event_reference_value'] = pd.to_numeric(combined_data['event_reference_value'], errors='coerce')
    
    # Rimuovi le righe con valori NaN
    cleaned_data = combined_data.dropna()
    
    logging.info("Pulizia completata. Salvataggio dei dati puliti...")
    
    # Salva il DataFrame pulito
    cleaned_data.to_csv(output_file, index=False)
    logging.info(f"Dati puliti salvati in {output_file}")
    
    # Stampa alcune statistiche di base
    logging.info("\nStatistiche di base per le colonne numeriche:")
    logging.info(cleaned_data.describe())
    
    logging.info("\nConteggi per le colonne categoriche:")
    for col in ['measure_name', 'event_variable', 'event_operator']:
        logging.info(f"\n{col}:")
        logging.info(cleaned_data[col].value_counts().head())
    
    # Verifica dei valori unici nelle colonne categoriche
    logging.info("\nValori unici nelle colonne categoriche:")
    for col in ['measure_name', 'event_variable', 'event_operator']:
        logging.info(f"\n{col}:")
        logging.info(cleaned_data[col].unique())
    
    # Verifica della distribuzione temporale
    logging.info("\nDistribuzione temporale dei dati:")
    logging.info(cleaned_data['measure_date'].describe())

if __name__ == "__main__":
    clean_and_combine_data()