import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carica i dati delle metriche
metrics_df = pd.read_csv('metrics_results.csv')

# Creazione del grafico a linee smussate per ogni dispositivo
device_ids = metrics_df['device_id'].unique()

for device_id in device_ids:
    device_data = metrics_df[metrics_df['device_id'] == device_id]
    
    plt.figure(figsize=(10, 6))
    
    # Plot accurato per ogni metrica con linee smussate
    plt.plot(device_data['iteration'], device_data['accuracy'], marker='o', linestyle='-', color='blue', label='Accuracy')
    plt.plot(device_data['iteration'], device_data['f1_score'], marker='s', linestyle='-', color='green', label='F1 Score')
    plt.plot(device_data['iteration'], device_data['roc_auc_score'], marker='^', linestyle='-', color='red', label='ROC-AUC')
    
    plt.title(f"Performance del Modello per Dispositivo {device_id}")
    plt.xlabel("Model Iteration")
    plt.ylabel("Metriche di Performance")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    # Salva il grafico per ogni dispositivo
    plt.savefig(f'plots_xgboost_optimized_2000000/smoothed_performance_chart_device_{device_id}.png')
    plt.show()
