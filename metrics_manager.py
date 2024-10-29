import json
from datetime import datetime
import os

class MetricsManager:
    def __init__(self, file_path='metrics.json'):
        self.file_path = file_path
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump([], f)

    def save_metric(self, model_name, metric_name, metric_value):
        try:
            with open(self.file_path, 'r+') as f:
                try:
                    metrics = json.load(f)
                except json.JSONDecodeError as e:
                    # Log l'errore e rigenera il file se corrotto
                    print(f"Errore durante il caricamento del JSON: {e}")
                    metrics = []
                
                metrics.append({
                    'timestamp': datetime.now().isoformat(),
                    'model_name': model_name,
                    'metric_name': metric_name,
                    'metric_value': metric_value
                })
                f.seek(0)
                json.dump(metrics, f, indent=4)
                f.truncate()
        except Exception as e:
            print(f"Errore durante il salvataggio della metrica: {e}")


    def get_latest_metrics(self, model_name=None, limit=10):
        with open(self.file_path, 'r') as f:
            metrics = json.load(f)
        
        if model_name:
            metrics = [m for m in metrics if m['model_name'] == model_name]
        
        return metrics[-limit:]

