import pymysql
import pandas as pd
import config
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configurazioni dal file config
DB_USER = config.DB_USER
DB_PASSWORD = config.DB_PASSWORD
DB_NAME = config.DB_NAME

# Parametri specifici
lookback_hours = 24  # Ore di lookback

def process_device(id_device):
    try:
        with pymysql.connect(
            host='54.195.165.244',
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=3306
        ) as connection:
            with connection.cursor() as cursor:
                # Esegui la query per ottenere tutti gli eventi per il dispositivo specificato
                event_query = """
                SELECT 
                    de.id AS event_id,
                    de.id_device,
                    de.date_start,
                    de.date_end,
                    de.variable AS event_variable,
                    de.operator,
                    de.reference_value,
                    d.name AS device_name,
                    d.ip_address
                FROM device_events de
                JOIN devices d ON de.id_device = d.id
                WHERE de.id_device = %s
                ORDER BY de.date_start;
                """
                cursor.execute(event_query, (id_device,))
                events = cursor.fetchall()

                # Lista per memorizzare i DataFrame delle misure
                all_measures_dataframes = []

                for event in events:
                    event_id = event[0]
                    date_start = event[2]
                    operator_value = event[5]

                    # Calcola il periodo di 24 ore prima dell'evento
                    lookback_start = pd.to_datetime(date_start) - pd.Timedelta(hours=lookback_hours)

                    # Esegui la query per ottenere tutte le misure delle 24 ore precedenti l'evento
                    measure_query = """
                    SELECT 
                        m.id AS measure_id,
                        m.oid,
                        m.name AS measure_name,
                        m.value,
                        m.value_label,
                        m.measure_date,
                        d.name AS device_name,
                        d.ip_address,
                        %s AS event_id,
                        %s AS event_variable,
                        %s AS event_operator,
                        %s AS event_reference_value
                    FROM measures m
                    JOIN devices d ON m.id_device = d.id
                    WHERE m.id_device = %s
                      AND m.measure_date BETWEEN %s AND %s
                    ORDER BY m.measure_date;
                    """
                    cursor.execute(measure_query, (event_id, event[4], event[5], event[6], id_device, lookback_start, date_start))
                    measures = cursor.fetchall()

                    # Se non ci sono misure per questo evento, continua con il prossimo
                    if not measures:
                        print(f"Nessuna misura trovata per l'evento {event_id} nelle 24 ore precedenti.")
                        continue

                    # Converti le misure in DataFrame
                    measure_columns = ['measure_id', 'oid', 'measure_name', 'value', 'value_label', 'measure_date', 'device_name', 'ip_address', 'event_id', 'event_variable', 'event_operator', 'event_reference_value']
                    measures_df = pd.DataFrame(measures, columns=measure_columns)

                    # Aggiungi il DataFrame alla lista
                    all_measures_dataframes.append(measures_df)

                # Controlla se ci sono DataFrame da concatenare
                if all_measures_dataframes:
                    # Combina tutti i DataFrame in uno solo
                    combined_df = pd.concat(all_measures_dataframes, ignore_index=True)

                    # Salva il DataFrame combinato in un file CSV per il dispositivo
                    combined_csv_path = f'measures_24h_before_events_device_{id_device}.csv'
                    combined_df.to_csv(combined_csv_path, index=False)
                    print(f"Tutte le misure per le 24 ore precedenti gli eventi del dispositivo {id_device} sono state salvate in {combined_csv_path}")
                else:
                    print(f"Nessuna misura trovata per le 24 ore precedenti gli eventi del dispositivo {id_device}.")
    except pymysql.MySQLError as e:
        print(f"Errore durante l'esecuzione della query: {e}")

if __name__ == "__main__":
    try:
        connection = pymysql.connect(
            host='54.195.165.244',
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=3306  # Assicurati di specificare la porta
        )
        with connection.cursor() as cursor:
            # Ottenere l'elenco unico dei dispositivi
            cursor.execute("SELECT DISTINCT id_device FROM device_events")
            devices = cursor.fetchall()

        # Parallelizza l'elaborazione dei dispositivi
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_device, device[0]) for device in devices]
            for future in as_completed(futures):
                future.result()

    except pymysql.MySQLError as e:
        print(f"Errore durante l'esecuzione della query: {e}")
    finally:
        connection.close()
