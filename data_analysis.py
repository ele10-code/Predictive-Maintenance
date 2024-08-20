import pymysql
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import config

# Configurazioni dal file config
try:
    DB_USER_SERVER = config.DB_USER_SERVER
    DB_PASSWORD_SERVER = config.DB_PASSWORD_SERVER
    DB_NAME_SERVER = config.DB_NAME_SERVER
except AttributeError as e:
    logging.error(f"Errore nella configurazione: {e}")
    DB_USER_SERVER = None
    DB_PASSWORD_SERVER = None
    DB_NAME_SERVER = None

# Parametri specifici
lookback_hours = 24  # Ore di lookback

# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_device(id_device, db_user, db_password, db_name, lookback_hours, host, port):
    logging.info(f"Inizio elaborazione dispositivo {id_device} su {host}:{port}")
    try:
        connection = pymysql.connect(
            host=host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=port
        )
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
                d.name AS device_name
            FROM device_events de
            JOIN devices d ON de.id_device = d.ID
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
                    m.ID AS measure_id,
                    m.oid,
                    m.name AS measure_name,
                    m.value,
                    m.value_label,
                    m.measure_date,
                    d.name AS device_name,
                    %s AS event_id,
                    %s AS event_variable,
                    %s AS event_operator,
                    %s AS event_reference_value
                FROM txControl.measures m
                JOIN txControl.devices d ON m.id_device = d.ID
                WHERE m.id_device = %s
                  AND m.measure_date BETWEEN %s AND %s
                ORDER BY m.measure_date;
                """
                cursor.execute(measure_query, (event_id, event[4], event[5], event[6], id_device, lookback_start, date_start))
                measures = cursor.fetchall()

                # Se non ci sono misure per questo evento, continua con il prossimo
                if not measures:
                    logging.info(f"Nessuna misura trovata per l'evento {event_id} nelle 24 ore precedenti.")
                    continue

                # Converti le misure in DataFrame
                measure_columns = ['measure_id', 'oid', 'measure_name', 'value', 'value_label', 'measure_date', 'device_name', 'event_id', 'event_variable', 'event_operator', 'event_reference_value']
                measures_df = pd.DataFrame(measures, columns=measure_columns)

                # Aggiungi il DataFrame alla lista
                all_measures_dataframes.append(measures_df)

            # Controlla se ci sono DataFrame da concatenare
            if all_measures_dataframes:
                # Combina tutti i DataFrame in uno solo
                combined_df = pd.concat(all_measures_dataframes, ignore_index=True)

                # Salva il DataFrame combinato in un file CSV per il dispositivo
                combined_csv_path = f'csv/measures_24h_before_events_device_{id_device}.csv'
                combined_df.to_csv(combined_csv_path, index=False)
                logging.info(f"Tutte le misure per le 24 ore precedenti gli eventi del dispositivo {id_device} sono state salvate in {combined_csv_path}")
            else:
                logging.info(f"Nessuna misura trovata per le 24 ore precedenti gli eventi del dispositivo {id_device}.")
    except pymysql.MySQLError as e:
        logging.error(f"Errore durante l'esecuzione della query: {e}")
    except Exception as e:
        logging.error(f"Errore generico: {e}")
    finally:
        connection.close()

def main():
    server = {
        'host': '18.200.74.134',
        'user': DB_USER_SERVER,
        'password': DB_PASSWORD_SERVER,
        'database': DB_NAME_SERVER,
        'port': 3306
    }

    if not all([server['user'], server['password'], server['database']]):
        logging.error("Errore: le configurazioni del database non sono definite correttamente.")
        return

    try:
        connection = pymysql.connect(
            host=server['host'],
            user=server['user'],
            password=server['password'],
            database=server['database'],
            port=server['port']  # Assicurati di specificare la porta
        )
        with connection.cursor() as cursor:
            # Ottenere l'elenco dei dispositivi con id_device_type = 1
            device_query = """
            SELECT d.ID
            FROM txControl.devices d
            JOIN txControl.device_models dm ON d.id_device_model = dm.ID
            WHERE dm.id_device_type = 1;
            """
            cursor.execute(device_query)
            devices = cursor.fetchall()

        # Parallelizza l'elaborazione dei dispositivi
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_device, device[0], server['user'], server['password'], server['database'], lookback_hours, server['host'], server['port']) for device in devices]
            for future in as_completed(futures):
                future.result()

    except pymysql.MySQLError as e:
        logging.error(f"Errore durante l'esecuzione della query: {e}")
    finally:
        if connection and connection.open:  # Verifica se la connessione è stata inizializzata e aperta
            connection.close()

if __name__ == "__main__":
    main()
