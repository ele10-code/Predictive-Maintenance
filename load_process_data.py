import pymysql
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import config
import os
from datetime import datetime

# Configurations from the config file
try:
    DB_USER_SERVER = config.DB_USER_SERVER
    DB_PASSWORD_SERVER = config.DB_PASSWORD_SERVER
    DB_NAME_SERVER = config.DB_NAME_SERVER
except AttributeError:
    DB_USER_SERVER = None
    DB_PASSWORD_SERVER = None
    DB_NAME_SERVER = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the 'csv' folder if it doesn't exist
csv_folder = 'csv'
os.makedirs(csv_folder, exist_ok=True)

def clean_data(df):
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'event_reference_value' in df.columns:
        df['event_reference_value'] = pd.to_numeric(df['event_reference_value'], errors='coerce')
    df = df.dropna(subset=['value', 'event_reference_value'])
    if 'measure_date' in df.columns:
        df['measure_date'] = pd.to_datetime(df['measure_date'], errors='coerce')
    return df

def process_device(id_device, db_user, db_password, db_name, host, port, year):
    logging.info(f"Starting processing for device {id_device} on {host}:{port}")
    try:
        # Attempt to establish a connection to the database
        connection = pymysql.connect(
            host=host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=port
        )
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        return  # Exit the function if connection fails

    try:
        with connection.cursor() as cursor:
            # Query to get events for the specified device in July and August
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
              AND YEAR(de.date_start) = %s
              AND MONTH(de.date_start) IN (7, 8)
            ORDER BY de.date_start;
            """
            cursor.execute(event_query, (id_device, year))
            events = cursor.fetchall()

            # List to store DataFrames of measures
            all_measures_dataframes = []

            for event in events:
                event_id = event[0]
                date_start = event[2]
                operator_value = event[5]

                # Get the date (day) of the event
                event_date = pd.to_datetime(date_start).date()

                # Define the start and end of the day
                # day_start = datetime.combine(event_date, datetime.min.time())
                day_start = date_start - pd.Timedelta(days=7)
                day_end = datetime.combine(event_date, datetime.max.time())

                # Query to get all measures on the day of the event
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
                cursor.execute(measure_query, (event_id, event[4], event[5], event[6], id_device, day_start, day_end))
                measures = cursor.fetchall()

                if not measures:
                    logging.info(f"No measures found for event {event_id} on the day of the event.")
                    continue

                # Convert measures to DataFrame
                measure_columns = [
                    'measure_id', 'oid', 'measure_name', 'value', 'value_label',
                    'measure_date', 'device_name', 'event_id', 'event_variable',
                    'event_operator', 'event_reference_value'
                ]
                measures_df = pd.DataFrame(measures, columns=measure_columns)

                # Add the DataFrame to the list
                all_measures_dataframes.append(measures_df)

            # Check if there are DataFrames to concatenate
            if all_measures_dataframes:
                # Combine all DataFrames into one
                combined_df = pd.concat(all_measures_dataframes, ignore_index=True)
                combined_df = clean_data(combined_df)

                # Create device folder inside 'csv'
                device_folder = os.path.join(csv_folder, f'device_{id_device}')
                os.makedirs(device_folder, exist_ok=True)

                # Save the combined DataFrame to a CSV file for the device in the 'csv/device_{id_device}' folder
                combined_csv_path = os.path.join(device_folder, f'measures_device_{id_device}_jul_aug_{year}.csv')
                combined_df.to_csv(combined_csv_path, index=False)
                logging.info(f"All measures on the day of events for device {id_device} for July and August {year} have been saved to {combined_csv_path}")
            else:
                logging.info(f"No measures found on the day of events for device {id_device} for July and August {year}.")

    except pymysql.MySQLError as e:
        logging.error(f"Error executing query for device {id_device}: {e}")
    except Exception as e:
        logging.error(f"General error for device {id_device}: {e}")
    finally:
        # Ensure that the connection is closed
        if connection and connection.open:
            connection.close()

def main():
    server = {
        'host': '18.200.74.134',
        'user': DB_USER_SERVER,
        'password': DB_PASSWORD_SERVER,
        'database': DB_NAME_SERVER,
        'port': 3306
    }

    # Get the current year
    current_year = datetime.now().year

    try:
        connection = pymysql.connect(
            host=server['host'],
            user=server['user'],
            password=server['password'],
            database=server['database'],
            port=server['port']
        )
        with connection.cursor() as cursor:
            # Get the list of devices with id_device_type = 1
            device_query = """
            SELECT d.ID
            FROM txControl.devices d
            JOIN txControl.device_models dm ON d.id_device_model = dm.ID
            WHERE dm.id_device_type = 1;
            """
            cursor.execute(device_query)
            devices = cursor.fetchall()

        # Parallelize the processing of devices
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    process_device,
                    device[0],
                    server['user'],
                    server['password'],
                    server['database'],
                    server['host'],
                    server['port'],
                    current_year
                ) for device in devices
            ]
            for future in as_completed(futures):
                future.result()

    except pymysql.MySQLError as e:
        logging.error(f"Error executing device query: {e}")
    except Exception as e:
        logging.error(f"General error in main: {e}")
    finally:
        if 'connection' in locals() and connection.open:
            connection.close()

if __name__ == "__main__":
    main()
