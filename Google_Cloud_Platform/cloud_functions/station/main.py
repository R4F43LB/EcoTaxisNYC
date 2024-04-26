from google.cloud import storage, bigquery
import pandas as pd
import os
import tempfile

# Variables globales del proyecto
project_id = 'sturdy-gate-417001'  # Proyecto
bucket_name = 'data_stations'  # Bucket 
dataset_id = 'data_clean'  # DataSet
table_id = 'ElectricalStation'  # Nombre de la tabla

def process_station(event, context):
    """Se activa cada vez que se sube un archivo al bucket 'data_stations'."""
    # Asegúrate de que esta línea refleje correctamente el nombre del archivo que deseas procesar
    file_name = event['name']

    client_storage = storage.Client()
    bucket = client_storage.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Ruta temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)
    
    # Leer el archivo CSV
    # Cargamos un dataframe con el dataset de las estaciones de carga en New York
    df = pd.read_csv(temp_file.name)

    # ETL
     # Seleccionar las columnas requeridas
    df = df[['Station Name', 'City', 'Latitude', 'Longitude']]
    # Asegurarse de que las columnas 'Latitude' y 'Longitude' están en el formato adecuado (float)
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    nuevos_nombres = {
        'Station Name': 'stationName',
        'City': 'city',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
    }

    # Renombrar las columnas utilizando el método rename()
    df = df.rename(columns=nuevos_nombres)

    # Inicializar cliente de BigQuery y configurar la carga
    client_bigquery = bigquery.Client()
    table_ref = client_bigquery.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Autodetectar esquema
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    # Cargar datos a BigQuery
    job = client_bigquery.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Esperar a la carga

    # Limpiar archivo temporal
    temp_file.close()
    os.remove(temp_file.name)
    
    print(f"El archivo {file_name} ha sido procesado y cargado a la tabla {table_id} de BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_station --runtime python39 --trigger-bucket data_stations --entry-point process_station --timeout 540s --memory 256MB
