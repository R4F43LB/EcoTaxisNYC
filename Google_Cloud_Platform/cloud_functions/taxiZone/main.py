from google.cloud import storage, bigquery
import pandas as pd
import os
import tempfile

# Variables globales del proyecto
project_id = 'sturdy-gate-417001'  # Proyecto
bucket_name = 'data_taxi_zone'  # Bucket 
dataset_id = 'data_clean'  # DataSet
table_id = 'TaxiZone'  # Nombre de la tabla

def process_taxi_zone(event, context):
    """Se activa cada vez que se sube un archivo al bucket 'data_taxi_zone'."""
    # Asegúrate de que esta línea refleje correctamente el nombre del archivo que deseas procesar
    file_name = event['name']

    client_storage = storage.Client()
    bucket = client_storage.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Ruta temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)
    
    # Leer el archivo CSV
    # Cargamos un dataframe con el dataset de las zonas de New York
    df = pd.read_csv(temp_file.name)

    # ETL

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
    
    print(f"El archivo {file_name} ha sido procesado y cargado a BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_taxi_zone --runtime python39 --trigger-bucket data_taxi_zone --entry-point process_taxi_zone --timeout 540s --memory 512MB
