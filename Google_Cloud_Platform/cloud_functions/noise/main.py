from google.cloud import storage, bigquery
import pandas as pd
import os
import tempfile

# Variables globales del proyecto
project_id = 'sturdy-gate-417001'  # Proyecto
bucket_name = 'data_ruidos_new_york'  # Bucket 
dataset_id = 'data_clean'  # DataSet
table_id = 'Noise'  # Nombre de la tabla

# Configuracion de credenciales para la conexion a Google Cloud Platform
# El archivo 'sturdy-gate-417001-19ab59ab9df1.json' que tiene las credenciales de la cuenta de
# servicio en GCP se descarga cuando se crea una nueva cuenta de servicio y se agrega una nueva clave.
#key_path = '../../sturdy-gate-417001-19ab59ab9df1.json'
#client = bigquery.Client.from_service_account_json(key_path)

def process_noise(event, context):
    """Se activa cada vez que se sube un archivo al bucket 'data_ruidos_new_york'."""
    # Asegúrate de que esta línea refleje correctamente el nombre del archivo que deseas procesar
    file_name = event['name']

    client_storage = storage.Client()
    bucket = client_storage.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Ruta temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)
    
    # Leer el archivo CSV
    # Cargamos un dataframe con el dataset de sonidos en New York
    df = pd.read_csv(temp_file.name)
    
    # ETL
    # Solo nos quedaremos con las siguientes variables:
    df_noise = df[[
        'borough',
        'block',
        'latitude',
        'longitude',
        'year',
        'week',
        'day',
        'hour',
        '1-1_small-sounding-engine_presence',
        '1-2_medium-sounding-engine_presence'
    ]]
    
    #Cambia el nombre de las columnas que empizan con un numero
    df_noise = df_noise.rename(
        columns={
            '1-1_small-sounding-engine_presence': 'smallSoundingEngine',
            '1-2_medium-sounding-engine_presence': 'mediumSoundingEngine'
        }
    )

    # Inicializar cliente de BigQuery y configurar la carga
    client_bigquery = bigquery.Client()
    table_ref = client_bigquery.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Autodetectar esquema
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    # Cargar datos a BigQuery
    job = client_bigquery.load_table_from_dataframe(df_noise, table_ref, job_config=job_config)
    job.result()  # Esperar a la carga

    # Limpiar archivo temporal
    temp_file.close()
    os.remove(temp_file.name)
    
    print(f"El archivo {file_name} ha sido procesado y cargado a BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_noise --runtime python39 --trigger-bucket data_ruidos_new_york --entry-point process_noise --timeout 540s --memory 256MB
