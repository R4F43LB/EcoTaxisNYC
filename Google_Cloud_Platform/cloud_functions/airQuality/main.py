from google.cloud import storage, bigquery
import pandas as pd
import os
import tempfile

# Variables globales del proyecto
project_id = 'sturdy-gate-417001'  # Proyecto
bucket_name = 'data_calidad_aire_new_york'  # Bucket 
dataset_id = 'data_clean'  # DataSet
table_id = 'AirQuality'  # Nombre de la tabla

# Configuracion de credenciales para la conexion a Google Cloud Platform
# El archivo 'sturdy-gate-417001-19ab59ab9df1.json' que tiene las credenciales de la cuenta de
# servicio en GCP se descarga cuando se crea una nueva cuenta de servicio y se agrega una nueva clave.
#key_path = '../../sturdy-gate-417001-19ab59ab9df1.json'
#client = bigquery.Client.from_service_account_json(key_path)

def process_air_quality(event, context):
    """Se activa cada vez que se sube un archivo al bucket 'data_calidad_aire_new_york'."""
    # Asegúrate de que esta línea refleje correctamente el nombre del archivo que deseas procesar
    file_name = event['name']

    client_storage = storage.Client()
    bucket = client_storage.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Ruta temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)
    
    # Leer el archivo CSV
    # Cargamos un dataframe con el dataset de calidad de aire de New York
    df = pd.read_csv(temp_file.name)
    
    # ETL
    df = df[df['Geo Type Name'].isin(['Borough'])]
    df = df.pivot_table(
        index=[
            'Unique ID',
            'Indicator ID',
            'Measure',
            'Measure Info',
            'Geo Type Name',
            'Geo Join ID',
            'Geo Place Name',
            'Time Period',
            'Start_Date'],
        columns='Name',
        values='Data Value'
    ).reset_index()
    
    variables = [
        'Time Period',
        'Geo Place Name',
        'Fine particles (PM 2.5)',
        'Nitrogen dioxide (NO2)',
        'Ozone (O3)'
    ]
    df = df[variables]
    df['Time Period'] = df['Time Period'].str.extract(r'(\d{4})').astype(int)
    df.rename(
        columns={
            'Time Period': 'year',
            'Geo Place Name': 'borough',
            'Fine particles (PM 2.5)': 'fineParticlesPM25',
            'Nitrogen dioxide (NO2)': 'nitrogenDioxideNO2',
            'Ozone (O3)': 'ozoneO3'
        },
        inplace=True
    )
    df.columns.name = None
    df = df.groupby(['year', 'borough']).agg('max').reset_index().dropna(how='any')

    # Inicializar cliente de BigQuery y configurar la carga
    client_bigquery = bigquery.Client()
    table_ref = client_bigquery.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Autodetectar esquema
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

    # Cargar datos a BigQuery
    job = client_bigquery.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Esperar a la carga

    # Limpiar archivo temporal
    temp_file.close()
    os.remove(temp_file.name)
    
    print(f"El archivo {file_name} ha sido procesado y cargado a BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_air_quality --runtime python39 --trigger-bucket data_calidad_aire_new_york --entry-point process_air_quality --timeout 540s --memory 256MB
