from google.cloud import storage, bigquery
import pandas as pd
import os
import re
import tempfile
import pyarrow.parquet as pq

# Variables globales del proyecto
project_id = 'sturdy-gate-417001'  # Proyecto
bucket_name = 'data_co2_new_york'  # Bucket 
dataset_id = 'data_clean'  # DataSet
table_id = 'CarbonEmissionNYC'  # Nombre de la tabla

# Configuracion de credenciales para la conexion a Google Cloud Platform
# El archivo 'sturdy-gate-417001-19ab59ab9df1.json' que tiene las credenciales de la cuenta de
# servicio en GCP se descarga cuando se crea una nueva cuenta de servicio y se agrega una nueva clave.
#key_path = '../../sturdy-gate-417001-19ab59ab9df1.json'
#client = bigquery.Client.from_service_account_json(key_path)

def process_emission(event, context):
    """Se activa cada vez que se sube un archivo al bucket 'data_co2_new_york'."""
    # Asegúrate de que esta línea refleje correctamente el nombre del archivo que deseas procesar
    file_name = event['name']

    client_storage = storage.Client()
    bucket = client_storage.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Ruta temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)
    
    # Leer el archivo de hoja de calculo 'CY_2005_CY_2022_citywide.xlsx'
    # Cargamos un dataframe con el dataset de las emisiones de carbono en New York
    df = pd.read_csv(temp_file.name)

    # ETL
    # Seleccionamos las columnas que vamos a utilizar
    columns_to_keep = ['Sectors Sector', 'Category Full', 'Category Label','Source Label'] + [col for col in df.columns if col.startswith('CY')]

    # Crear un nuevo DataFrame con las columnas seleccionadas
    df_normalized = df[columns_to_keep].copy()

    # Realizar la operación de fundido (melt)
    df_normalized = pd.melt(df_normalized, id_vars=['Sectors Sector', 'Category Full', 'Category Label','Source Label'], var_name='Año', value_name='Valor')

    # Seleccionamos las columnas que vamos a utilizar
    columns_to_keep = ['Sectors Sector', 'Category Full', 'Category Label','Source Label'] + [col for col in df.columns if col.startswith('CY')]

    # Crear un nuevo DataFrame con las columnas seleccionadas
    df_normalized = df[columns_to_keep].copy()

    # Realizar la operación de fundido (melt)
    df_normalized = pd.melt(df_normalized, id_vars=['Sectors Sector', 'Category Full', 'Category Label','Source Label'], var_name='Año', value_name='Valor')

    # Crear una columna auxiliar 'Tipo' y procesar la columna 'Año'
    df_normalized['Tipo'] = df_normalized['Año'].apply(lambda x: re.findall(r'(?:\d{4})\s(.+)', x)[0])
    df_normalized['Año'] = df_normalized['Año'].apply(lambda x: re.findall(r'(\d{4})', x)[0])

    # Filtrar solo las filas donde el tipo es 'Consumed' o 'tCO2e'
    df_normalized = df_normalized[df_normalized['Tipo'].isin(['Consumed', 'tCO2e','Source MMBtu'])]

    # Utilizar pivot_table para convertir las categorías de 'Tipo' en columnas
    df_pivot = df_normalized.pivot_table(index=['Sectors Sector', 'Category Full', 'Category Label','Source Label', 'Año'],
                                     columns='Tipo', values='Valor', aggfunc='sum').reset_index()

    # Eliminar registros nulos
    df_pivot.dropna(inplace=True)

    # Eliminar registros duplicados
    df_pivot.drop_duplicates(inplace=True)

    # Inicializar cliente de BigQuery y configurar la carga
    client_bigquery = bigquery.Client()
    table_ref = client_bigquery.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Autodetectar esquema
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    #job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

    # Cargar datos a BigQuery
    job = client_bigquery.load_table_from_dataframe(df_pivot, table_ref, job_config=job_config)
    job.result()  # Esperar a la carga

    # Limpiar archivo temporal
    temp_file.close()
    os.remove(temp_file.name)
   
    print(f"El archivo {file_name} ha sido procesado y cargado a BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_emission --runtime python39 --trigger-bucket data_co2_new_york --entry-point process_emission --timeout 540s --memory 256MB
