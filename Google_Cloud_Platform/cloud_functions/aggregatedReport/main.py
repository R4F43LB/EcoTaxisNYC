# by Rafael Balestrini y Carlos Macea

from google.cloud import storage, bigquery
import pandas as pd
import os
import tempfile
import pyarrow.parquet as pq

# Variables globales del proyecto
PROJECT_ID = 'sturdy-gate-417001'  # Proyecto
BUCKET_NAME = 'data_aggregated_report'  # Nombre del Bucket
DATASET_ID = 'data_clean'  # ID del DataSet
TABLE_ID = 'AggregatedReport'  # Nombre de la tabla

def process_aggregated_report(event, context):
    """Función que se activa con cada archivo subido al bucket especificado."""
    FILE_NAME = event['name']
    client_storage = storage.Client()
    bucket = client_storage.bucket(BUCKET_NAME)
    blob = bucket.blob(FILE_NAME)
    TEMP_FILE = tempfile.NamedTemporaryFile(delete=False)

#    try:
    blob.download_to_filename(TEMP_FILE.name)
    df = pd.read_csv(TEMP_FILE.name)

    # Proceso ETL
    df = df[df['License Class'].isin(['FHV - High Volume', 'Green', 'Yellow'])]
    df['License Class'] = df['License Class'].replace({'FHV - High Volume': 'UberLyft'})
    df['Percent of Trips Paid with Credit Card'] = df['Percent of Trips Paid with Credit Card'].str.replace('%', '')

    columns_to_replace = ['Trips Per Day', 'Farebox Per Day',
           'Unique Drivers', 'Unique Vehicles', 'Vehicles Per Day',
           'Avg Days Vehicles on Road', 'Avg Hours Per Day Per Vehicle',
           'Avg Days Drivers on Road', 'Avg Hours Per Day Per Driver',
           'Avg Minutes Per Trip', 'Percent of Trips Paid with Credit Card',
           'Trips Per Day Shared']
    # Renombrar las columnas 'Edad' y 'Departamento'
    df = df.rename(columns={'Month/Year': 'monthYear'})
    print(df.head(2))
    for column in columns_to_replace:
        if df[column].dtype == 'object':  # Verificar si la columna es de tipo cadena
            df[column] = df[column].str.replace(',', '')
        df[column] = df[column].replace('-', '0')
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Convertir a tipo float32, ignorando los errores de conversión

    print(df.head(2))
    # Convertir los float64 a float32
    df[[
        'Avg Days Vehicles on Road', 'Avg Hours Per Day Per Vehicle', 'Avg Days Drivers on Road',
        'Avg Hours Per Day Per Driver', 'Avg Minutes Per Trip'
    ]] = df[[
            'Avg Days Vehicles on Road', 'Avg Hours Per Day Per Vehicle', 'Avg Days Drivers on Road',
            'Avg Hours Per Day Per Driver', 'Avg Minutes Per Trip'
         ]].astype('float32')
    
    df[[
        'Trips Per Day', 'Farebox Per Day', 'Unique Drivers', 'Unique Vehicles',
        'Vehicles Per Day', 'Percent of Trips Paid with Credit Card', 'Trips Per Day Shared'
    ]] = df[[
            'Trips Per Day', 'Farebox Per Day', 'Unique Drivers', 'Unique Vehicles',
            'Vehicles Per Day', 'Percent of Trips Paid with Credit Card', 'Trips Per Day Shared'
         ]].astype('float32')  # Convertir a float32

    df[['Year', 'Month']] = df['monthYear'].str.split('-', expand=True)
    
    # Convertir las columnas "Year" y "Month" a tipo numérico
    df['Year'] = pd.to_numeric(df['Year'])
    df['Month'] = pd.to_numeric(df['Month'])
    
    # Filtrar para incluir solo los registros a partir de 2020
    df = df[df['Year'] >= 2020]
    
    # Convertir Year y Month a int16
    df[['Year', 'Month']] = df[['Year', 'Month']].astype('int16')
    
    # Filtrar para incluir solo los registros a partir de 2020
    df = df[df['Year'] >= 2020]

    # Ordenar columnas y eliminar las que no serán necesarias
    # Definir las dimensiones de agrupación y las variables de agregación
    dimensiones = [ 'Year', 'Month', 'License Class']
    variables_agregacion = ['Unique Drivers', 'Unique Vehicles', 'Vehicles Per Day',
           'Avg Days Vehicles on Road', 'Avg Hours Per Day Per Vehicle',
           'Avg Days Drivers on Road', 'Avg Hours Per Day Per Driver',
           'Avg Minutes Per Trip', 'Percent of Trips Paid with Credit Card',
           'Trips Per Day Shared']
    
    # Reordenar las columnas para que primero estén las dimensiones y luego las variables de agregación
    columnas_ordenadas = dimensiones + variables_agregacion
    
    # Reordenar el DataFrame agrupado
    df = df[columnas_ordenadas]


    # Configuración del cliente de BigQuery y carga de datos
    #client_bigquery = bigquery.Client()
    #table_ref = client_bigquery.dataset(DATASET_ID).table(TABLE_ID)
    #job_config = bigquery.LoadJobConfig()
    #job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

    # Cargar datos a BigQuery
    df.to_gbq(destination_table=f'{DATASET_ID}.{TABLE_ID}', project_id=PROJECT_ID, if_exists='replace')
    #client_bigquery.load_table_from_dataframe(df, table_ref, job_config=job_config).result()

    print(f"El archivo {FILE_NAME} ha sido procesado y cargado correctamente en BigQuery.")
#    except Exception as e:
#        print(f"Error procesando el archivo {FILE_NAME}: {e}")
#    finally:
#        TEMP_FILE.close()
#        os.remove(TEMP_FILE.name)

# Nota: Este código es para uso dentro de Google Cloud Functions.

# gcloud functions deploy process_aggregated_report --runtime python39 --trigger-bucket data_aggregated_report --entry-point process_aggregated_report --timeout 540s --memory 512MB