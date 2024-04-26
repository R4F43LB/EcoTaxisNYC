from google.cloud import storage, bigquery
import pandas as pd
import os
import tempfile

# Variables globales del proyecto
# project_id = 'sturdy-gate-417001'  # Proyecto
BUCKET_NAME = 'data_aggregated_report'  # Bucket 
DATASET_ID = 'data_clean'  # DataSet
TABLE_ID = 'AggregatedReport'  # Nombre de la tabla

# Configuracion de credenciales para la conexion a Google Cloud Platform
# El archivo 'sturdy-gate-417001-19ab59ab9df1.json' que tiene las credenciales de la cuenta de
# servicio en GCP se descarga cuando se crea una nueva cuenta de servicio y se agrega una nueva clave.
#key_path = '../../sturdy-gate-417001-19ab59ab9df1.json'
#client = bigquery.Client.from_service_account_json(key_path)

def process_aggregated_report(event, context):
    """Se activa cada vez que se sube un archivo al bucket 'data_aggregated_report'."""
    # Asegúrate de que esta línea refleje correctamente el nombre del archivo que deseas procesar
    FILE_NAME = event['name']

    client_storage = storage.Client()
    bucket = client_storage.bucket(BUCKET_NAME)
    blob = bucket.blob(FILE_NAME)

    # Ruta temporal
    TEMP_FILE = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(TEMP_FILE.name)
    
    # Leer el archivo CSV
    # Cargamos un dataframe con el dataset de sonidos en New York
    df = pd.read_csv(TEMP_FILE.name)
    
    # ETL
    # Renombrar datos y filtrar solo UberLyft, Amarillos y Verdes
    # Filtrar el DataFrame por 'License Class' y renombrar valores
    df = df[df['License Class'].isin(['FHV - High Volume', 'Green', 'Yellow'])]
    df['License Class'] = df['License Class'].replace({'FHV - High Volume': 'UberLyft'})

    # Corregir formato de valores y disminuir tamaño
    # Reemplazar el símbolo "%" por una cadena vacía en la columna "Percent of Trips Paid with Credit Card"
    df['Percent of Trips Paid with Credit Card'] = df['Percent of Trips Paid with Credit Card'].str.replace('%', '')
    
    # Reemplazar los guiones "-" con ceros en las columnas que deseas convertir a float32
    columns_to_replace = ['Trips Per Day', 'Farebox Per Day',
           'Unique Drivers', 'Unique Vehicles', 'Vehicles Per Day',
           'Avg Days Vehicles on Road', 'Avg Hours Per Day Per Vehicle',
           'Avg Days Drivers on Road', 'Avg Hours Per Day Per Driver',
           'Avg Minutes Per Trip', 'Percent of Trips Paid with Credit Card',
           'Trips Per Day Shared']
    
    for column in columns_to_replace:
        if df[column].dtype == 'object':  # Verificar si la columna es de tipo cadena
            df[column] = df[column].str.replace(',', '')
    
    for column in columns_to_replace:
        df[column] = df[column].replace('-', '0')
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Convertir a tipo float32, ignorando los errores de conversión
    
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

    # Filtrar datos desde el 2020 en adelante
    # Month/Year en esta columna el año viene separado del mes con "-" por ejemplo 2024-01 para
    # indicar enero de 2024. Se requiere separar mes y año y solo dejar los registros de los años
    # 2020 en adelante
    # Separar la columna "Month/Year" en "Year" y "Month"
    df[['Year', 'Month']] = df['Month/Year'].str.split('-', expand=True)
    
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

    # Inicializar cliente de BigQuery y configurar la carga
    client_bigquery = bigquery.Client()
    TABLE_REF = client_bigquery.dataset(DATASET_ID).table(TABLE_ID)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Autodetectar esquema
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

    # Cargar datos a BigQuery
    job = client_bigquery.load_table_from_dataframe(df, TABLE_REF, job_config=job_config)
    job.result()  # Esperar a la carga

    # Limpiar archivo temporal
    TEMP_FILE.close()
    os.remove(TEMP_FILE.name)
    
    print(f"El archivo {FILE_NAME} ha sido procesado y cargado a BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_aggregated_report --runtime python39 --trigger-bucket data_aggregated_report --entry-point process_aggregated_report --timeout 540s --memory 256MB
