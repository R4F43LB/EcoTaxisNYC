from google.cloud import storage, bigquery
import pandas as pd
import os
import tempfile
import numpy as np
from bs4 import BeautifulSoup
import requests
import pyarrow.parquet as pq
import pandas_gbq

# Variables globales del proyecto
project_id = 'sturdy-gate-417001'  # Proyecto
bucket_name = 'data_electric_car'  # Bucket 
dataset_id = 'data_clean'  # DataSet
table_id = 'ElectricCar'  # Nombre de la tabla

def process_electric_car(event, context):
    """Se activa cada vez que se sube un archivo al bucket 'data_electric_car'."""
    file_name = event['name']

    client_storage = storage.Client()
    bucket = client_storage.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Ruta temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)
    
    # Leer el archivo CSV
    # Cargamos un dataframe con el dataset de vehiculos electricos
    df = pd.read_csv(temp_file.name)
    
    #################
    ### TRANSFORM ###
    #################
    # Corregir formato de números y renombrar columnas
    # Renombrar las columnas
    df.rename(columns={
        'Velocidad máxima': 'Velocidad Máxima (Km/h)',
        'Tiempo de 0 a 100 km/h': 'Aceleración 0 a 100 Km/h (seg)',
        'Rango': 'Rango (Km)',
        'Velocidad de carga rápida': 'Carga rápida (Km/h)'
    }, inplace=True)

    # Limpiar los datos numéricos
    df['Velocidad Máxima (Km/h)'] = df['Velocidad Máxima (Km/h)'].str.extract('(\d+)').astype(float)
    df['Aceleración 0 a 100 Km/h (seg)'] = df['Aceleración 0 a 100 Km/h (seg)'].str.extract('(\d+\.?\d*)').astype(float)
    df['Rango (Km)'] = df['Rango (Km)'].str.extract('(\d+)').astype(float)
    df['Carga rápida (Km/h)'] = df['Carga rápida (Km/h)'].str.extract('(\d+)').astype(float)

    # Renombrar columnas de precios y arreglar valores
    # Suponiendo que df es tu DataFrame y 'precios por país' es la columna con los precios
    # Crea columnas vacías para los precios de cada país
    df['Precio Alemania (€)'] = ''
    df['Precio Holanda (€)'] = ''
    df['Precio Inglaterra (£)'] = ''

    # Itera sobre las filas del DataFrame
    for index, row in df.iterrows():
        # Extrae el diccionario de precios de la columna 'precios por país'
        #precios_dict = row['Precios por país']
        precios = row['Precios por país']
        precios_dict = eval(precios)

        # Extrae el precio para Alemania
        if 'FLAG-ICON-DE' in precios_dict:
            df.at[index, 'Precio Alemania (€)'] = precios_dict['FLAG-ICON-DE']
        
        # Extrae el precio para Holanda
        if 'FLAG-ICON-NL' in precios_dict:
            df.at[index, 'Precio Holanda (€)'] = precios_dict['FLAG-ICON-NL']
    
        # Extrae el precio para Inglaterra
        if 'FLAG-ICON-GB' in precios_dict:
            df.at[index, 'Precio Inglaterra (£)'] = precios_dict['FLAG-ICON-GB']

    # Quitar el símbolo del euro (€) de las columnas de precios de Alemania y Holanda
    df['Precio Alemania (€)'] = df['Precio Alemania (€)'].str.replace('€', '')
    df['Precio Holanda (€)'] = df['Precio Holanda (€)'].str.replace('€', '')

    # Quitar el símbolo de la libra esterlina (£) de la columna de precios de Inglaterra
    df['Precio Inglaterra (£)'] = df['Precio Inglaterra (£)'].str.replace('£', '')

    # Reemplazar "N/A" con NaN
    df['Precio Alemania (€)'] = df['Precio Alemania (€)'].replace('N/A', np.nan)
    df['Precio Holanda (€)'] = df['Precio Holanda (€)'].replace('N/A', np.nan)
    df['Precio Inglaterra (£)'] = df['Precio Inglaterra (£)'].replace('N/A', np.nan)

    # Quitar las comas y asteriscos de los valores y convertirlos a tipo numérico
    #print(f"df['Precio Alemania (€)']={df['Precio Alemania (€)']}")
    #print(f"df['Precio Alemania (€)'].str.replace(',', '')  =  {df['Precio Alemania (€)'].str.replace(',', '')}")
    df['Precio Alemania (€)'] = df['Precio Alemania (€)'].str.replace(',', '').str.replace('*', '').astype(float)
    df['Precio Holanda (€)'] = df['Precio Holanda (€)'].str.replace(',', '').str.replace('*', '').astype(float)
    df['Precio Inglaterra (£)'] = df['Precio Inglaterra (£)'].str.replace(',', '').str.replace('*', '').astype(float)

    # Unificar precios a promedio Europa en Euros
    # Crear la columna "Europa (€)" que promedia los valores de las tres columnas de precios, ignorando los valores nulos
    df['Europa (€)'] = df[['Precio Alemania (€)', 'Precio Holanda (€)', 'Precio Inglaterra (£)']].mean(axis=1, skipna=True)

    # Eliminar las columnas individuales de precios
    df.drop(columns=['Precio Alemania (€)', 'Precio Holanda (€)', 'Precio Inglaterra (£)','Precios por país'], inplace=True)

    # Eliminar registros nulos y duplicados
    # Eliminar registros nulos
    df.dropna(inplace=True)

    # Eliminar registros duplicados
    df.drop_duplicates(inplace=True)

    df['Europa (€)']=df['Europa (€)'].round(decimals=2)

    # Cambio de precio a Dolar
    # Convertir a dolar mediante búsqueda de la tasa Scraping a
    # "https://www.google.com/finance/quote/EUR-USD?sa=X&ved=2ahUKEwjUhIO6x9SFAxUjfDABHXpWBXEQmY0JegQIGhAv"

    # URL de la página con la tasa de cambio
    url = "https://www.google.com/finance/quote/EUR-USD?sa=X&ved=2ahUKEwjUhIO6x9SFAxUjfDABHXpWBXEQmY0JegQIGhAv"

    # Realizar la solicitud GET a la página
    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")

    # Encontrar el elemento con la clase "YMlKec fxKbKc"
    div_tasa_cambio = soup.find("div", class_="YMlKec fxKbKc")

    tasa_cambio = div_tasa_cambio.text.strip().replace(',', '.')  # Reemplazar "," por "." para convertir a número decimal
    tasa_cambio = float(div_tasa_cambio.text.strip().replace(',', '.'))

    # Obtener la tasa de conversión actual de EUR a USD
    tasa_conversion = tasa_cambio

    df['Precio($USD)'] = round(df['Europa (€)'] * tasa_conversion, 2)

    df = df.drop(columns=['Europa (€)'])

    # Renombrar las columnas
    df = df.rename(columns={
            'Nombre del vehículo': 'makeModel',
            'Velocidad Máxima (Km/h)': 'maxSpeedKmH',
            'Aceleración 0 a 100 Km/h (seg)': 'zeroToHundredKmH',
            'Rango (Km)': 'rangeKm',
            'Carga rápida (Km/h)': 'fastChargingKmH',
            'Precio($USD)': 'usdPrice'}
    )
    print(df.head(3))
    # Inicializar cliente de BigQuery y configurar la carga
    client_bigquery = bigquery.Client()
    table_ref = client_bigquery.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Autodetectar esquema
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    # Cargar datos a BigQuery
    df.to_gbq(destination_table=f'{dataset_id}.{table_id}', project_id=project_id, if_exists='replace')

    job = client_bigquery.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Esperar a la carga

    # Limpiar archivo temporal
    temp_file.close()
    os.remove(temp_file.name)
    
    print(f"El archivo {file_name} ha sido procesado y cargado a la tabla {table_id} de BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_electric_car --runtime python39 --trigger-bucket data_electric_car --entry-point process_electric_car --timeout 540s --memory 512MB
