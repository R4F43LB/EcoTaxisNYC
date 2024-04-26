from google.cloud import bigquery, storage
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import functions_framework
import tempfile

# Variables globales del proyecto
dataset_id = 'data_clean'  # DataSet
table_id = 'ElectricCar'  # Nombre de la tabla

@functions_framework.cloud_event
def process_electric_car(cloud_event):
    """Se activa una vez al mes."""

    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    local_file_path = f'/tmp/{file_name}'

    ###############
    ### EXTRACT ###
    ###############
    client_storage = storage.Client()
    bucket = client_storage.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Ruta temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    # Download the csv file to a local file in the /tmp directory
    blob.download_to_filename(temp_file.name)

    # Leer el archivo CSV
    # Cargamos los datos de los vehiculos electricos en un DataFrame
    df = pd.read_csv(temp_file.name)

    # Clean up
    # blob.delete()

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
        precios_dict = row['Precios por país']

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

    ############
    ### Load ###
    ############
    # Inicializar cliente de BigQuery y configurar la carga
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    # Cargar datos a BigQuery
    client.load_table_from_dataframe(df, table_ref).result()

#    # Inicializar cliente de BigQuery y configurar la carga
#    client_bigquery = bigquery.Client()
#    table_ref = client_bigquery.dataset(dataset_id).table(table_id)
#    job_config = bigquery.LoadJobConfig()
#    job_config.autodetect = True  # Autodetectar esquema
#    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
#
#    # Cargar datos a BigQuery
#    job = client_bigquery.load_table_from_dataframe(df, table_ref, job_config=job_config)
#    job.result()  # Esperar a la carga
    
    print(f"Los vehiculos electricos has sido extraidos desde el bucket {bucket_name} y cargados a la tabla {table_id} de BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_electric_car --gen2 --region us-east1 --runtime python310 --trigger-bucket data_electric_car --entry-point process_electric_car --timeout 540s --memory 256MB

# gcloud functions deploy process_electric_car \
# --gen2 \
# --region=us-east1 \
# --runtime=python310 \
# --trigger-bucket=data_electric_car \
# --entry-point=process_electric_car \
# --timeout 540s \
# --memory 256MB


# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_uberLyft_G2 --gen2 --region us-east1 --runtime python310 --trigger-bucket data_alquiler_gran_volumen --entry-point process_uberLyft_G1 --timeout 540s --memory 16GB


# gcloud functions deploy process_uberLyft_G2 \
# --gen2 \
# --region=us-east1 \
# --runtime=python310 \
# --entry-point=process_uberLyft_G2 \
# --trigger-bucket=data_alquiler_gran_volumen \