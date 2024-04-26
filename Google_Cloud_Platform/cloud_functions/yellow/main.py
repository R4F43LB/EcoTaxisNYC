from google.cloud import storage, bigquery
import pandas as pd
import numpy as np
import os
import tempfile

# Variables globales del proyecto
project_id = 'sturdy-gate-417001'  # Proyecto
bucket_name = 'data_taxis_amarillos'  # Bucket 
dataset_id = 'data_clean'  # DataSet
table_id = 'Trip'  # Nombre de la tabla

# Configuracion de credenciales para la conexion a Google Cloud Platform
# El archivo 'sturdy-gate-417001-19ab59ab9df1.json' que tiene las credenciales de la cuenta de
# servicio en GCP se descarga cuando se crea una nueva cuenta de servicio y se agrega una nueva clave.
#key_path = '../../sturdy-gate-417001-19ab59ab9df1.json'
#client = bigquery.Client.from_service_account_json(key_path)

def process_yellow(event, context):
    """Se activa cada vez que se sube un archivo al bucket 'data_taxis_amarillos'."""
    # Asegúrate de que esta línea refleje correctamente el nombre del archivo que deseas procesar
    file_name = event['name']

    client_storage = storage.Client()
    bucket = client_storage.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Ruta temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)
    
    # Leer el archivo parquet
    # Cargamos un dataframe con el dataset de taxis amarillos en New York
    yellow = pd.read_parquet(temp_file.name)

    # ETL
    # Eliminar comlumnas
    columnas_eliminar = ['VendorID', 'RatecodeID', 'store_and_fwd_flag', 'payment_type']
    yellow = yellow.drop(columns=columnas_eliminar)

    # Eliminar filas con tarifa base menor a $3
    # Eliminar las filas donde 'fare_amount' menor a 3 dólares (tarifa base taxis, $5 la de cancelación uber)
    yellow = yellow[yellow['fare_amount'] >= 3]

    # Imputar valores Errados
    # Calcula la moda de 'congestion_surcharge' y AIRPORT_FEE excluyendo los valores cero
    moda_congestion = yellow[yellow['congestion_surcharge'] != 0]['congestion_surcharge'].mode()[0]
    moda_Improvement = yellow[yellow['improvement_surcharge'] != 0]['improvement_surcharge'].mode()[0]
    
    # Imputa los valores erróneos utilizando la moda
    yellow['congestion_surcharge'] = yellow['congestion_surcharge'].apply(lambda x: moda_congestion if x > 0 else x)

    AIRPORT_FEE = None
    if 'Airport_fee' in yellow.columns:
        AIRPORT_FEE = 'Airport_fee'
    elif 'airport_fee' in yellow.columns:
        AIRPORT_FEE = 'airport_fee'

    moda_airport = yellow[yellow[AIRPORT_FEE] != 0][AIRPORT_FEE].mode()[0]
    yellow[AIRPORT_FEE] = yellow[AIRPORT_FEE].apply(lambda x: moda_airport if x > 0 else x)
    yellow[AIRPORT_FEE] = yellow['improvement_surcharge'].apply(lambda x: moda_Improvement if x > 0 else x)

    # Eliminación de Outliers
    # Eliminar las filas donde 'base_passenger_fare' supera los $300
    # No se considera real una tarifa mayor $150 y se permite que esta se duplique.
    yellow = yellow[yellow['fare_amount'] <= 300]

    # Eliminar las filas donde 'trip_miles' supera las 100 millas
    # la distancia máxima sería 50 millas, se puede suponer ida y vuelta
    yellow = yellow[yellow['trip_distance'] <= 100]

    # Eliminar las filas donde 'tolls' supera los $60 dólares
    yellow = yellow[yellow['tolls_amount'] <= 60]

    # Tranformar fechas y crear tiempos de espera y de viaje
    # 1. Crear la columna "time_out" que sea la diferencia entre tpep_pickup_datetime y request_datetime
    # En los taxis no hay registro, porque la mayoría son servicion recogidos por parada en la calle
    yellow['time_out'] = 0

    # 2. Crear la columna "travel_time" que sea la diferencia entre dropoff_datetime y tpep_pickup_datetime
    yellow['travel_time'] = yellow['tpep_dropoff_datetime'] - yellow['tpep_pickup_datetime']

    # 3. Convertir valores negativos en cero
    yellow['travel_time'] = yellow['travel_time'].clip(lower=pd.Timedelta(0))

    # 4. Crear las columnas "year", "month", "day", "hour" a partir de la columna tpep_pickup_datetime
    yellow['year'] = yellow['tpep_pickup_datetime'].dt.year
    yellow['month'] = yellow['tpep_pickup_datetime'].dt.month
    yellow['day'] = yellow['tpep_pickup_datetime'].dt.day
    yellow['hour'] = yellow['tpep_pickup_datetime'].dt.hour

    # 5. Redondear la columna "hour" al entero más cercano (de 1 a 24)
    yellow['hour'] = yellow['hour'].apply(lambda x: round(x))

    # 6. Crear la columna "day_of_week" a partir de la columna tpep_pickup_datetime
    yellow['day_of_week'] = yellow['tpep_pickup_datetime'].dt.day_name()

    # 7. Eliminar las columnas tpep_pickup_datetime y dropoff_datetime
    yellow = yellow.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    # Imputar tarifas de cargos según horario y unificación  de tarifa base y tarifa extra
    # Primero se imputan y corrigen valores de impuestos fijos según horarios
    # Definir una función para asignar los valores según la columna "hour"
    def asignar_valor(row):
        if (row['hour'] >= 16 and row['hour'] <= 24) or (row['hour'] <= 6):
            return 1.5 if (row['hour'] >= 16 and row['hour'] <= 20) else 0.5
        else:
            return 0

    # Aplicar la función a la columna extra
    yellow['extra'] = yellow.apply(asignar_valor, axis=1)

    # Luego se unifican tarifa base y extras
    # Crear la columna "fare_surcharges" que sea la suma de tolls, mta_tax, congestion_surcharge, airport_fee
    yellow['fare_surcharges'] = yellow['tolls_amount'] + yellow['extra'] + yellow['congestion_surcharge'] + yellow[AIRPORT_FEE]

    # Crear la columna "base_fare" que sea la suma de base_passenger_fare y sales_tax
    yellow['base_fare'] = yellow['fare_amount'] + 0.8 # 0.5 MTA + 0.3 Improvement

    # Eliminar las columnas de tarifas individuales
    columnas_eliminar = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
        'improvement_surcharge', 'total_amount', 'congestion_surcharge', AIRPORT_FEE]
    yellow = yellow.drop(columns=columnas_eliminar)

    # Cambiar formato de tiempos de viaje y espera
    # Convertir los datos de timedelta64[us] a minutos y luego a un entero
    yellow['travel_time_minutes'] = yellow['travel_time'] / pd.Timedelta(minutes=1)

    # Convertir a entero
    yellow['travel_time_minutes'] = yellow['travel_time_minutes'].astype(int)

    # Convertir a entero de 32 bits
    yellow['time_out'] = yellow['time_out'].astype('int32')
    yellow['travel_time'] = yellow['travel_time_minutes'].astype('int32')

    yellow = yellow.drop(columns=['travel_time_minutes'])

    # Eliminar los registros donde el tiempo de trayecto supera las 4 horas (240 minutos)
    yellow =yellow[yellow['travel_time'] <= 240]

    # Imputar fechas mal registradas
    # Servicios que superan 4 horas de trayecto
    moda_Año = yellow[yellow['year'] != 0]['year'].mode()[0]
    moda_Mes = yellow[yellow['month'] != 0]['month'].mode()[0]

    yellow['year'] = yellow['year'].apply(lambda x: moda_Año if x > 0 else x)
    yellow['month'] = yellow['month'].apply(lambda x: moda_Mes if x > 0 else x)

    # Crear columnas auxiliares para identificar nuevos outliers Costo por minuto y Costo por milla
    # Filtrar los registros donde trip_distance es cero y asignar un valor NaN
    yellow['$mile'] = np.where(yellow['trip_distance'] != 0, yellow['base_fare'] / yellow['trip_distance'], np.nan)

    # Filtrar los registros donde trip_minutes es cero y asignar un valor NaN
    yellow['$minute'] = np.where(yellow['travel_time'] != 0, yellow['base_fare'] / yellow['travel_time'], np.nan)

    # Eliminar Outliers basado en rango intercuartílico
    # Calcular los cuartiles
    Q1_mile = yellow['$mile'].quantile(0.25)
    Q3_mile = yellow['$mile'].quantile(0.75)
    IQR_mile = Q3_mile - Q1_mile

    Q1_minute = yellow['$minute'].quantile(0.25)
    Q3_minute = yellow['$minute'].quantile(0.75)
    IQR_minute = Q3_minute - Q1_minute

    # Multiplicador para rango intercuartílico:
    multiplicador = 2

    # Definir los límites superior e inferior
    umbral_inf_mile = Q1_mile - multiplicador * IQR_mile
    umbral_sup_mile = Q3_mile + multiplicador * IQR_mile

    umbral_inf_minute =  Q3_minute - multiplicador * IQR_minute # para evitar borrar todos los registros
    umbral_sup_minute = Q3_minute + multiplicador * IQR_minute

    # Filtrar los registros que caen dentro del rango intercuartílico para cada columna
    yellow = yellow[
        (yellow['$mile'] >= umbral_inf_mile) & (yellow['$mile'] <= umbral_sup_mile) &
        (yellow['$minute'] >= umbral_inf_minute) & (yellow['$minute'] <= umbral_sup_minute)
    ]

    # Eliminar datos con ubicaciones no definidas
    # Eliminar los registros con DOLocationID igual a 264
    yellow = yellow.loc[yellow['DOLocationID'] != 264]

    # Eliminar los registros con DOLocationID igual a 265
    yellow = yellow.loc[yellow['DOLocationID'] != 265]

    # Eliminar los registros con PULocationID igual a 264
    yellow = yellow.loc[yellow['PULocationID'] != 264]

    # Eliminar los registros con PULocationID igual a 265
    yellow = yellow.loc[yellow['PULocationID'] != 265]

    # Imputar registros unificando los polígonos de los ID 55+56 y 103+104+105
    # Imputar los valores según las condiciones dadas
    yellow.replace(
        {'PULocationID': {57: 56, 105: 103, 104: 103}, 'DOLocationID': {57: 56, 105: 103,  104: 103}},
        inplace=True
    )

    # Realizar agregación de yellow
    # Crear la columna "service_number" y asignar el valor 1 a todas las filas
    yellow['service_number'] = 1

    # Crear la columna "service_Type" y asignar el valor yellow a todas las filas
    yellow['service_type'] = 'yellow'

    # Renombrar columnas (normalizadas a UberLyft) 
    yellow.rename(columns={'trip_distance': 'trip_miles'}, inplace=True)

    # Definir las dimensiones de agrupación y las variables de agregación
    dimensiones = ['service_type','year', 'month', 'day', 'day_of_week','hour', 'PULocationID', 'DOLocationID']
    variables_agregacion = ['trip_miles', 'time_out', 'travel_time', 'fare_surcharges', 'base_fare', 'service_number']

    # Agrupar el DataFrame y calcular la suma de las variables de agregación
    yellow = yellow.groupby(dimensiones)[variables_agregacion].sum().reset_index()


    # Inicializar cliente de BigQuery y configurar la carga
    client_bigquery = bigquery.Client()
    table_ref = client_bigquery.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Autodetectar esquema
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

    # Cargar datos a BigQuery
    job = client_bigquery.load_table_from_dataframe(yellow, table_ref, job_config=job_config)
    job.result()  # Esperar a la carga

    # Limpiar archivo temporal
    temp_file.close()
    os.remove(temp_file.name)

    print(f"El archivo {file_name} ha sido transformado y cargado a BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_yellow --runtime python39 --trigger-bucket data_taxis_amarillos --entry-point process_yellow --timeout 540s --memory 8GB
