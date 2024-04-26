from google.cloud import storage, bigquery
import pandas as pd
import numpy as np
import os
import tempfile

# Variables globales del proyecto
project_id = 'sturdy-gate-417001'  # Proyecto
bucket_name = 'data_taxis_verdes'  # Bucket 
dataset_id = 'data_clean'  # DataSet
table_id = 'Trip'  # Nombre de la tabla

# Configuracion de credenciales para la conexion a Google Cloud Platform
# El archivo 'sturdy-gate-417001-19ab59ab9df1.json' que tiene las credenciales de la cuenta de
# servicio en GCP se descarga cuando se crea una nueva cuenta de servicio y se agrega una nueva clave.
#key_path = '../../sturdy-gate-417001-19ab59ab9df1.json'
#client = bigquery.Client.from_service_account_json(key_path)

def process_green(event, context):
    """Se activa cada vez que se sube un archivo al bucket 'data_taxis_verdes'."""
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
    green = pd.read_parquet(temp_file.name)

    # ETL
    # Eliminar columnas innecesarias
    columnas_eliminar = ['VendorID', 'RatecodeID', 'store_and_fwd_flag',
           'payment_type','trip_type']

    green = green.drop(columns=columnas_eliminar)

    # Convertir números negativos a positivos
    # Crear una lista de las columnas en las que deseas convertir los valores negativos a positivos
    columnas_a_convertir = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
        'improvement_surcharge', 'total_amount', 'congestion_surcharge']

    # Aplicar una función lambda a cada valor en cada una de esas columnas para cambiar los valores
    # negativos a positivos
    green[columnas_a_convertir] = green[columnas_a_convertir].applymap(lambda x: abs(x) if x < 0 else x)

    # Eliminar filas con tarifas menores a $3
    # Eliminar las filas donde 'fare_amount' menor a 3 dólares (tarifa base taxis, $5 la de cancelación uber)
    green = green[green['fare_amount'] >= 3]

    # Imputar valores errados de recargos fijos o conidicionales al horario
    # Calcula la moda de 'congestion_surcharge' y 'Airport_fee' excluyendo los valores cero
    moda_congestion = green[green['congestion_surcharge'] != 0]['congestion_surcharge'].mode()[0]
    moda_Improvement = green[green['improvement_surcharge'] != 0]['improvement_surcharge'].mode()[0]

    # Imputa los valores erróneos utilizando la moda
    green['congestion_surcharge'] = green['congestion_surcharge'].apply(lambda x: moda_congestion if x > 0 else x)
    green['Airport_fee'] = green['improvement_surcharge'].apply(lambda x: moda_Improvement if x > 0 else x)

    # Elimnación de Outliers
    # Eliminar las filas donde 'base_passenger_fare' supera los $400
    green = green[green['fare_amount'] <= 300] # No se considera real una tarifa mayor $150 y se permite que esta se duplique.

    # Eliminar las filas donde 'trip_miles' supera las 100 millas
    green = green[green['trip_distance'] <= 100] # la distancia máxima sería 50 millas, se puede suponer ida y vuelta

    # Eliminar las filas donde 'tolls' supera los $60 dólares
    green = green[green['tolls_amount'] <= 60]

    # Transformar datos de fecha e incluir tiempos de viaje y espera
    # 1. Crear la columna "time_out" que sea la diferencia entre lpep_pickup_datetime y request_datetime
    green['time_out'] = 0 #En los taxis no hay registro, porque la mayoría son servicion recogidos por parada en la calle

    # 2. Crear la columna "travel_time" que sea la diferencia entre dropoff_datetime y lpep_pickup_datetime
    green['travel_time'] = green['lpep_dropoff_datetime'] - green['lpep_pickup_datetime']

    # 3. Convertir valores negativos en cero
    green['travel_time'] = green['travel_time'].clip(lower=pd.Timedelta(0))

    # 4. Crear las columnas "year", "month", "day", "hour" a partir de la columna lpep_pickup_datetime
    green['year'] = green['lpep_pickup_datetime'].dt.year
    green['month'] = green['lpep_pickup_datetime'].dt.month
    green['day'] = green['lpep_pickup_datetime'].dt.day
    green['hour'] = green['lpep_pickup_datetime'].dt.hour

    # 5. Redondear la columna "hour" al entero más cercano (de 1 a 24)
    green['hour'] = green['hour'].apply(lambda x: round(x))

    # 6. Crear la columna "day_of_week" a partir de la columna lpep_pickup_datetime
    green['day_of_week'] = green['lpep_pickup_datetime'].dt.day_name()

    # 7. Eliminar las columnas lpep_pickup_datetime y dropoff_datetime
    green = green.drop(columns=['lpep_pickup_datetime', 'lpep_dropoff_datetime'])

    # Transformar tarifas y unificar tarifas base y tarifas extra
    # Definir una función para asignar los valores según la columna "hour"
    def asignar_valor(row):
        if (row['hour'] >= 16 and row['hour'] <= 24) or (row['hour'] <= 6):
            return 1.5 if (row['hour'] >= 16 and row['hour'] <= 20) else 0.5
        else:
            return 0

    # Aplicar la función a la columna extra
    green['extra'] = green.apply(asignar_valor, axis=1)

    # Unificar tarifas
    # Crear la columna "fare_surcharges" que sea la suma de tolls, mta_tax, congestion_surcharge, airport_fee
    green['fare_surcharges'] = green['tolls_amount'] + green['extra'] + green['congestion_surcharge'] + green['Airport_fee']

    # Crear la columna "base_fare" que sea la suma de base_passenger_fare y sales_tax
    green['base_fare'] = green['fare_amount'] + 0.8 # 0.5 MTA + 0.3 Improvement

    # Eliminar las columnas de tarifas individuales
    columnas_eliminar = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
           'improvement_surcharge', 'total_amount', 'congestion_surcharge']
    green = green.drop(columns=columnas_eliminar)

    # Cambiar formato de tiempos a enteros
    # Convertir los datos de timedelta64[us] a minutos y luego a un entero
    green['travel_time_minutes'] = green['travel_time'] / pd.Timedelta(minutes=1)

    # Convertir a entero
    green['travel_time_minutes'] = green['travel_time_minutes'].astype(int)

    # Convertir a entero de 32 bits
    green['time_out'] = green['time_out'].astype('int32')
    green['travel_time'] = green['travel_time_minutes'].astype('int32')

    green = green.drop(columns=['travel_time_minutes'])

    # Eliminar registros con tiempo de viaje mayor a 240 minutos
    # Eliminar los registros donde el tiempo de trayecto supera las 4 horas (240 minutos)
    green =green[green['travel_time'] <= 240]

    # Imputar fechas mal registradas de año y mes
    # Servicios que superan 4 horas de trayecto
    moda_Año = green[green['year'] != 0]['year'].mode()[0]
    moda_Mes = green[green['month'] != 0]['month'].mode()[0]

    green['year'] = green['year'].apply(lambda x: moda_Año if x > 0 else x)
    green['month'] = green['month'].apply(lambda x: moda_Mes if x > 0 else x)

    # Agregar columnas auxiliares para definir nuevos outliers
    # Filtrar los registros donde trip_distance es cero y asignar un valor NaN
    green['$mile'] = np.where(green['trip_distance'] != 0, green['base_fare'] / green['trip_distance'], np.nan)

    # Filtrar los registros donde trip_minutes es cero y asignar un valor NaN
    green['$minute'] = np.where(green['travel_time'] != 0, green['base_fare'] / green['travel_time'], np.nan)

    # Eliminar outliers mediante rangos intercuartílicos
    # Calcular los cuartiles
    Q1_mile = green['$mile'].quantile(0.25)
    Q3_mile = green['$mile'].quantile(0.75)
    IQR_mile = Q3_mile - Q1_mile

    Q1_minute = green['$minute'].quantile(0.25)
    Q3_minute = green['$minute'].quantile(0.75)
    IQR_minute = Q3_minute - Q1_minute

    # Multiplicador para rango intercuartílico:
    multiplicador = 2

    # Definir los límites superior e inferior
    umbral_inf_mile = Q1_mile - multiplicador * IQR_mile
    umbral_sup_mile = Q3_mile + multiplicador * IQR_mile

    umbral_inf_minute =  Q3_minute - multiplicador * IQR_minute # para evitar borrar todos los registros
    umbral_sup_minute = Q3_minute + multiplicador * IQR_minute

    # Filtrar los registros que caen dentro del rango intercuartílico para cada columna
    green = green[
        (green['$mile'] >= umbral_inf_mile) & (green['$mile'] <= umbral_sup_mile) &
        (green['$minute'] >= umbral_inf_minute) & (green['$minute'] <= umbral_sup_minute)
    ]

    # Eliminar registros con Ubicaciones no definidas
    # Eliminar los registros con DOLocationID igual a 264
    green = green.loc[green['DOLocationID'] != 264]

    # Eliminar los registros con DOLocationID igual a 265
    green = green.loc[green['DOLocationID'] != 265]

    # Eliminar los registros con PULocationID igual a 264
    green = green.loc[green['PULocationID'] != 264]

    # Eliminar los registros con PULocationID igual a 265
    green = green.loc[green['PULocationID'] != 265]

    # Imputar registros unificando los polígonos de los ID 55+56 y 103+104+105

    # Imputar los valores según las condiciones dadas
    green.replace({
        'PULocationID': {57: 56, 105: 103, 104: 103},
        'DOLocationID': {57: 56, 105: 103,  104: 103}},
        inplace=True
    )

    # Realizar agregaciones
    # Crear la columna "service_number" y asignar el valor 1 a todas las filas
    green['service_number'] = 1
    
    # Crear la columna "service_Type" y asignar el valor green a todas las filas
    green['service_type'] = 'green'
     
    green.rename(columns={'trip_distance': 'trip_miles'}, inplace=True)
    
    # Definir las dimensiones de agrupación y las variables de agregación
    dimensiones = ['service_type','year', 'month', 'day', 'day_of_week','hour', 'PULocationID', 'DOLocationID']
    variables_agregacion = ['trip_miles', 'time_out', 'travel_time', 'fare_surcharges', 'base_fare', 'service_number']
    
    # Agrupar el DataFrame y calcular la suma de las variables de agregación
    green = green.groupby(dimensiones)[variables_agregacion].sum().reset_index()

    # Inicializar cliente de BigQuery y configurar la carga
    client_bigquery = bigquery.Client()
    table_ref = client_bigquery.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Autodetectar esquema
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

    # Cargar datos a BigQuery
    job = client_bigquery.load_table_from_dataframe(green, table_ref, job_config=job_config)
    job.result()  # Esperar a la carga

    # Limpiar archivo temporal
    temp_file.close()
    os.remove(temp_file.name)

    print(f"El archivo {file_name} ha sido transformado y cargado a la tabla {table_id} de BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_green --runtime python39 --trigger-bucket data_taxis_verdes --entry-point process_green --timeout 540s --memory 8GB
