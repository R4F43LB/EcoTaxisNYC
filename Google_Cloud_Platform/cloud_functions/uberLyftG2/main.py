import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from google.cloud import storage
from google.cloud import bigquery
import functions_framework

# Variables globales del proyecto
dataset_id = 'data_clean'  # DataSet
table_id = 'Trip7'  # Nombre de la tabla

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def hello_gcs(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    local_file_path = f'/tmp/{file_name}'
    ###############
    ### EXTRACT ###
    ###############
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download the Parquet file to a local file in the /tmp directory
    blob.download_to_filename(local_file_path)

    # Read the local Parquet file using pyarrow
    table = pq.read_table(local_file_path)
    uberLyft = table.to_pandas()

    # Clean up
    blob.delete()

    #################
    ### TRANSFORM ###
    #################
    # Eliminar columnas innecesarias
    columnas_eliminar = ['hvfhs_license_num', 'dispatching_base_num', 'originating_base_num',
           'on_scene_datetime','trip_time', 'driver_pay',
           'shared_request_flag', 'shared_match_flag', 'access_a_ride_flag',
           'wav_request_flag', 'wav_match_flag','tips']

    uberLyft = uberLyft.drop(columns=columnas_eliminar)

    # Convertir números negativos a positivos
    # Crear una lista de las columnas en las que deseas convertir los valores negativos a positivos
    columnas_a_convertir = ['base_passenger_fare', 'tolls', 'bcf', 'sales_tax', 'congestion_surcharge', 'airport_fee']

    # Aplicar una función lambda a cada valor en cada una de esas columnas para cambiar los valores negativos a positivos
    uberLyft[columnas_a_convertir] = uberLyft[columnas_a_convertir].applymap(lambda x: abs(x) if x < 0 else x)    

    # Eliminar las filas donde 'base_passenger_fare' menor a 3 dólares (tarifa base taxis, $5 la de cancelación uber)
    uberLyft = uberLyft[uberLyft['base_passenger_fare'] >= 3]

    # Imputar valores errados de impuestos fijos o condicionados a hora o lugar
    # Calcula la moda de 'congestion_surcharge' y 'airport_fee' excluyendo los valores cero
    moda_congestion = uberLyft[uberLyft['congestion_surcharge'] != 0]['congestion_surcharge'].mode()[0]
    moda_airport = uberLyft[uberLyft['airport_fee'] != 0]['airport_fee'].mode()[0]

    # Imputa los valores erróneos utilizando la moda
    uberLyft['congestion_surcharge'] = uberLyft['congestion_surcharge'].apply(lambda x: moda_congestion if x > 0 else x)
    uberLyft['airport_fee'] = uberLyft['airport_fee'].apply(lambda x: moda_airport if x > 0 else x)

    # Corregir sales_tax al 8.875% de acuerdo con [el siguiente link]
    # (https://www.tax.ny.gov/pubs_and_bulls/tg_bulletins/st/translated/sales-tax-rates-spa.htm)
    # Filtra los valores no cero en 'sales_tax' y 'base_passenger_fare'
    filtered_data = uberLyft[(uberLyft['sales_tax'] != 0) & (uberLyft['base_passenger_fare'] != 0)]

    # Calcula la relación entre 'sales_tax' y 'base_passenger_fare' y el promedio redondeado a 2
    # cifras decimales
    promedio_relacion = filtered_data['sales_tax'] / filtered_data['base_passenger_fare']
    promedio_redondeado = round(promedio_relacion.mean(), 5)

    # Imputar los valores en cero de sales_tax
    uberLyft['sales_tax'] = uberLyft['sales_tax']* promedio_redondeado

    # Eliminar outliers
    # Eliminar las filas donde 'base_passenger_fare' supera los $400
    # No se considera real una tarifa mayor $150 y se permite que esta se duplique.
    uberLyft = uberLyft[uberLyft['base_passenger_fare'] <= 300]

    # Eliminar las filas donde 'trip_miles' supera las 100 millas
    # la distancia máxima sería 50 millas, se puede suponer ida y vuelta
    uberLyft = uberLyft[uberLyft['trip_miles'] <= 100] 

    # Eliminar las filas donde 'tolls' supera los $60 dólares
    uberLyft = uberLyft[uberLyft['tolls'] <= 60]

    # Transformar datos de fecha y adicionar columnas de tiempos de espera y de viaje
    # 1. Crear la columna "time_out" que sea la diferencia entre pickup_datetime y request_datetime
    uberLyft['time_out'] = uberLyft['pickup_datetime'] - uberLyft['request_datetime']

    # 2. Crear la columna "travel_time" que sea la diferencia entre dropoff_datetime y pickup_datetime
    uberLyft['travel_time'] = uberLyft['dropoff_datetime'] - uberLyft['pickup_datetime']

    # 3. Convertir valores negativos en cero
    uberLyft['travel_time'] = uberLyft['travel_time'].clip(lower=pd.Timedelta(0))
    uberLyft['time_out'] = uberLyft['time_out'].clip(lower=pd.Timedelta(0))

    # 4. Crear las columnas "year", "month", "day", "hour" a partir de la columna pickup_datetime
    uberLyft['year'] = uberLyft['pickup_datetime'].dt.year
    uberLyft['month'] = uberLyft['pickup_datetime'].dt.month
    uberLyft['day'] = uberLyft['pickup_datetime'].dt.day
    uberLyft['hour'] = uberLyft['pickup_datetime'].dt.hour

    # 5. Redondear la columna "hour" al entero más cercano (de 1 a 24)
    uberLyft['hour'] = uberLyft['hour'].apply(lambda x: round(x))

    # 6. Crear la columna "day_of_week" a partir de la columna pickup_datetime
    uberLyft['day_of_week'] = uberLyft['pickup_datetime'].dt.day_name()

    # 7. Eliminar las columnas request_datetime, pickup_datetime, dropoff_datetime
    uberLyft = uberLyft.drop(columns=['request_datetime', 'pickup_datetime', 'dropoff_datetime'])

    # Transformar datos de tarifas base y recargos
    # Crear la columna "fare_surcharges" que sea la suma de tolls, bcf, congestion_surcharge, airport_fee
    uberLyft['fare_surcharges'] = uberLyft['tolls'] + uberLyft['bcf'] + uberLyft['congestion_surcharge'] + uberLyft['airport_fee']

    # Crear la columna "base_fare" que sea la suma de base_passenger_fare y sales_tax
    uberLyft['base_fare'] = uberLyft['base_passenger_fare'] + uberLyft['sales_tax']

    # Eliminar las columnas tolls, bcf, congestion_surcharge, airport_fee, base_passenger_fare y sales_tax
    columnas_eliminar = ['tolls', 'bcf', 'congestion_surcharge', 'airport_fee', 'base_passenger_fare', 'sales_tax']
    uberLyft = uberLyft.drop(columns=columnas_eliminar)

    # Cambiar formato de tiempos de viaje y espera
    # Convertir los datos de timedelta64[us] a minutos y luego a un entero
    uberLyft['time_out_minutes'] = uberLyft['time_out'] / pd.Timedelta(minutes=1)
    uberLyft['travel_time_minutes'] = uberLyft['travel_time'] / pd.Timedelta(minutes=1)

    # Convertir a entero
    uberLyft['time_out_minutes'] = uberLyft['time_out_minutes'].astype(int)
    uberLyft['travel_time_minutes'] = uberLyft['travel_time_minutes'].astype(int)

    # Convertir a entero de 32 bits
    uberLyft['time_out'] = uberLyft['time_out_minutes'].astype('int32')
    uberLyft['travel_time'] = uberLyft['travel_time_minutes'].astype('int32')

    uberLyft = uberLyft.drop(columns=['time_out_minutes', 'travel_time_minutes'])

    # Eliminar outliers de tiempos de viaje y espera
    # Eliminar los registros donde el tiempo de trayecto supera las 4 horas (240 minutos)
    uberLyft = uberLyft[uberLyft['travel_time'] <= 240]

    # Eliminar los registros donde el tiempo de espera es mayor a 2 horas (120 minutos)
    uberLyft = uberLyft[uberLyft['time_out'] <= 120]

    # Eliminar los registros donde el tiempo de espera es menor a 1 hora (-60 minutos)
    uberLyft = uberLyft[uberLyft['time_out'] >= -60]

    # Crear columnas auxiliares de costo por minuto y costo por milla para identificar nuevos outliers
    # Filtrar los registros donde trip_miles es cero y asignar un valor NaN
    uberLyft['$mile'] = np.where(uberLyft['trip_miles'] != 0, uberLyft['base_fare'] / uberLyft['trip_miles'], np.nan)

    # Filtrar los registros donde trip_minutes es cero y asignar un valor NaN
    uberLyft['$minute'] = np.where(uberLyft['travel_time'] != 0, uberLyft['base_fare'] / uberLyft['travel_time'], np.nan)

    # Eliminar los Outliers basado en los rangos intercuartílicos
    # Se decide evitar el sesgo de costos que se mantienen muy lejos de la relación Tarifa por milla
    # o Tarifa por minuto así que se eliminan los registros que estén por encima de este cálculo. No
    # se usa 3 sigma debido a que el sesgo de datos atípicos es muy alto. También se decide eliminar
    # registros que superan los 250 dólares por no representar la realidad de un servicio normal
    # dentro de la ciudad de NYC (se puede suponer hasta normal $60, probable $100 y muy poco
    # probable $250 o más)
    # Calcular los cuartiles
    Q1_mile = uberLyft['$mile'].quantile(0.25)
    Q3_mile = uberLyft['$mile'].quantile(0.75)
    IQR_mile = Q3_mile - Q1_mile

    Q1_minute = uberLyft['$minute'].quantile(0.25)
    Q3_minute = uberLyft['$minute'].quantile(0.75)
    IQR_minute = Q3_minute - Q1_minute

    # Multiplicador para rango intercuartílico:
    multiplicador = 2

    # Definir los límites superior e inferior
    umbral_inf_mile = Q1_mile - multiplicador * IQR_mile
    umbral_sup_mile = Q3_mile + multiplicador * IQR_mile

    umbral_inf_minute = Q1_minute - multiplicador * IQR_minute
    umbral_sup_minute = Q3_minute + multiplicador * IQR_minute

    # Filtrar los registros que caen dentro del rango intercuartílico para cada columna
    uberLyft = uberLyft[
        (uberLyft['$mile'] >= umbral_inf_mile) & (uberLyft['$mile'] <= umbral_sup_mile) &
        (uberLyft['$minute'] >= umbral_inf_minute) & (uberLyft['$minute'] <= umbral_sup_minute)
    ]

    # Eliminar registros de ubicaciones no definidas
    # Eliminar los registros con DOLocationID igual a 264
    uberLyft = uberLyft.loc[uberLyft['DOLocationID'] != 264]

    # Eliminar los registros con DOLocationID igual a 265
    uberLyft = uberLyft.loc[uberLyft['DOLocationID'] != 265]

    # Eliminar los registros con PULocationID igual a 264
    uberLyft = uberLyft.loc[uberLyft['PULocationID'] != 264]

    # Eliminar los registros con PULocationID igual a 265
    uberLyft = uberLyft.loc[uberLyft['PULocationID'] != 265]

    # Imputar registros unificando los polígonos de los ID 55+56 y 103+104+105
    # Imputar los valores según las condiciones dadas
    uberLyft.replace({
        'PULocationID': {57: 56, 105: 103, 104: 103},
        'DOLocationID': {57: 56, 105: 103,  104: 103}},
        inplace=True
    )

    # Realizar agregacion de uberLyft
    # Crear la columna "service_number" y asignar el valor 1 a todas las filas
    uberLyft['service_number'] = 1
    
    # Crear la columna "service_Type" y asignar el valor uberLyft a todas las filas
    uberLyft['service_type'] = 'uberLyft'
    
    # Definir las dimensiones de agrupación y las variables de agregación
    dimensiones = ['service_type','year', 'month', 'day', 'day_of_week','hour', 'PULocationID', 'DOLocationID']
    variables_agregacion = ['trip_miles', 'time_out', 'travel_time', 'fare_surcharges', 'base_fare', 'service_number']
    
    # Agrupar el DataFrame y calcular la suma de las variables de agregación
    uberLyft = uberLyft.groupby(dimensiones)[variables_agregacion].sum().reset_index()

    ############
    ### Load ###
    ############
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    client.load_table_from_dataframe(uberLyft, table_ref).result()

    print(f"El archivo {file_name} ha sido transformado y cargado a la tabla {table_id} de BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_uberLyft_G2 --gen2 --region us-east1 --runtime python310 --trigger-bucket data_alquiler_gran_volumen --entry-point process_uberLyft_G1 --timeout 540s --memory 16GB


# gcloud functions deploy process_uberLyft_G2 \
# --gen2 \
# --region=us-east1 \
# --runtime=python310 \
# --entry-point=process_uberLyft_G2 \
# --trigger-bucket=data_alquiler_gran_volumen \