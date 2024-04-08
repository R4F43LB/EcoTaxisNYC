from shapely.geometry import Polygon, MultiPolygon, shape    # Simplificar Polígonos
from google.cloud import storage, bigquery
from shapely.ops import unary_union                          # Combinar polígonos
import geopandas as gpd                                      # leer los archivos .shx y .shp
import tempfile
import os

# Variables globales del proyecto
project_id = 'sturdy-gate-417001'  # Proyecto
bucket_name = 'data_taxi_zone'  # Bucket 
dataset_id = 'data_clean'  # DataSet
table_id = 'TaxiZone'  # Nombre de la tabla

def process_taxi_zone(event, context):
    """Se activa cada vez que se sube un archivo al bucket 'data_taxi_zone'."""
    # Asegúrate de que esta linea refleje correctamente el nombre del archivo que deseas procesar
    #file_name = event['name']
    file_name = 'taxi_zones.shx'

    client_storage = storage.Client()
    bucket = client_storage.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Ruta temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)

    # Leer el archivo .shx
    # Para poder leer el archivo taxi_zones.shx en la misma carpeta debe estar el arcivo taxi_zones.shp
    # Cargamos un dataframe con el dataset de sonidos en New York
    data = gpd.read_file('temp_file.name')

    # ETL
    # Transformar coordenadas planas ATEST WELL-KNOWN IDENTIFIER 2263 a la proyección inversa
    # mundial (EPSG:4326):
    gdf = gpd.GeoDataFrame(data)

    # Definir EPSG:2263 que está en pies (LATEST WELL-KNOWN IDENTIFIER 2263):
    gdf.crs = 'EPSG:2263'
    
    # Corregir coordenadas planas a mundiales:
    gdf_latlon = gdf.to_crs('EPSG:4326')  # Proyección inversa al sistema de coordenadas geográficas
    
    # Calcular los centroides de los polígonos
    gdf_latlon['centroid'] = gdf_latlon['geometry'].centroid
    
    # Extraer las coordenadas x e y de los centroides
    gdf_latlon['centroid_x'] = gdf_latlon['centroid'].x
    gdf_latlon['centroid_y'] = gdf_latlon['centroid'].y

    # Seleccionar las columnas deseadas
    gdf_latlon = gdf_latlon[['OBJECTID', 'borough', 'zone', 'centroid_x', 'centroid_y','geometry']]
    
    # Renombrar las columnas centroid_x y centroid_y a longitud y latitud respectivamente
    gdf_latlon = gdf_latlon.rename(
        columns={
            'OBJECTID': 'locationId',
            'centroid_x': 'longitud',
            'centroid_y': 'latitud'
        }
    )

    # Definir la tolerancia para la simplificación
    tolerancia = 0.0001  # ajusta este valor según sea necesario
    
    # Función para simplificar una geometría individual
    def simplificar_geometria(geom, tolerancia):
        return geom.simplify(tolerancia)
    
    # Aplicar la función de simplificación a cada geometría en el GeoDataFrame
    gdf_latlon['geometry'] = gdf_latlon['geometry'].apply(lambda geom: simplificar_geometria(geom, tolerancia))

    # Seleccionar los polígonos correspondientes a los LocationID 56 y 57
    poligonos_56_57 = gdf_latlon.loc[gdf_latlon['LocationID'].isin([56, 57]), 'geometry']
    
    # Crear un MultiPolygon combinando los polígonos
    multi_poligono_56_57 = unary_union(poligonos_56_57)
    
    # Reemplazar el polígono del LocationID 56 con el MultiPolygon combinado
    gdf_latlon.loc[gdf_latlon['LocationID'] == 56, 'geometry'] = multi_poligono_56_57
    
    # Eliminar el registro correspondiente al LocationID 57
    gdf_latlon= gdf_latlon[gdf_latlon['LocationID'] != 57]

    # Seleccionar los polígonos correspondientes a los LocationID 103, 104 y 105
    poligonos_103_104_105 = gdf_latlon.loc[gdf_latlon['LocationID'].isin([103, 104, 105]), 'geometry']
    
    # Crear un MultiPolygon combinando los polígonos
    multi_poligono_103_104_105 = unary_union(poligonos_103_104_105)
    
    # Reemplazar el polígono del LocationID 103 con el MultiPolygon combinado
    gdf_latlon.loc[gdf_latlon['LocationID'] == 103, 'geometry'] = multi_poligono_103_104_105
    
    # Eliminar el registro correspondiente al LocationID 104 y 105
    gdf_latlon = gdf_latlon[gdf_latlon['LocationID'] != 104]
    gdf_latlon = gdf_latlon[gdf_latlon['LocationID'] != 105]

    # Inicializar cliente de BigQuery y configurar la carga
    client_bigquery = bigquery.Client()
    table_ref = client_bigquery.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True  # Autodetectar esquema
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

    # Cargar datos a BigQuery
    job = client_bigquery.load_table_from_dataframe(gdf_latlon, table_ref, job_config=job_config)
    job.result()  # Esperar a la carga

    # Limpiar archivo temporal
    temp_file.close()
    os.remove(temp_file.name)
    
    print(f"El archivo {file_name} ha sido procesado y cargado a BigQuery.")

# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_taxi_zone --runtime python39 --trigger-bucket data_taxi_zone --entry-point process_taxi_zone --timeout 540s --memory 256MB
