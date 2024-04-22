import streamlit as st
import pandas as pd
import requests
import os
import tempfile
from google.cloud import storage
import openpyxl
import geopandas as gpd
import re

    
# Obtener la ruta absoluta del directorio actual y agregar el nombre del archivo de imagen____________________________:
GCP_imagen = os.path.join(os.path.dirname(__file__), "../Asset/GCP.jpg")
st.image(GCP_imagen)

# Definir las opciones para la lista desplegable______________________________________________________________________:

opciones = ['Emisiones de CO2 en NYC', 'Taxi_Zone','Electric_Stations']
# Mostrar el selectbox para elegir una opción
opcion_seleccionada = st.selectbox('Seleccione un archivo para subir a Google Cloud Platform:', opciones)


### SUBIR DATOS DE EMISIONES DE CO2___________________________________________________________________________________:
 
if opcion_seleccionada == 'Emisiones de CO2 en NYC':
    # Establecer la variable de entorno GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sturdy-gate-417001-1fe4b0dcfb9d.json"

    # Variables globales del proyecto
    project_id = 'sturdy-gate-417001'  # Proyecto
    bucket_name = 'data_co2_new_york'  # Bucket 

    # Función para convertir el archivo Excel a CSV
    def convert_to_csv(file_path):
        # Leer el archivo Excel
        df = pd.read_excel(file_path)
        
        # Guardar el DataFrame como archivo CSV
        csv_path = os.path.splitext(file_path)[0] + ".csv"
        #df.to_csv(csv_path, index=False,sep=';')
        df.to_csv(csv_path, index=False, sep=';', na_rep='NA', encoding='utf-8')
        
        return csv_path


    # Función para realizar el ETL
    def perform_etl(df):
        # Seleccionamos las columnas que vamos a utilizar
        columns_to_keep = ['Sectors Sector', 'Category Label','Source Label','Source Units'] + [col for col in df.columns if col.startswith('CY')]

        # Crear un nuevo DataFrame con las columnas seleccionadas
        df_normalized = df[columns_to_keep].copy()

        # Realizar la operación de fundido (melt)
        df_normalized = pd.melt(df_normalized, id_vars=['Sectors Sector', 'Category Label','Source Label','Source Units'], var_name='Year', value_name='Valor')

        # Seleccionamos las columnas que vamos a utilizar
        columns_to_keep = ['Sectors Sector', 'Category Label','Source Label','Source Units'] + [col for col in df.columns if col.startswith('CY')]

        # Crear un nuevo DataFrame con las columnas seleccionadas
        df_normalized = df[columns_to_keep].copy()

        # Realizar la operación de fundido (melt)
        df_normalized = pd.melt(df_normalized, id_vars=['Sectors Sector', 'Category Label','Source Label','Source Units'], var_name='Year', value_name='Valor')

        # Crear una columna auxiliar 'Tipo' y procesar la columna 'Year'
        df_normalized['Tipo'] = df_normalized['Year'].apply(lambda x: re.findall(r'(?:\d{4})\s(.+)', x)[0])
        df_normalized['Year'] = df_normalized['Year'].apply(lambda x: re.findall(r'(\d{4})', x)[0])

        # Filtrar solo las filas donde el tipo es 'Consumed' o 'tCO2e'
        df_normalized = df_normalized[df_normalized['Tipo'].isin(['Consumed', 'tCO2e','Source MMBtu'])]

        # Utilizar pivot_table para convertir las categorías de 'Tipo' en columnas
        df_pivot = df_normalized.pivot_table(index=['Year','Sectors Sector', 'Category Label', 'Source Label' ,'Source Units'],
                                            columns='Tipo', values='Valor', aggfunc='sum').reset_index()

        # Eliminar registros nulos
        df_pivot.dropna(inplace=True)

        # Eliminar registros duplicados
        df_pivot.drop_duplicates(inplace=True)

        nuevos_nombres = {
            'Year': 'year',
            'Sectors Sector': 'sectorsSector',
            'Category Full': 'categoryFull',
            'Category Label': 'categoryLabel',
            'Source Label': 'sourceLabel',
            'Source Units': 'sourceUnits',
            'Source MMBtu': 'SourceMMBtu',
            'tCO2e':'tCOe',
            'Consumed':'consumed'
        }

        # Renombrar las columnas utilizando el método rename()
        df_pivot = df_pivot.rename(columns=nuevos_nombres)

        # Lista de columnas que deseas convertir a tipo float
        columnas_float = ['consumed', 'SourceMMBtu', 'tCOe']

        # Convertir las columnas a tipo float
        df_pivot[columnas_float] = df_pivot[columnas_float].astype(float)     

        return df_pivot


    # Configurar la interfaz de usuario en Streamlit
    #st.title("Cargar Emisiones de CO2 NYC:")

    excel_url = st.text_input("Ingrese la URL de la fuente: (https://climate.cityofnewyork.us/initiatives/nyc-greenhouse-gas-inventories/):",value=str("https://drive.google.com/uc?id=1ZFhHtf64BgVZhzneOv1WhsyXXIX8ZyPn&export=download"))

    if st.button("Cargar desde URL"):
        try:
            # Realizar la solicitud para obtener el contenido del archivo Excel
            response = requests.get(excel_url)
            response.raise_for_status()
            
            # Guardar el contenido del archivo en un archivo temporal
            temp_excel_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            temp_excel_file.write(response.content)
            temp_excel_file.close()

            # Convertir el archivo Excel a CSV
            temp_csv_file = convert_to_csv(temp_excel_file.name)

            # Cargar el CSV en un DataFrame
            df = pd.read_csv(temp_csv_file, sep=';')

            # Realizar el ETL
            df_transformed = perform_etl(df)

            # Guardar el DataFrame transformado como un archivo CSV temporal
            temp_csv_transformed = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df_transformed.to_csv(temp_csv_transformed.name, sep=',',index=False, encoding='utf-8')
                   
            # Inicializar el cliente de Google Cloud Storage
            client_storage = storage.Client()
            bucket = client_storage.bucket(bucket_name)

            # Subir el archivo CSV al bucket en Google Cloud Storage
            blob_name ='CY_2005_CY_2022_citywide.csv'
            blob = bucket.blob(blob_name)
            #blob.upload_from_filename(temp_csv_file)
            blob.upload_from_filename(temp_csv_transformed.name, content_type='text/csv')
            
            st.success(f"El archivo {blob_name} ha sido cargado a Google Cloud Storage.")

            # Eliminar archivos temporales
            os.remove(temp_excel_file.name)
            os.remove(temp_csv_file)
 
            
        except Exception as e:
            st.error(f"Error al cargar el archivo desde la URL: {e}")

### SUBIR DATOS DE ZONAS DE TAXI___________________________________________________________________________________:

# Establecer la variable de entorno GOOGLE_APPLICATION_CREDENTIALS

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sturdy-gate-417001-1fe4b0dcfb9d.json"

# Función para cargar el archivo al bucket de Google Cloud Storage
def upload_to_gcs(file_path, bucket_name, blob_name):
    # Inicializar el cliente de Google Cloud Storage
    client = storage.Client()

    # Obtener el bucket
    bucket = client.get_bucket(bucket_name)

    # Subir el archivo al bucket
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)

    return f"El archivo {blob_name} ha sido cargado a Google Cloud Storage."

# Definir la transformación ETL para SHX
def etl_estaciones(file_path):
    # Cargar el archivo SHX usando GeoPandas
    data = gpd.read_file(file_path)

    # Definir EPSG:2263 que está en pies (LATEST WELL-KNOWN IDENTIFIER 2263)
    data.crs = 'EPSG:2263'

    # Corregir coordenadas planas a mundiales
    data_latlon = data.to_crs('EPSG:4326')

    # Calcular los centroides de los polígonos
    data_latlon['centroid'] = data_latlon['geometry'].centroid

    # Extraer las coordenadas x e y de los centroides
    data_latlon['longitud'] = data_latlon['centroid'].x
    data_latlon['latitud'] = data_latlon['centroid'].y

    # Seleccionar las columnas deseadas y renombrarlas
    data_latlon = data_latlon[['objectid', 'borough', 'zone', 'longitud', 'latitud']]
    
    # Eliminar el registro correspondiente al objectid 57
    data_latlon = data_latlon [data_latlon ['objectid'] != "57"]

    data_latlon = data_latlon [data_latlon ['objectid'] != "104"]
    data_latlon = data_latlon [data_latlon ['objectid'] != "105"]

    return data_latlon

# Interfaz de usuario de Streamlit

# Verificar si la opción seleccionada es "Taxi_Zone"
if opcion_seleccionada == "Taxi_Zone":
    # Obtener el archivo json desde el usuario
    file = st.file_uploader("Subir archivo taxi_zone.json (fuente: https://catalog.data.gov/dataset/nyc-taxi-zones)", key="file_uploader")

    # Procesar el archivo json si se ha cargado
    if file is not None:
        # Guardar el archivo json en el sistema temporalmente
        with open("temp_file.json", "wb") as f:
            f.write(file.getvalue())

        # Realizar la transformación ETL
        etl_data = etl_estaciones("temp_file.json")

        # Guardar el resultado de la transformación como un archivo CSV
        csv_filename = "Taxi_Zone.csv"
        etl_data.to_csv(csv_filename, index=False)

        # Nombre del archivo en el bucket
        blob_name = csv_filename

        # Nombre del bucket
        bucket_name = "data_taxi_zone"  # Reemplaza "data_taxi_zone" por el nombre de tu bucket

        # Subir el archivo CSV al bucket
        message = upload_to_gcs(csv_filename, bucket_name, blob_name)

        # Mostrar mensaje de éxito
        st.success(message)

        # Eliminar archivos temporales
        os.remove(csv_filename )
        os.remove("temp_file.json")


### SUBIR DATOS DE ESTACIONES DE SERVICIO___________________________________________________________________________________:

# Establecer la variable de entorno GOOGLE_APPLICATION_CREDENTIALS

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sturdy-gate-417001-1fe4b0dcfb9d.json"

# Función para cargar el archivo al bucket de Google Cloud Storage
def upload_to_gcs(file_path, bucket_name, blob_name):
    # Inicializar el cliente de Google Cloud Storage
    client = storage.Client()

    # Obtener el bucket
    bucket = client.get_bucket(bucket_name)

    # Subir el archivo al bucket
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)

    return f"El archivo {blob_name} ha sido cargado a Google Cloud Storage."

# Interfaz de usuario de Streamlit

# Verificar si la opción seleccionada es "Electric_Stations"
if opcion_seleccionada == "Electric_Stations":
    # Obtener el archivo csv desde el usuario
    file = st.file_uploader("Subir archivo alt_fuel_stations (fuente: https://afdc.energy.gov/data_download)", key="file_uploader")

    # Procesar el archivo csv si se ha cargado
    if file is not None:
        # Guardar el archivo csv en el sistema temporalmente
        with open("temp_file.csv", "wb") as f:
            f.write(file.getvalue())

        # Guardar el resultado de la transformación como un archivo CSV
        csv_filename = "temp_file.csv"
        
        # Nombre del archivo en el bucket
        blob_name = csv_filename

        # Nombre del bucket
        bucket_name = "data_stations" 

        # Subir el archivo CSV al bucket
        message = upload_to_gcs(csv_filename, bucket_name, blob_name)

        # Mostrar mensaje de éxito
        st.success(message)

        # Eliminar archivos temporales
        os.remove(csv_filename)