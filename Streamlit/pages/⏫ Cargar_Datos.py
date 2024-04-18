import streamlit as st
import pandas as pd
import requests
import os
import tempfile
from google.cloud import storage
import openpyxl

    
# Obtener la ruta absoluta del directorio actual y agregar el nombre del archivo de imagen
GCP_imagen = os.path.join(os.path.dirname(__file__), "../GCP.jpeg")

st.image(GCP_imagen)
    

st.markdown('### Elija un Archivo para Cargar a Google Cloud:')

if st.checkbox('Emisiones de CO2 en NYC'):
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
        df.to_csv(csv_path, index=False)
        
        return csv_path

    # Configurar la interfaz de usuario en Streamlit
    st.title("Cargar Emisiones de CO2 NYC:")

    excel_url = st.text_input("Ingrese la URL del archivo Excel:")
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

            # Inicializar el cliente de Google Cloud Storage
            client_storage = storage.Client()
            bucket = client_storage.bucket(bucket_name)

            # Subir el archivo CSV al bucket en Google Cloud Storage
            blob = bucket.blob(os.path.basename(temp_csv_file))
            blob.upload_from_filename(temp_csv_file)

            st.success(f"El archivo {os.path.basename(temp_csv_file)} ha sido cargado a Google Cloud Storage.")

            # Eliminar archivos temporales
            os.remove(temp_excel_file.name)
            os.remove(temp_csv_file)
            
        except Exception as e:
            st.error(f"Error al cargar el archivo desde la URL: {e}")

st.checkbox('Electric Car Data')