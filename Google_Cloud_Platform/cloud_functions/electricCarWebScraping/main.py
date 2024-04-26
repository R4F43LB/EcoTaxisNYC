from google.cloud import storage
from bs4 import BeautifulSoup
import requests
import pandas as pd
from io import StringIO
import functions_framework

# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def process_electric_car_web_scraping(cloud_event):
  # URL base de la página a hacer scraping
  #base_url = 'https://ev-database.org/'
  base_url = 'https://ev-database.org/'
  url = base_url

  # Lista para almacenar los datos
  electric_cars = []
  while True:
    # Realizar la solicitud GET a la página
    response = requests.get(url)
    # Verificar si la solicitud fue exitosa (código de estado 200)
    if response.status_code == 200:
      # Parsear el contenido HTML
      soup = BeautifulSoup(response.text, 'html.parser')
      # Encontrar todos los elementos <div> con la clase 'list-item'
      list_items = soup.find_all('div', class_='list-item')

      # Iterar sobre los elementos 'list-item' y extraer información
      for item in list_items:
        # Buscar el elemento <div> con la clase 'item-data'
        item_data = item.find('div', class_='item-data')
        if item_data:
          # Encontrar el nombre del vehículo
          car_name = item_data.find('a', class_='title').text.strip()

          # Encontrar la velocidad máxima
          top_speed_tag = item_data.find('span', class_='topspeed')
          top_speed = top_speed_tag.text.strip() if top_speed_tag else "Velocidad máxima no encontrada"

          # Encontrar el tiempo de 0 a 100 km/h
          acceleration_tag = item_data.find('span', class_='acceleration')
          acceleration = acceleration_tag.text.strip() if acceleration_tag else "Tiempo de 0 a 100 km/h no encontrado"

          # Encontrar el rango
          range_tag = item_data.find('span', class_='erange_real')
          range_value = range_tag.text.strip() if range_tag else "Rango no encontrado"

          # Encontrar la velocidad de carga rápida
          fastcharge_speed_tag = item_data.find('span', class_='fastcharge_speed_print')
          fastcharge_speed = fastcharge_speed_tag.text.strip() if fastcharge_speed_tag else "Velocidad de carga rápida no encontrada"

          # Encontrar el contenedor de precios
          pricing_container = item.find('div', class_='pricing')
          if pricing_container:
            # Encontrar todos los elementos de precio
            pricing_elements = pricing_container.find_all('div', class_='price_buy')
                    
            # Crear un diccionario para almacenar los precios de cada país
            prices = {}
                    
            # Iterar sobre los elementos de precio
            for pricing_element in pricing_elements:
              # Extraer el precio y el país
              price_text = pricing_element.find('span').text.strip()
              country_code = pricing_element.find('span', class_='flag-icon').attrs['class'][1].replace('country_', '').upper()
              # Agregar el precio al diccionario de precios
              prices[country_code] = price_text
                
              # Agregar los datos a la lista
              electric_cars.append({
                        "Nombre del vehículo": car_name,
                        "Velocidad máxima": top_speed,
                        "Tiempo de 0 a 100 km/h": acceleration,
                        "Rango": range_value,
                        "Velocidad de carga rápida": fastcharge_speed,
                        "Precios por país": prices
              })
      # Encontrar el enlace a la siguiente página, si existe
      next_page_link = soup.find('a', class_='next')
      if next_page_link:
        url = base_url + next_page_link['href']
      else:
        break  # Salir del bucle si no hay más páginas disponibles

    else:
      print('Error al acceder a la página:', response.status_code)
      break

  # Cargamos los datos de los vehiculos electricos en un DataFrame
  df = pd.DataFrame(electric_cars)

  # Guardar el DataFrame como un archivo CSV
  csv_buffer = StringIO()
  df.to_csv(csv_buffer, index=False)

  ############
  ### Load ###
  ############
  # Subir el archivo CSV a Google Cloud Storage
  client = storage.Client()
  bucket = client.get_bucket("data_electric_car")
  blob = bucket.blob("electricCar.csv")
  blob.upload_from_string(csv_buffer.getvalue())

  print(f"Ha sido creado el archivo 'electricCars.csv' y guardado en el bucket 'data_electric_car'.")
    
# Para cargar esta Cloud Function desde mi pc local debo ejecutar desde la terminal:
# rafael@i5G9:~$ gcloud functions deploy process_electric_car_web_scraping --gen2 --region us-east1 --runtime python310 --trigger-http --allow-unauthenticated --entry-point process_electric_car_web_scraping --timeout 540s --memory 256MB

# gcloud functions deploy process_electric_car_web_scraping \
# --gen2 \
# --region=us-east1 \
# --runtime=python310 \
# --trigger-bucket=data_alquiler_gran_volumen \
# --entry-point=process_electric_car_web_scraping \
# --timeout 540s \
# --memory 256MB
