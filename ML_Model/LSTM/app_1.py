import streamlit as st
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
from retry_requests import retry
import holidays
from datetime import datetime
from funciones_LSTM import *
import warnings
warnings.filterwarnings('ignore')
import io
import sys
import plotly.graph_objects as go

# Definir una función para redirigir la salida estándar
def suppress_stdout():
    sys.stdout = io.StringIO()

# Definir una función para restaurar la salida estándar
def restore_stdout():
    sys.stdout = sys.__stdout__

# Cargar el modelo previamente entrenado
modelo = load("model_app_1.joblib")
scaler_X = load('scaler_X.joblib')
scaler_Y = load('scaler_Y.joblib')

def predecir_demanda():
    suppress_stdout()
    cantidad_dias = 7
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40.71427,
        "longitude": -74.00597,
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snowfall"],
        "timezone": "America/New_York",
        "past_days": 1,
        "forecast_days": cantidad_dias
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["rain"] = hourly_rain
    hourly_data["snowfall"] = hourly_snowfall

    df = pd.DataFrame(data = hourly_data)
    df['fecha'] = df['date'].dt.date
    df['año'] = df['date'].dt.year
    df['mes'] = df['date'].dt.month
    df['dia'] = df['date'].dt.day
    df['hora'] = df['date'].dt.hour
    df['dia_semana'] = df['date'].dt.weekday + 1
    year = 2024
    us_holidays = holidays.US(years=year)
    df['holiday'] = np.where(df['fecha'].isin(us_holidays), 1, 0)

    hora_actual = datetime.now().time()
    fecha_actual = datetime.now().date()
    hora = hora_actual.hour
    dia = fecha_actual.day
    mes = fecha_actual.month
    anio = fecha_actual.year
    indice = df.loc[(df['año'] == anio) & (df['mes'] == mes) & (df['dia'] == dia) & (df['hora'] == hora)].index[0]
    df = df.loc[indice-4:,:]
    columnas_X = ['año', 'mes', 'dia', 'hora', 'dia_semana', 'holiday', 'temperature_2m', 'rain', 'relative_humidity_2m', 'snowfall']
    X = df[columnas_X]
    X = X.reset_index(drop=True)
    X_scaled = scaler_X.transform(X)
    X_reshaped = generador_X(X_scaled, p=5)
    columnas_Y = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']

    
    Y_predict = modelo.predict(X_reshaped) / 5
    

    Y_predict_df = pd.DataFrame(Y_predict, columns=columnas_Y)
    Y_predict_original = scaler_Y.inverse_transform(Y_predict_df)
    Y_predict_original_df = pd.DataFrame(Y_predict_original, columns=columnas_Y)

    prediccion = pd.concat([X[['año', 'mes', 'dia', 'hora']].loc[4:,:].reset_index(drop=True), Y_predict_original_df], axis=1)
    prediccion.dropna(inplace=True)
    restore_stdout()
    return prediccion


# Configuración de la página
st.title("Predicción de Demanda")
columnas_Y = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
# Botón para predecir la demanda
if st.button("Predecir demanda"):
    demanda_predicha = predecir_demanda()
    
    st.subheader("Demanda Predicha")
    st.table(demanda_predicha.round().astype(int))
    
    st.subheader("Gráficos")

    data = pd.DataFrame({
        'Horas': demanda_predicha.index,
        **{columna: demanda_predicha[columna] for columna in columnas_Y}
    })

    # Configurar el diseño de la página
    st.title("Gráficos de Líneas")

    # Crear una figura de Plotly para los gráficos
    fig = go.Figure()

    # Agregar cada línea al gráfico
    for columna in columnas_Y:
        fig.add_trace(go.Scatter(x=data['Horas'], y=data[columna], mode='lines', name=columna))

    # Actualizar el diseño del gráfico
    fig.update_layout(title='Demanda Predicha',
                    xaxis_title='Hora',
                    yaxis_title='Demanda',
                    legend_title='Barrios',
                    height=800)  # Ajustar la altura del gráfico

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)

