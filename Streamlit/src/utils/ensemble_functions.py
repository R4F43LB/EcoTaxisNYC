# este archivo contiene las funciones para generar un ensemble de modelos LSTM, RF, XGB y LGBM y funciones para realizar predicciones
# a partir del ensemble construido

# importamos librerias necesarias

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import holidays
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from scikeras.wrappers import KerasRegressor
from keras.initializers import HeNormal
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
from joblib import load
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)

def generador_X_lstm(X, p=5):
    ''' 
    A partir un dataframe, X, y los pasos de tiempo p, retorna un tensor 3D para X.
    args:
    - X: pandas dataframe, con m filas (muestras) y n columnas (features)
    - p (int): número de pasos de tiempo a considerar
    returns:
    - X_reshaped: numpy array de forma (m, p, n)
    '''
    
    X_reshaped_list = list()

    m = len(X)
    X = np.array(X)

    for i in range(m-p):
        X_reshaped_list.append(X[i:i+p, :])
    
    X_reshaped = np.array(X_reshaped_list)

    return X_reshaped

def generar_X(nro_pasos=5, cantidad_dias=7):
    '''
    Genera un DataFrame con las features necesarias para usar como entrada en modelos de predicción.

    Argumentos:
    - nro_pasos (int): Número de pasos de tiempo utilizados en el reshape de LSTM. Por defecto es 5.
    - cantidad_dias (int): Cantidad de días de datos meteorológicos a obtener. Por defecto es 7.

    Retorna:
    - df_pred (DataFrame): Un DataFrame con las caracteristicas de entrada para el modelo.

    Descripción detallada:
    La función utiliza la API de Open-Meteo para obtener datos meteorológicos recientes, incluyendo temperatura, humedad relativa, lluvia y nieve.
    Luego, procesa los datos y los organiza en un DataFrame, incluyendo la fecha y hora, el año, el mes, el día, la hora, el día de la semana y si es feriado en EE.UU.
    Finalmente, selecciona los últimos 'nro_pasos' pasos de tiempo y devuelve el DataFrame con las características seleccionadas.
    '''
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

    df_pred = pd.DataFrame(data = hourly_data)

    hora_actual = datetime.now().time()
    fecha_actual = datetime.now().date()
    hora = hora_actual.hour
    dia = fecha_actual.day
    mes = fecha_actual.month
    anio = fecha_actual.year
    datetime_actual = pd.to_datetime(str(anio) + '-' + str(mes) + '-' + str(dia) + ' ' + str(hora) + ':00:00')

    df_pred['date'] = df_pred['date'].dt.tz_localize(None)
    df_pred['fecha'] = df_pred['date'].dt.date
    df_pred['año'] = df_pred['date'].dt.year
    df_pred['mes'] = df_pred['date'].dt.month
    df_pred['dia'] = df_pred['date'].dt.day
    df_pred['hora'] = df_pred['date'].dt.hour
    df_pred['dia_semana'] = df_pred['date'].dt.weekday + 1
    us_holidays = holidays.US(years=[anio - 1, anio, anio + 1])
    us_holidays
    df_pred['holiday'] = np.where(df_pred['fecha'].isin(us_holidays), 1, 0)
    df_pred.drop(columns='fecha', inplace=True)
    df_pred.rename(columns={'date': 'datetime'}, inplace=True)
    df_pred.set_index('datetime', inplace=True)

    df_pred = df_pred.loc[datetime_actual - timedelta(hours=nro_pasos-1):,:]
    columnas_X = ['año', 'mes', 'dia', 'hora', 'dia_semana', 'holiday', 'temperature_2m', 
                                            'rain', 'relative_humidity_2m', 'snowfall']
    df_pred = df_pred[columnas_X]
    
    return df_pred

def predecir(ensemble, cant_dias=7, verbose=0, nro_pasos=5, test=False, test_data=None, ponderacion= 'lineal', alpha=1):
    
    '''
    Realiza predicciones utilizando el ensemble de modelos de machine learning.

    Argumentos:
    - ensemble (dict): Diccionario que contiene el ensemble de modelos de machine learning previamente entrenados.
    - cant_dias (int): Cantidad de días a predecir.
    - verbose (int): Nivel de detalle de los mensajes de progreso (0 para no mostrar mensajes, 1 para mostrar).
    - nro_pasos (int): Número de pasos hacia atrás a considerar para la predicción de cada modelo.
    - test (bool): Indica si se están realizando pruebas (True) o no (False).
    - test_data (DataFrame): DataFrame que contiene los datos de prueba, solo se utiliza si test=True.
    - ponderacion (str): 'lineal' indica función de ponderación lineal, 'exp' indica función de ponderación exponencial negativa.
    - alpha (float): indica el valor del factor en el exponente en el caso de ponderación exponencial. posibles valores: [0.1, 0.5, 1, 10, 100]

    Retorna:
    - nyc_predictions (dict): Diccionario que contiene las predicciones del ensemble para cada distrito.

    Descripción detallada:
    La función utiliza un modelo de ensemble previamente construido para hacer predicciones sobre datos meteorológicos recientes.
    Primero, genera los datos meteorológicos con la función `generar_X` utilizando un número específico de pasos de tiempo y días.
    Luego, hace predicciones utilizando los modelos LSTM, RandomForest, XGBoost y LightGBM del ensemble.
    Después, combina las predicciones de los distintos modelos utilizando las ponderaciones correspondientes y retorna un diccionario de DataFrames con las predicciones.
    
    '''

    if test == True:
        X = test_data
    if test == False:
        X = generar_X(nro_pasos=nro_pasos, cantidad_dias=cant_dias)

    columnas_Y = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']

    lstm_model = ensemble['models']['lstm']
    scaler_X = MinMaxScaler()
    scaler_Y = ensemble['lstm_data']['scaler_y']
    X_scaled = scaler_X.fit_transform(X)
    X_reshaped = generador_X_lstm(X_scaled)
    lstm_pred = lstm_model.predict(X_reshaped, verbose=verbose)
    Y_predict_df = pd.DataFrame(lstm_pred, columns=columnas_Y)
    Y_predict_original = scaler_Y.inverse_transform(Y_predict_df)
    indice_pred = X.iloc[nro_pasos-1:-1,:].index
    lstm_predictions_df = pd.DataFrame(Y_predict_original, columns= columnas_Y, index=indice_pred)

    ponderaciones = ensemble['ponderaciones']

    lstm_predictions = {}
    rf_predictions = {}
    xgb_predictions = {}
    lgbm_predictions = {}

    nyc_predictions = {}
    for district in columnas_Y:

        lstm_predictions[district] = pd.DataFrame(lstm_predictions_df[district].rename('lstm'))

        rf_model = ensemble['models']['rf'][district]
        rf_pred = rf_model.predict(X.iloc[nro_pasos-1:-1,:])
        rf_pred_df = pd.DataFrame({'rf': rf_pred}, index=indice_pred)
        rf_predictions[district] = rf_pred_df

        xgb_model = ensemble['models']['xgb'][district]
        xgb_pred = xgb_model.predict(X.iloc[nro_pasos-1:-1,:])
        xgb_pred_df = pd.DataFrame({'xgb': xgb_pred}, index=indice_pred)
        xgb_predictions[district] = xgb_pred_df

        lgbm_model = ensemble['models']['lgbm'][district]
        lgbm_pred = lgbm_model.predict(X.iloc[nro_pasos-1:-1,:])
        lgbm_pred_df = pd.DataFrame({'lgbm': lgbm_pred}, index=indice_pred)
        lgbm_predictions[district] = lgbm_pred_df

        if ponderacion == 'lineal':

            lstm_pond = ponderaciones[district][ponderacion]['lstm']
            rf_pond = ponderaciones[district][ponderacion]['rf']
            xgb_pond = ponderaciones[district][ponderacion]['xgb']
            lgbm_pond = ponderaciones[district][ponderacion]['lgbm']

        if ponderacion == 'exp':
            lstm_pond = ponderaciones[district][ponderacion][alpha]['lstm']
            rf_pond = ponderaciones[district][ponderacion][alpha]['rf']
            xgb_pond = ponderaciones[district][ponderacion][alpha]['xgb']
            lgbm_pond = ponderaciones[district][ponderacion][alpha]['lgbm']

        ensemble_predictions = lstm_pond*lstm_predictions[district]['lstm'] + rf_pond*rf_pred_df['rf'] + xgb_pond*xgb_pred_df['xgb'] + lgbm_pond*lgbm_pred_df['lgbm']
        ensemble_predictions.rename('ensemble', inplace=True)
        ensemble_predictions_df = pd.DataFrame(ensemble_predictions, index=indice_pred)

        nyc_predictions[district] = pd.concat([lstm_predictions[district], rf_pred_df, xgb_pred_df, lgbm_pred_df, ensemble_predictions_df ], axis=1)
        
        condicion = lambda x: 0 if x < 0 else x
        nyc_predictions[district] = nyc_predictions[district].applymap(condicion)
        nyc_predictions[district] = nyc_predictions[district].astype(int)
        
    return nyc_predictions
    