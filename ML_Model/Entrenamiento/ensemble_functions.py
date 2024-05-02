# este archivo contiene las funciones para generar un ensemble de modelos LSTM, RF, XGB y LGBM y funciones para realizar predicciones
# a partir del ensemble construido

# importamos librerias necesarias

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
sns.set_theme()
import holidays
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
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
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import pytz

np.random.seed(42)

# Funciones para el modelo de LSTM

def generador_X_Y(X, Y, p=5):
    ''' 
    A partir dos dataframes, X e Y, y los pasos de tiempo, retorna un tensor 3D para X y una array 2D para Y.
    args:
    - X: pandas dataframe, con m filas (muestras) y n columnas (features)
    - Y: pandas dataframe, con m filas (muestras) e y columnas (longitud de vector de respuesta)
    - p (int): número de pasos de tiempo a considerar
    returns:
    - X_reshaped: numpy array de forma (m, p, n)
    - Y_reshaped: numpy array de forma (m, y)
    '''
    
    X_reshaped_list = list()
    Y_rehaped_list = list()

    m = len(X)
    X = np.array(X)
    Y = np.array(Y)

    for i in range(m-p):
        X_reshaped_list.append(X[i:i+p, :])
        Y_rehaped_list.append(Y[i+p-1, :])
    
    X_reshaped = np.array(X_reshaped_list)
    Y_reshaped = np.array(Y_rehaped_list)

    return X_reshaped, Y_reshaped

def crear_modelo(X_reshaped, Y_reshaped, nro_capas_lstm=1, unidades_capa= [50], nro_capas_dense=0, unidades_dense=[0], loss_= 'mse', 
                 act_salida='linear', act_lstm= 'relu', drop_out= 0, learning_r=0.001, act_dense= 'linear'):
    '''
    Crea un modelo de red neuronal recurrente (RNN) utilizando la arquitectura de Long Short-Term Memory (LSTM) para tareas de aprendizaje automático, como la predicción de series temporales.

    Argumentos:
    - X_reshaped: Array numpy que contiene los datos de entrada con la forma (número de muestras, número de pasos de tiempo, número de características). Representa las secuencias de entrada al modelo.
    - Y_reshaped: Array numpy que contiene los datos de salida esperados con la forma (número de muestras, número de características). Representa las etiquetas correspondientes a las secuencias de entrada.
    - nro_capas_lstm: Número entero que indica la cantidad de capas LSTM que se deben agregar al modelo. Por defecto, se establece en 1.
    - unidades_capa: Lista de enteros que especifica el número de unidades en cada capa LSTM. Cada elemento de la lista corresponde al número de unidades en una capa LSTM particular. Por defecto, se establece en [50].
    - nro_capas_dense: Número entero que indica la cantidad de capas densas que se deben agregar al modelo después de las capas LSTM. Por defecto, se establece en 0.
    - unidades_dense: Lista de enteros que especifica el número de unidades en cada capa densa. Cada elemento de la lista corresponde al número de unidades en una capa densa particular. Por defecto, se establece en [0].
    - loss_: Cadena que indica la función de pérdida a utilizar durante el entrenamiento del modelo. Por defecto, se establece en 'mse' (Mean Squared Error).
    - act_salida: Cadena que especifica la función de activación a utilizar en la capa de salida del modelo. Por defecto, se establece en 'linear'.
    - act_lstm: Cadena que indica la función de activación a utilizar en las capas LSTM del modelo. Por defecto, se establece en 'relu'.
    - drop_out: Valor flotante que representa la tasa de abandono para las capas Dropout, utilizadas para regularizar el modelo y prevenir el sobreajuste. Por defecto, se establece en 0.
    - learning_r: Valor flotante que indica la tasa de aprendizaje del optimizador Adam utilizado para entrenar el modelo. Por defecto, se establece en 0.001.
    - act_dense: Cadena que especifica la función de activación a utilizar en las capas densas del modelo. Por defecto, se establece en 'linear'.

    Devuelve:
    - model: Modelo secuencial de Keras configurado de acuerdo a los parámetros especificados.

    '''

    # Construimos el modelo LSTM
    model = Sequential()
    if nro_capas_lstm > 1:

        model.add(LSTM(units=unidades_capa[0], activation=act_lstm, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]), return_sequences=True))
        model.add(Dropout(drop_out))
        for i in range(1, nro_capas_lstm-1):
            model.add(LSTM(units=unidades_capa[i], activation=act_lstm, return_sequences=True))
            model.add(Dropout(drop_out))
        model.add(LSTM(units=unidades_capa[-1], activation=act_lstm))
 

    if nro_capas_lstm == 1:
        model.add(LSTM(units=unidades_capa[0], activation=act_lstm, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
        model.add(Dropout(drop_out))
    
    if nro_capas_dense > 0:
        for i in range(nro_capas_dense):
            model.add(Dense(units=unidades_dense[i], activation=act_dense, kernel_initializer=HeNormal()))
    
    model.add(Dense(units=Y_reshaped.shape[1], activation=act_salida, kernel_initializer=HeNormal() ))

    opt = Adam(learning_rate=learning_r)

    model.compile(optimizer=opt, loss=loss_)  # Compilación del modelo 

    return model

def construir_modelo_cv_lstm(X, Y, nro_pasos=5, nro_capas_lstm=2, unidades_capa= [100,60], loss_= 'mse', act_salida='relu', act_lstm= 'relu', 
                     epochs= 350, batch= 32, drop_out= 0, scaler= 'minmax', learning_r=0.0001, nro_capas_dense=0, unidades_dense=[0], 
                     act_dense= 'linear', verbose=0):
    '''
    Construye y entrena un modelo de red neuronal utilizando la técnica de validación cruzada en series temporales.

    Argumentos:
    - X: Array numpy que contiene los datos de entrada.
    - Y: Array numpy que contiene los datos de salida.
    - nro_pasos: Número entero que indica la cantidad de pasos de tiempo a considerar para la secuencia de entrada. Por defecto, se establece en 10.
    - nro_capas_lstm: Número entero que indica la cantidad de capas LSTM que se deben agregar al modelo. Por defecto, se establece en 1.
    - unidades_capa: Lista de enteros que especifica el número de unidades en cada capa LSTM. Cada elemento de la lista corresponde al número de unidades en una capa LSTM particular. Por defecto, se establece en [50].
    - loss_: Cadena que indica la función de pérdida a utilizar durante el entrenamiento del modelo. Por defecto, se establece en 'mse' (Mean Squared Error).
    - act_salida: Cadena que especifica la función de activación a utilizar en la capa de salida del modelo. Por defecto, se establece en 'linear'.
    - act_lstm: Cadena que indica la función de activación a utilizar en las capas LSTM del modelo. Por defecto, se establece en 'relu'.
    - epochs: Número entero que indica la cantidad de épocas de entrenamiento. Por defecto, se establece en 50.
    - batch: Número entero que indica el tamaño del lote durante el entrenamiento. Por defecto, se establece en 32.
    - drop_out: Valor flotante que representa la tasa de abandono para las capas Dropout, utilizadas para regularizar el modelo y prevenir el sobreajuste. Por defecto, se establece en 0.
    - scaler: Cadena que indica el método de escalado a aplicar a los datos. Puede ser 'minmax' para escalamiento Min-Max o 'standard' para escalamiento estándar. Por defecto, se establece en 'minmax'.
    - learning_r: Valor flotante que indica la tasa de aprendizaje del optimizador Adam utilizado para entrenar el modelo. Por defecto, se establece en 0.001.
    - nro_capas_dense: Número entero que indica la cantidad de capas densas que se deben agregar al modelo después de las capas LSTM. Por defecto, se establece en 0.
    - unidades_dense: Lista de enteros que especifica el número de unidades en cada capa densa. Cada elemento de la lista corresponde al número de unidades en una capa densa particular. Por defecto, se establece en [0].
    - act_dense: Cadena que especifica la función de activación a utilizar en las capas densas del modelo. Por defecto, se establece en 'linear'.

    Devuelve:
    - modelo: Modelo entrenado.
    - scaler_X: Objeto de la clase MinMaxScaler o StandardScaler utilizado para escalar los datos de entrada.
    - scaler_Y: Objeto de la clase MinMaxScaler o StandardScaler utilizado para escalar los datos de salida.
    - predictions: diccionario con las predicciones para cada distrito del set de testeo.
    - scores: Lista de puntajes de error cuadrático negativo obtenidos durante la validación cruzada.
    - history: Objeto que contiene el historial de entrenamiento del modelo.
    - loss: Valor de la pérdida final obtenido en el conjunto de pruebas.
    - rmse_dict = diccionario con los valores de RMSE en el set de testeo para cada distrito.
    '''

    if verbose==1:
        print('Inicio de construcción modelo LSTM')
    # Normalizar los datos
    if scaler == 'minmax':
        scaler_X = MinMaxScaler()
        scaler_Y = MinMaxScaler()
    if scaler == 'standard':
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    X_test_scaled = scaler_X.transform(X_test)
    Y_test_scaled = scaler_Y.transform(Y_test)

    # Reshape de los datos para que sean 3D (número de muestras, número de pasos de tiempo, número de características)
    X_train_reshaped, Y_train_reshaped = generador_X_Y(X_train_scaled, Y_train_scaled, nro_pasos)
    X_test_reshaped, Y_test_reshaped = generador_X_Y(X_test_scaled, Y_test_scaled, nro_pasos)

    # Envolvemos el modelo en un estimador de Scikit-Learn
    estimator = KerasRegressor(
        build_fn=crear_modelo,
        X_reshaped=X_train_reshaped,
        Y_reshaped=Y_train_reshaped,
        nro_capas_lstm=nro_capas_lstm,
        unidades_capa=unidades_capa,
        nro_capas_dense = nro_capas_dense,
        unidades_dense = unidades_dense,
        loss_=loss_,
        act_salida= act_salida,
        act_lstm= act_lstm,
        act_dense = act_dense,
        drop_out=drop_out,
        learning_r=learning_r,
        epochs=epochs,
        batch_size=batch,
        verbose=0
    )

    # Usamos el regresor en cross_val_score

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(estimator, X_train_reshaped, Y_train_reshaped, cv=tscv, scoring='neg_mean_squared_error', verbose=0)

    # entrenamos el modelo

    modelo = crear_modelo(X_train_reshaped, Y_train_reshaped, nro_capas_lstm, unidades_capa, nro_capas_dense, unidades_dense, loss_, 
                    act_salida, act_lstm, drop_out, learning_r, act_dense)
    
    history = modelo.fit(X_train_reshaped, Y_train_reshaped, epochs=epochs, batch_size=batch, validation_data=(X_test_reshaped, Y_test_reshaped), verbose=0)

    if loss_ == 'mse':
        loss = modelo.evaluate(X_test_reshaped, Y_test_reshaped, verbose=verbose)
    else:
        y_pred = modelo.predict(X_test_reshaped)
        loss = np.mean(tf.keras.losses.mean_squared_error(Y_test_reshaped, y_pred).numpy())
    
    Y_predict = modelo.predict(X_test_reshaped, verbose=verbose)
    Y_predict_df = pd.DataFrame(Y_predict, columns=Y.columns)
    Y_predict_original = scaler_Y.inverse_transform(Y_predict_df)
    Y_predict_original_df = pd.DataFrame(Y_predict_original, columns= Y.columns, index=Y_test.iloc[nro_pasos-1:-1,:].index)

    Y_test_df = pd.DataFrame(Y_test_reshaped, columns=Y.columns)
    Y_test_original = scaler_Y.inverse_transform(Y_test_df)
    Y_test_original_df = pd.DataFrame(Y_test_original, columns=Y.columns, index= Y_test.iloc[nro_pasos-1:-1,:].index)

    rmse_dict = {}
    test_predictions = {}

    for district in Y.columns:
        y_pred = Y_predict_original_df[district]
        y_test = Y_test_original_df[district]
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        rmse_dict[district] = rmse

        predictions_df = pd.DataFrame({
        'Predicted': y_pred,
        'Real values': y_test
        }, index=y_test.index)

        test_predictions[district] = predictions_df
    
    if verbose ==1:
        print('Finalizada la construcción de LSTM')
        print('RMSE por distrito:')
        print(rmse_dict)

    return modelo, scaler_X, scaler_Y, test_predictions, scores, history, loss, rmse_dict

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

# Funciones para Random Forest

def generar_modelos_cv_rf(df, columnas_X = ['año', 'mes', 'dia', 'hora', 'dia_semana', 'holiday', 
                                         'temperature_2m', 'rain', 'relative_humidity_2m', 'snowfall'],
                           columnas_Y= ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'],
                           verbose=0):
    '''
    Genera modelos RandomForest utilizando validación cruzada en series temporales para cada distrito de Nueva York.

    Argumentos:
    - df (DataFrame): DataFrame de pandas que contiene los datos.
    - columnas_X (list): Lista de nombres de columnas a utilizar como características para el modelo. Por defecto, se incluyen columnas relacionadas con la fecha y datos meteorológicos.
    - columnas_Y (list): Lista de nombres de columnas que representan las variables objetivo para predecir. Por defecto, se incluyen los distritos de Nueva York.
    - verbose (int): Nivel de verbosidad. Si es 1, imprime información sobre el proceso de construcción del modelo y evaluación para cada distrito. Si es 0, no imprime nada. Por defecto, es 0.

    Retorna:
    - models (dict): Diccionario que contiene los modelos RandomForest entrenados para cada distrito, donde las claves son los nombres de los distritos y los valores son los modelos entrenados.
    - test_predictions (dict): Diccionario que contiene las predicciones realizadas por los modelos RandomForest en el conjunto de prueba para cada distrito, donde las claves son los nombres de los distritos y los valores son DataFrames con las predicciones y los valores reales.
    - errors (dict): Diccionario que contiene las métricas de error calculadas para cada modelo RandomForest en el conjunto de prueba para cada distrito, donde las claves son los nombres de los distritos y los valores son diccionarios con las métricas (MAE, MSE, RMSE, MAPE).
    '''

    if verbose==1:
        print('Inicio de construcción modelo RF')

    X = df[columnas_X] 
    Y = df[columnas_Y]

    models = {}
    test_predictions = {}
    errors = {}

    for district in columnas_Y:
        if verbose==1:
            print('Inicio de evaluación de', district)
        X_train, X_test, y_train, y_test = train_test_split(X, Y[district], test_size=0.2, random_state=42, shuffle=False)

        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        # Creamos el modelo RandomForest
        model = RandomForestRegressor(random_state=42) 

        # creamos gridsearch con validación cruzada
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Obtenemos el mejor modelo
        best_model = grid_search.best_estimator_
        models[district] = best_model
       
        best_params = grid_search.best_params_
        if verbose==1:
            print('Mejores Hiperparámetros:', best_params)
        # Predecimos los datos de prueba
        y_pred = best_model.predict(X_test)

        # Evaluamos el modelo
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        errors[district] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}
        if verbose==1:
            print('RMSE:', rmse)

        predictions_df = pd.DataFrame({
        'Predicted': y_pred,
        'Real values': y_test
        }, index=y_test.index )

        test_predictions[district] = predictions_df
        if verbose==1:
            print('-----------------------------------------------------------------------------------------')
    return models, test_predictions, errors

# Funciones para XGBoost

def generar_modelos_cv_xgb(df, columnas_X= ['año', 'mes', 'dia', 'hora', 'dia_semana', 'holiday', 
                                        'temperature_2m', 'rain', 'relative_humidity_2m', 'snowfall'],
                            columnas_Y= ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'],
                            verbose=0):
    '''
    Genera modelos XGBoost utilizando validación cruzada en series temporales para cada distrito de Nueva York.

    Argumentos:
    - df (DataFrame): DataFrame de pandas que contiene los datos.
    - columnas_X (list): Lista de nombres de columnas a utilizar como características para el modelo. Por defecto, se incluyen columnas relacionadas con la fecha y datos meteorológicos.
    - columnas_Y (list): Lista de nombres de columnas que representan las variables objetivo para predecir. Por defecto, se incluyen los distritos de Nueva York.
    - verbose (int): Nivel de verbosidad. Si es 1, imprime información sobre el proceso de construcción del modelo y evaluación para cada distrito. Si es 0, no imprime nada. Por defecto, es 0.

    Retorna:
    - models (dict): Diccionario que contiene los modelos XGBoost entrenados para cada distrito, donde las claves son los nombres de los distritos y los valores son los modelos entrenados.
    - test_predictions (dict): Diccionario que contiene las predicciones realizadas por los modelos XGBoost en el conjunto de prueba para cada distrito, donde las claves son los nombres de los distritos y los valores son DataFrames con las predicciones y los valores reales.
    - errors (dict): Diccionario que contiene las métricas de error calculadas para cada modelo XGBoost en el conjunto de prueba para cada distrito, donde las claves son los nombres de los distritos y los valores son diccionarios con las métricas (MAE, MSE, RMSE, MAPE).
    '''
    
    if verbose==1:
        print('Inicio de construcción modelo XGB')

    X = df[columnas_X]
    Y = df[columnas_Y]

    models = {}
    test_predictions = {}
    errors = {}

    # definimos el split de time series
    tscv = TimeSeriesSplit(n_splits=5)

    # Definir los hiperparámetros para GridSearchCV
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'n_estimators': [100, 500, 1000]
    }

    for district in columnas_Y:
        if verbose==1:
            print('Inicio de evaluación de', district)
        X_train, X_test, y_train, y_test = train_test_split(X, Y[district], test_size=0.2, random_state=42, shuffle=False)

        # Crear  modelo RandomForest
        model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.5)

        # Realizar la búsqueda de hiperparámetros mediante GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=0)
        grid_search.fit(X_train, y_train)

        # Obtener el mejor modelo
        best_model = grid_search.best_estimator_
        models[district] = best_model

        best_params = grid_search.best_params_
        if verbose==1:
            print('Mejores Hiperparámetros:', best_params)

        # Predecir los datos de prueba
        y_pred = best_model.predict(X_test)

        # Evaluar el modelo
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        errors[district] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}
        if verbose==1:
            print('RMSE:', rmse)

        predictions_df = pd.DataFrame({
        'Predicted': y_pred,
        'Real values': y_test
        }, index=y_test.index )

        test_predictions[district] = predictions_df
        if verbose==1:
            print('-----------------------------------------------------------------------------------------')
    return models, test_predictions, errors

# funciones para LGBM

def generar_modelos_cv_lgbm(df, columnas_X = ['año', 'mes', 'dia', 'hora', 'dia_semana', 'holiday', 'temperature_2m', 
                                      'rain', 'relative_humidity_2m', 'snowfall'],
                        columnas_Y = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'], verbose=0):

    '''
    Genera modelos LightGBM utilizando validación cruzada en series temporales para cada distrito de Nueva York.

    Argumentos:
    - df (DataFrame): DataFrame de pandas que contiene los datos.
    - columnas_X (list): Lista de nombres de columnas a utilizar como características para el modelo. Por defecto, se incluyen columnas relacionadas con la fecha y datos meteorológicos.
    - columnas_Y (list): Lista de nombres de columnas que representan las variables objetivo para predecir. Por defecto, se incluyen los distritos de Nueva York.
    - verbose (int): Nivel de verbosidad. Si es 1, imprime información sobre el proceso de construcción del modelo y evaluación para cada distrito. Si es 0, no imprime nada. Por defecto, es 0.

    Retorna:
    - models (dict): Diccionario que contiene los modelos LightGBM entrenados para cada distrito, donde las claves son los nombres de los distritos y los valores son los modelos entrenados.
    - test_predictions (dict): Diccionario que contiene las predicciones realizadas por los modelos LightGBM en el conjunto de prueba para cada distrito, donde las claves son los nombres de los distritos y los valores son DataFrames con las predicciones y los valores reales.
    - errors (dict): Diccionario que contiene las métricas de error calculadas para cada modelo LightGBM en el conjunto de prueba para cada distrito, donde las claves son los nombres de los distritos y los valores son diccionarios con las métricas (MAE, MSE, RMSE, MAPE).
    '''

    if verbose==1:
        print('Inicio de construcción modelo LGBM')
        
    X = df[columnas_X]
    Y = df[columnas_Y]

    models = {}
    test_predictions = {}
    errors = {}
    # definimos el generador de splits
    tscv = TimeSeriesSplit(n_splits=5)

    # Definir los parámetros para la búsqueda de hiperparámetros
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 500, 1000]
    }

    for district in columnas_Y:
        if verbose==1:
            print('Inicio de evaluación de', district)
        X_train, X_test, y_train, y_test = train_test_split(X, Y[district], test_size=0.2, random_state=42, shuffle=False)

        # Inicializar el modelo LightGBM
        model = lgb.LGBMRegressor(verbosity=-1)

        # Realizar la búsqueda de hiperparámetros mediante GridSearchCV con TimeSeriesSplit
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=0)
        grid_search.fit(X_train, y_train)

        # Obtener el mejor modelo
        best_model = grid_search.best_estimator_
        models[district] = best_model

        best_params = grid_search.best_params_
        if verbose==1:
            print('Mejores Hiperparámetros:', best_params)

        # Predecir los datos de prueba
        y_pred = best_model.predict(X_test)

        # Evaluar el modelo
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        errors[district] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

        if verbose==1:
            print('RMSE:', rmse)

        predictions_df = pd.DataFrame({
            'Predicted': y_pred,
            'Real values': y_test
        }, index=y_test.index)

        test_predictions[district] = predictions_df
        
        if verbose==1:
            print('-----------------------------------------------------------------------------------------')
    return models, test_predictions, errors

# Función para ponderar los diferentes modelos:

def generar_ponderaciones(lstm_errors, lgbm_errors, xgb_errors, rf_errors):
    '''
    Genera ponderaciones para los modelos LSTM, LightGBM, XGBoost y RandomForest basadas en los errores RMSE de cada modelo.

    Argumentos:
    - lstm_errors (dict): Diccionario que contiene los errores RMSE de los modelos LSTM para cada distrito, donde las claves son los nombres de los distritos y los valores son los RMSE.
    - lgbm_errors (dict): Diccionario que contiene los errores RMSE de los modelos LightGBM para cada distrito, donde las claves son los nombres de los distritos y los valores son diccionarios con las métricas (MAE, MSE, RMSE, MAPE).
    - xgb_errors (dict): Diccionario que contiene los errores RMSE de los modelos XGBoost para cada distrito, donde las claves son los nombres de los distritos y los valores son diccionarios con las métricas (MAE, MSE, RMSE, MAPE).
    - rf_errors (dict): Diccionario que contiene los errores RMSE de los modelos RandomForest para cada distrito, donde las claves son los nombres de los distritos y los valores son diccionarios con las métricas (MAE, MSE, RMSE, MAPE).

    Retorna:
    - ponderaciones (dict): Diccionario que contiene las ponderaciones para cada modelo LSTM, LightGBM, XGBoost y RandomForest para cada distrito, donde las claves son los nombres de los distritos y los valores son diccionarios con las ponderaciones de cada modelo.
    
    Descripción detallada de los elementos en el diccionario 'ponderaciones':
    - 'lin': ponderaciones calculadas con la fórmula: pond = 1 - rmse/total_rmse, normalizada dividiendo por la suma de las ponderaciones.
    - 'exp': ponderaciones calculadas con la fórmula: pond = np.exp(-alpha*rmse), normalizada dividiendo por la suma de las ponderaciones.
        -subclave 'alpha': contiene las ponderaciones calculadas para distintos valores de alpha: alphas = [0.001, 0.01, 0.1, 0.5]
    - 'inv': ponderaciones calculadas con la fórmula: pond = 1/rmse, normalizada dividiendo por la suma de las ponderaciones.
    '''

    ponderaciones = {}
    distritos = list(lstm_errors.keys())

    for dist in distritos:
        
        lstm_rmse = lstm_errors[dist]
        lgbm_rmse = lgbm_errors[dist]['RMSE']
        xgb_rmse = xgb_errors[dist]['RMSE']
        rf_rmse = rf_errors[dist]['RMSE']
        total_rmse = lstm_rmse + lgbm_rmse + xgb_rmse + rf_rmse
        # ponderación lineal
        pond_lstm = 1 - lstm_rmse/total_rmse
        pond_lgbm = 1 - lgbm_rmse/total_rmse
        pond_xgb = 1 - xgb_rmse/total_rmse
        pond_rf = 1 - rf_rmse/total_rmse
        sum_pond = pond_lstm + pond_lgbm + pond_xgb + pond_rf
        pond_lstm /= sum_pond
        pond_lgbm /= sum_pond
        pond_xgb /= sum_pond
        pond_rf /= sum_pond

        ponderaciones[dist] = {}
        ponderaciones[dist]['lin'] = {
            'lstm': pond_lstm,
            'lgbm': pond_lgbm,
            'xgb': pond_xgb,
            'rf': pond_rf
        }
        # ponderación exponencial
        ponderaciones[dist]['exp'] = {}
        alphas = [0.001, 0.01, 0.1, 0.5]
        for alpha in alphas:

            pond_lstm = np.exp(-alpha*lstm_rmse)
            pond_lgbm = np.exp(-alpha*lgbm_rmse)
            pond_xgb = np.exp(-alpha*xgb_rmse)
            pond_rf = np.exp(-alpha*rf_rmse)
            sum_pond = pond_lstm + pond_lgbm + pond_xgb + pond_rf
            pond_lstm /= sum_pond
            pond_lgbm /= sum_pond
            pond_xgb /= sum_pond
            pond_rf /= sum_pond

            ponderaciones[dist]['exp'][alpha] = {
                'lstm': pond_lstm,
                'lgbm': pond_lgbm,
                'xgb': pond_xgb,
                'rf': pond_rf
            }
        # ponderación inversa RMSE
        pond_lstm = 1/lstm_rmse
        pond_lgbm = 1/lgbm_rmse
        pond_xgb = 1/xgb_rmse
        pond_rf = 1/rf_rmse
        sum_pond = pond_lstm + pond_lgbm + pond_xgb + pond_rf
        pond_lstm /= sum_pond
        pond_lgbm /= sum_pond
        pond_xgb /= sum_pond
        pond_rf /= sum_pond
        ponderaciones[dist]['inv'] = {}
        ponderaciones[dist]['inv'] = {
            'lstm': pond_lstm,
            'lgbm': pond_lgbm,
            'xgb': pond_xgb,
            'rf': pond_rf
        }
        
    return ponderaciones

# Función para generar el ensemble:

def generar_ensemble(df, p= 5, columnas_X = ['año', 'mes', 'dia', 'hora', 'dia_semana', 'holiday', 'temperature_2m', 
                                            'rain', 'relative_humidity_2m', 'snowfall'],
                               columnas_Y = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'], verbose=0,
                               exportar='si', file='ensemble.joblib'):
    '''
    Genera un ensemble de modelos LSTM, RandomForest, XGBoost y LightGBM y exporta los resultados si se especifica.

    Argumentos:
    - df (DataFrame): El DataFrame que contiene los datos para entrenamiento y prueba.
    - p (int): El número de pasos para los datos de entrada en el modelo LSTM. Por defecto es 5.
    - columnas_X (list): Lista de columnas a usar como características para el modelo. Por defecto contiene las columnas de características.
    - columnas_Y (list): Lista de columnas a predecir. Por defecto contiene las columnas de distritos.
    - verbose (int): Nivel de verbosidad de la función. 0 para silencioso, 1 para mostrar detalles. Por defecto es 0.
    - exportar (str): Si se debe exportar el ensemble. 'si' para exportar, 'no' para no exportar. Por defecto es 'si'.
    - file (str): El nombre del archivo de salida si se exporta el ensemble. Por defecto es 'ensemble.joblib'.

    Retorna:
    - ensemble (dict): Un diccionario que contiene los modelos, los datos LSTM, los errores, las predicciones de prueba y las ponderaciones de los modelos.

    Descripción detallada de los elementos en el diccionario 'ensemble':
        - models (dict): Diccionario que contiene los modelos entrenados. Las claves son los nombres de los modelos ('lstm', 'rf', 'xgb', 'lgbm').
        - lstm_data (dict): Diccionario que contiene los datos relacionados con el modelo LSTM, como los escaladores, puntajes, historial de entrenamiento y pérdida.
        - errors (dict): Diccionario que contiene los errores de los modelos. Las claves son los nombres de los modelos ('lstm', 'rf', 'xgb', 'lgbm') y los valores son los errores asociados.
        - test_predictions (dict): Diccionario que contiene las predicciones de prueba de los modelos. Las claves son los nombres de los modelos y los valores son los DataFrames de predicciones.
        - ponderaciones (dict): Diccionario que contiene las ponderaciones de los modelos LSTM, RandomForest, XGBoost y LightGBM para cada distrito.
    '''

    X=df[columnas_X]
    Y=df[columnas_Y]

    lstm_model, scaler_X, scaler_Y, lstm_predictions, scores, history, loss, lstm_errors = construir_modelo_cv_lstm(X, Y, nro_pasos=p, 
                                                                                                                    verbose=verbose)

    rf_models, rf_predictions, rf_errors = generar_modelos_cv_rf(df, columnas_X=columnas_X, columnas_Y=columnas_Y, verbose=verbose)

    xgb_models, xgb_predictions, xgb_errors = generar_modelos_cv_xgb(df, columnas_X=columnas_X, columnas_Y=columnas_Y, verbose=verbose)

    lgbm_models, lgbm_predictions, lgbm_errors = generar_modelos_cv_lgbm(df, columnas_X=columnas_X, columnas_Y=columnas_Y, verbose=verbose)

    ponderaciones = generar_ponderaciones(lstm_errors, rf_errors, xgb_errors, lgbm_errors)

    models = {}
    models['lstm'] = lstm_model
    models['rf'] = rf_models
    models['xgb'] = xgb_models
    models['lgbm'] = lgbm_models

    lstm_data = {}
    lstm_data['scaler_x'] = scaler_X
    lstm_data['scaler_y'] = scaler_Y
    lstm_data['scores'] = scores
    lstm_data['history'] = history
    lstm_data['loss'] = loss

    errors = {}
    errors['lstm'] = lstm_errors
    errors['rf'] = rf_errors
    errors['xgb']= xgb_errors
    errors['lgbm']= lgbm_errors

    test_predictions = {}
    test_predictions['lstm'] = lstm_predictions
    test_predictions['rf'] = rf_predictions
    test_predictions['xgb']= xgb_predictions
    test_predictions['lgbm']= lgbm_predictions

    ensemble = {}
    ensemble['models'] = models
    ensemble['lstm_data'] = lstm_data
    ensemble['errors'] = errors
    ensemble['test_predictions'] = test_predictions
    ensemble['ponderaciones'] = ponderaciones

    if exportar == 'si':
        dump(ensemble, file)

    return ensemble

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

    hora_fecha_actual_NYC = datetime.now(pytz.timezone('America/New_York'))
    hora_actual = hora_fecha_actual_NYC.time()
    fecha_actual = hora_fecha_actual_NYC.date()
    
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

def predecir(ensemble, cant_dias=7, verbose=0, nro_pasos=5, test=False, test_data=None, ponderacion= 'lin', alpha=1,
             multiple=False, mejores_pond_por_distrito=None):
    
    '''
    Realiza predicciones utilizando el ensemble de modelos de machine learning.

    Argumentos:
    - ensemble (dict): Diccionario que contiene el ensemble de modelos de machine learning previamente entrenados.
    - cant_dias (int): Cantidad de días a predecir.
    - verbose (int): Nivel de detalle de los mensajes de progreso (0 para no mostrar mensajes, 1 para mostrar).
    - nro_pasos (int): Número de pasos hacia atrás a considerar para la predicción de cada modelo.
    - test (bool): Indica si se están realizando pruebas (True) o no (False).
    - test_data (DataFrame): DataFrame que contiene los datos de prueba, solo se utiliza si test=True.
    - ponderacion (str): 'lin' indica función de ponderación lineal, 'exp' indica función de ponderación exponencial negativa.
    - alpha (float): indica el valor del factor en el exponente en el caso de ponderación exponencial. posibles valores: [0.001, 0.5, 1, 10, 100]
    - multiple (bool): indica se se utilizarán distintas ponderaciones para cada distrito. Predeterminado False.
    - mejor_pond_por_distrito (dict): diccionario conteniendo las mejores funciones de ponderación para cada distrito. Se utiliza sólo en el caso en que multiple=True
    
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
    scaler_X = ensemble['lstm_data']['scaler_x']
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

        if multiple == True:
            ponderacion = mejores_pond_por_distrito[district][0]
            alpha = mejores_pond_por_distrito[district][1]

        if ponderacion != 'exp':
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
    
def graficar_predicciones(pred):
    
    '''
    Grafica las predicciones realizadas por el modelo de ensemble.

    Argumentos:
    - pred (dict): Diccionario que contiene los DataFrames de predicciones para cada distrito.

    Descripción detallada:
    La función recibe un diccionario de DataFrames de predicciones para cada distrito y grafica las predicciones realizadas por los modelos LSTM, RandomForest, XGBoost, LightGBM y el ensemble ponderado.
    Cada gráfico muestra las predicciones de cada modelo por separado y la predicción del ensemble.
    '''

    for district in pred.keys():
        plt.figure(figsize=(12,12))
        sns.set_style('darkgrid')
        sns.lineplot(pred[district]['lstm'], label='LSTM', color='red')
        sns.lineplot(pred[district]['rf'], label='RF', color='green')
        sns.lineplot(pred[district]['xgb'], label='XGB', color='blue')
        sns.lineplot(pred[district]['lgbm'], label='LGBM', color='orange')
        sns.lineplot(pred[district]['ensemble'], label='Ensemble', lw=2.5, color='black')
        plt.title(f'Pedicciones actuales de demanda en {district} utilizando Weighted Ensemble', fontsize=14)
        plt.xlabel('Fecha-hora', fontsize=12)
        plt.ylabel('Demanda', fontsize=12)
        plt.legend(title='Modelo', fontsize=10, title_fontsize='12')
        plt.tight_layout()
        plt.show()


def train_test_ensemble(data, verbose=0, test_size=0.2, nro_pasos=5, exportar='si', file='ensemble.joblib', ensemble=None):
    '''
    Entrena y prueba un ensemble de modelos de machine learning.

    Argumentos:
    - data (DataFrame): DataFrame que contiene los datos a utilizar.
    - verbose (int): Nivel de detalle de los mensajes de progreso (0 para no mostrar mensajes, 1 para mostrar).
    - test_size (float): Proporción de datos a utilizar como conjunto de prueba.
    - nro_pasos (int): Número de pasos hacia atrás a considerar para la predicción de cada modelo.
    - exportar (str): Opción para exportar el ensemble entrenado ('si' para exportar, 'no' para no exportar).
    - file (str): Nombre del archivo para exportar el ensemble.
    - ensemble (dict): Ensemble preentrenado si se va a realizar solo el testeo.

    Descripción detallada:
    Esta función entrena y prueba un ensemble de modelos de machine learning utilizando un conjunto de datos. Se pueden proporcionar datos de entrenamiento y prueba, o se pueden utilizar datos de prueba preexistentes si se proporciona un ensemble preentrenado. Se utilizan varios modelos (LSTM, RandomForest, XGBoost, LightGBM) para predecir la demanda en diferentes distritos de Nueva York. Luego, se genera un ensemble ponderado utilizando los resultados de cada modelo. Los errores de predicción se calculan para cada distrito y se devuelven como parte de los resultados.

    Retorna:
    - ensemble (dict): Diccionario que contiene el ensemble de modelos entrenados.
    - ensemble_predictions (dict): Diccionario que contiene las predicciones del ensemble para cada distrito.
    - ensemble_errors (dict): Diccionario que contiene los errores de predicción del ensemble para cada distrito.
    - ponderacion (dict): Diccionario que contiene la mejor ponderacion de modelos para efectuar predicciones.
    '''

    if (verbose==1) and (ensemble==None):
        print('Comienzo del entrenamiento del ensemble')
    if (verbose==1) and (ensemble != None):
        print('Comienzo del testeo del ensemble')

    columnas_X = ['año', 'mes', 'dia', 'hora', 'dia_semana', 'holiday', 'temperature_2m', 'rain', 'relative_humidity_2m', 'snowfall']
    columnas_Y = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
    
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

    if ensemble == None:
        ensemble = generar_ensemble(train_data, exportar=exportar, file=file, p=nro_pasos, verbose=verbose)
    
    ensemble_predictions = {}
    ensemble_errors = {}
    ponderaciones = ['lin', 'exp', 'inv']
    alphas = [0.001, 0.01, 0.1, 0.5]

    for ponderacion in ponderaciones:

        ensemble_errors[ponderacion] = {}
        ensemble_predictions[ponderacion] = {}

        if ponderacion != 'exp':
            if verbose == 1:
                print('Ponderacion:', ponderacion)

            test_pred_ensemble = predecir(ensemble, test=True, test_data=test_data[columnas_X], verbose=verbose, ponderacion=ponderacion)

            for district in columnas_Y:
                y_pred = test_pred_ensemble[district]['ensemble']
                y_test = test_data[district].iloc[nro_pasos-1:-1]
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                nrmse = rmse / y_test.mean()

                ensemble_errors[ponderacion][district] = {'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse}

                if verbose==1:
                    print('Distrito:', district)
                    print('RMSE:', rmse)

                predictions_df = pd.DataFrame({
                    'Predicted': y_pred,
                    'Real values': y_test
                }, index=y_test.index)

                ensemble_predictions[ponderacion][district] = predictions_df

        if ponderacion == 'exp':
            for alpha in alphas:
                if verbose == 1:
                    print('Ponderacion:', ponderacion, 'Alpha:', alpha)
                
                ensemble_errors[ponderacion][alpha] = {}
                ensemble_predictions[ponderacion][alpha] = {}

                test_pred_ensemble = predecir(ensemble, test=True, test_data=test_data[columnas_X], verbose=verbose, 
                                              ponderacion=ponderacion, alpha=alpha)

                for district in columnas_Y:
                    y_pred = test_pred_ensemble[district]['ensemble']
                    y_test = test_data[district].iloc[nro_pasos-1:-1]
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    nrmse = rmse / y_test.mean()

                    ensemble_errors[ponderacion][alpha][district] = {'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse}

                    if verbose==1:
                        print('Distrito:', district)
                        print('RMSE:', rmse)

                    predictions_df = pd.DataFrame({
                        'Predicted': y_pred,
                        'Real values': y_test
                    }, index=y_test.index)

                    ensemble_predictions[ponderacion][alpha][district] = predictions_df
    
    nrmse_dict = {}
    lin_nrmse = np.mean([ensemble_errors['lin'][district]['NRMSE'] for district in columnas_Y])
    nrmse_dict['lin'] = lin_nrmse
    inv_nrmse = np.mean([ensemble_errors['inv'][district]['NRMSE'] for district in columnas_Y])
    nrmse_dict['inv'] = inv_nrmse
    best_nrmse = np.inf
    for alpha in alphas:
        exp_nrmse = np.mean([ensemble_errors['exp'][alpha][district]['NRMSE'] for district in columnas_Y])
        if exp_nrmse < best_nrmse:
            best_nrmse = exp_nrmse
            best_alpha = alpha
            nrmse_dict['exp'] = exp_nrmse

    best_pond = min(nrmse_dict, key=nrmse_dict.get)

    if (verbose==1):
        print('Finalizado el testeo del ensemble')
        print('Mejor ponderación:', best_pond)
        print('Mejor alpha:', best_alpha)
        print('Errores', ensemble_errors[best_pond])

    ponderacion = {}

    ponderacion['tipo'] = best_pond
    ponderacion['alpha'] = best_alpha

    return ensemble, ensemble_predictions, ensemble_errors, ponderacion

def train_ensemble(data, verbose=0, test_size=0.2, nro_pasos=5, exportar='si', file='ensemble.joblib'):
    '''
    Entrena un ensemble de modelos de machine learning.

    Argumentos:
    - data (DataFrame): DataFrame que contiene los datos a utilizar.
    - verbose (int): Nivel de detalle de los mensajes de progreso (0 para no mostrar mensajes, 1 para mostrar).
    - test_size (float): Proporción de datos a utilizar como conjunto de prueba.
    - nro_pasos (int): Número de pasos hacia atrás a considerar para la predicción de cada modelo.
    - exportar (str): Opción para exportar el ensemble entrenado ('si' para exportar, 'no' para no exportar).
    - file (str): Nombre del archivo para exportar el ensemble.

    Descripción detallada:
    Esta función entrena un ensemble de modelos de machine learning utilizando un conjunto de datos. Utiliza la función train_test_ensemble para realizar el entrenamiento y devolver el ensemble entrenado junto con las predicciones y errores asociados.

    Retorna:
    - ensemble (dict): Diccionario que contiene el ensemble de modelos entrenados.
    - ensemble_predictions (dict): Diccionario que contiene las predicciones del ensemble para cada distrito.
    - ensemble_errors (dict): Diccionario que contiene los errores de predicción del ensemble para cada distrito.
    - ponderacion (dict): Diccionario que contiene la mejor ponderacion de modelos para efectuar predicciones.
    '''
    
    ensemble, ensemble_predictions, ensemble_errors, ponderacion = train_test_ensemble(data=data, verbose=verbose, test_size=test_size,
                                                                            nro_pasos=nro_pasos, exportar=exportar, file=file
                                                                            )
    
    return ensemble, ensemble_predictions, ensemble_errors, ponderacion

def test_ensemble(data, ensemble, verbose=0, test_size=0.2, nro_pasos=5, exportar='no', file='ensemble.joblib'):
    '''
    Realiza pruebas del ensemble de modelos de machine learning.

    Argumentos:
    - data (DataFrame): DataFrame que contiene los datos a utilizar para las pruebas.
    - ensemble (dict): Diccionario que contiene el ensemble de modelos previamente entrenados.
    - verbose (int): Nivel de detalle de los mensajes de progreso (0 para no mostrar mensajes, 1 para mostrar).
    - test_size (float): Proporción de datos a utilizar como conjunto de prueba.
    - nro_pasos (int): Número de pasos hacia atrás a considerar para la predicción en modelo LSTM.
    - exportar (str): Opción para exportar los resultados de las pruebas ('si' para exportar, 'no' para no exportar).
    - file (str): Nombre del archivo para exportar los resultados de las pruebas.

    Descripción detallada:
    Esta función realiza pruebas del ensemble de modelos de machine learning utilizando un conjunto de datos de prueba. Utiliza la función train_test_ensemble para realizar las pruebas y devuelve el ensemble junto con las predicciones y errores asociados.

    Retorna:
    - ensemble_predictions (dict): Diccionario que contiene las predicciones del ensemble para cada distrito.
    - ensemble_errors (dict): Diccionario que contiene los errores de predicción del ensemble para cada distrito.
    - ponderacion (dict): Diccionario que contiene la mejor ponderacion de modelos para efectuar predicciones.
    '''
    
    _, ensemble_predictions, ensemble_errors, ponderacion = train_test_ensemble(data=data, verbose=verbose, test_size=test_size,
                                                                            nro_pasos=nro_pasos, exportar=exportar, file=file, ensemble=ensemble
                                                                            )
    return ensemble_predictions, ensemble_errors, ponderacion

def graficar_pred_ensemble(pred, numero_dias=5):
    
    for district in pred.keys():
        plt.figure(figsize=(12,12))
        sns.set_style('darkgrid')
        sns.lineplot(pred[district]['Predicted'][:24*numero_dias], label='Predicted values', color='blue')
        sns.lineplot(pred[district]['Real values'][:24*numero_dias], label='Real values', color='green')
        plt.title(f'Pedicciones de demanda en {district} utilizando Weighted Ensemble', fontsize=14)
        plt.xlabel('Fecha-hora', fontsize=12)
        plt.ylabel('Demanda', fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()


def graficar_predicciones_modelos(ensemble, numero_dias=5):
    '''
    Grafica las predicciones del set de testeo del entrenamiento de los distintos modelos correspondeintes al Ensemble.
    '''
    pred_modelos = ensemble['test_predictions']
    districts = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
    for district in districts:
            
            plt.figure(figsize=(12,12))
            sns.set_style('darkgrid')
            sns.lineplot(pred_modelos['lstm'][district]['Predicted'][:numero_dias*24], label='LSTM', color='red')
            sns.lineplot(pred_modelos['rf'][district]['Predicted'][:numero_dias*24], label='RF', color='green')
            sns.lineplot(pred_modelos['xgb'][district]['Predicted'][:numero_dias*24], label='XGB', color='blue')
            sns.lineplot(pred_modelos['lgbm'][district]['Predicted'][:numero_dias*24], label='LGBM', color='orange')
            sns.lineplot(pred_modelos['lstm'][district]['Real values'][:numero_dias*24], label= 'Valores reales', color= 'black')
            plt.title(f'Pedicciones en {district} de los diferentes modelos en el set de testeo', fontsize=14)
            plt.xlabel('Fecha-hora', fontsize=12)
            plt.ylabel('Demanda', fontsize=12)
            plt.legend(title='Modelo', fontsize=10, title_fontsize='12')
            plt.tight_layout()
            plt.show()

def exportar_modelo(ensemble, ensemble_predictions, ensemble_errors, ponderacion, archivo= 'modelo.joblib'):

    modelo = {'ensemble': ensemble, 'ensemble_predictions': ensemble_predictions, 'ensemble_errors': ensemble_errors, 'ponderacion': ponderacion}
    dump(modelo, archivo)

    print('Modelo exportado:', archivo)

def graficar_feature_importances(ensemble):
    districts = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
    columnas_X = ['año', 'mes', 'dia', 'hora', 'dia_semana', 'holiday', 'temperature_2m', 'rain', 'relative_humidity_2m', 'snowfall']
    models = ['rf', 'xgb', 'lgbm']
    for model in models:
        for district in districts:
            temp_model = ensemble['models'][model][district]
            feature_importances = pd.Series(temp_model.feature_importances_, index=columnas_X)
            feature_importances.nlargest(10).plot(kind='barh')
            plt.title(f'Feature Importances del modelo {model} en el distrito {district} ')
            plt.show()

def obtener_mejores_ponderaciones(ensemble_errors):
    '''
    Obtiene las mejores ponderaciones por distrito basadas en los errores RMSE de cada método de ponderación.

    Argumentos:
    - ensemble_errors (dict): Diccionario que contiene los errores RMSE de los modelos LSTM, LightGBM, XGBoost, RandomForest y el ensemble ponderado para cada distrito.
        El diccionario tiene las siguientes claves:
            - 'lin': Errores RMSE del método de ponderación lineal para cada distrito.
            - 'inv': Errores RMSE del método de ponderación inversa del error para cada distrito.
            - 'exp': Errores RMSE del método de ponderación exponencial para cada distrito, organizados por distintos valores de alpha.
                Ejemplo: ensemble_errors['exp'][0.001]['Bronx'] devuelve el RMSE del distrito Bronx utilizando un alpha de 0.001.

    Retorna:
    - mejores_pond_por_distrito (dict): Diccionario que contiene las mejores ponderaciones por distrito.
        Cada clave es un distrito y cada valor es una tupla que indica la mejor ponderación y su correspondiente RMSE.
        Ejemplo: {'Bronx': ('lin', 10.5), 'Brooklyn': ('inv', 9.8), 'Manhattan': ('exp', 0.01, 8.3)}
            - Para el caso de ponderación exponencial, la tupla incluye también el valor de alpha.
    '''
    mejores_pond_por_distrito = {}
    alphas = [0.001, 0.01, 0.1, 0.5]
    districts = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']

    for district in districts:
        lin_rmse = ensemble_errors['lin'][district]['RMSE']
        inv_rmse = ensemble_errors['inv'][district]['RMSE']
        best_exp_rmse = np.inf

        for alpha in alphas:
            exp_rmse = ensemble_errors['exp'][alpha][district]['RMSE']
            if exp_rmse < best_exp_rmse:
                best_exp_rmse = exp_rmse
                best_alpha = alpha
        
        if lin_rmse < best_exp_rmse and lin_rmse < inv_rmse:
            mejores_pond_por_distrito[district] = ('lin', lin_rmse)
        elif inv_rmse < best_exp_rmse and inv_rmse < lin_rmse:
            mejores_pond_por_distrito[district] = ('inv', inv_rmse)
        else:
            mejores_pond_por_distrito[district] = ('exp', best_alpha, best_exp_rmse)
    
    return mejores_pond_por_distrito

def comparar_con_media(preds, hist_means, umbral=10):
    '''
    Compara las predicciones de demanda con la media histórica para ese día de la semana y esa hora. Si supera el umbral asignado, se señala como alta demanda.
    Argumentos:
    - preds (dict): Diccionario que contiene para cada distrito, las predicciones de demanda (dataframes con columnas para los modelos y el ensemble).
    - hist_means (Dataframe): Dataframe de pandas que contiene las medias históricas para cada distrito, por día de la semana y hora.
    - umbral (int): indica el porcentaje por encima de la media que debe ser superado por la demanda predicha para ser considerado alta demanda.

    Retorna:
    - alta_demanda (dict): Diccionario que contiene para cada distrito, una serie con los días y horas de alta demanda según el umbral determinado
    '''
    distritos = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
    alta_demanda = {}

    for district in distritos:
        pred_dist = preds[district].reset_index()
        pred_dist['dia_semana'] = pred_dist['datetime'].dt.weekday +1
        pred_dist['hora'] = pred_dist['datetime'].dt.hour
        pred_dist.set_index('datetime', inplace=True)
        pred_dist = pred_dist[['dia_semana', 'hora', 'ensemble']]
        result = pd.merge(pred_dist, hist_means[['dia_semana', 'hora', district]], how= 'left', on=['dia_semana', 'hora'])
        result.index = pred_dist.index
        result['alta_demanda'] = result['ensemble'] > (1+umbral/100)*result[district]
        result = result[result['alta_demanda']]
        alta_demanda[district] = result
    
    return alta_demanda