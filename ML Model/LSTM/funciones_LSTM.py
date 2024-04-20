import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
from keras.utils import plot_model
import time

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

def construir_modelo_cv(X, Y, nro_pasos=10, nro_capas_lstm=1, unidades_capa= [50], loss_= 'mse', act_salida='linear', act_lstm= 'relu', 
                     epochs= 50, batch= 32, drop_out= 0, scaler= 'minmax', learning_r=0.001, nro_capas_dense=0, unidades_dense=[0], 
                     act_dense= 'linear'):
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
    - X_test_reshaped: Array numpy de datos de entrada transformado para pruebas.
    - Y_test_reshaped: Array numpy de datos de salida transformado para pruebas.
    - scores: Lista de puntajes de error cuadrático negativo obtenidos durante la validación cruzada.
    - history: Objeto que contiene el historial de entrenamiento del modelo.
    - loss: Valor de la pérdida final obtenido en el conjunto de pruebas.
    '''
    
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
        loss = modelo.evaluate(X_test_reshaped, Y_test_reshaped)
    else:
        y_pred = modelo.predict(X_test_reshaped)
        loss = np.mean(tf.keras.losses.mean_squared_error(Y_test_reshaped, y_pred).numpy())
    
    Y_predict = modelo.predict(X_test_reshaped)
    Y_predict_df = pd.DataFrame(Y_predict, columns=Y.columns)
    Y_predict_original = scaler_Y.inverse_transform(Y_predict_df)
    Y_predict_original_df = pd.DataFrame(Y_predict_original, columns= Y.columns)

    rmse_dict = {}
    for district in Y.columns:
        y_pred = Y_predict_original_df[district]
        y_test = pd.DataFrame(scaler_Y.inverse_transform(Y_test_reshaped), columns=Y.columns)[district]
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        rmse_dict[district] = rmse


    return modelo, scaler_X, scaler_Y, X_test_reshaped, Y_test_reshaped, scores, history, loss, rmse_dict

def evaluar_modelos(X, Y, nro_pasos=25, nro_capas_lstm=[1], unidades_capa= [50], loss_= ['mse', 'mae'], act_salida=['linear'], act_lstm= ['relu'],
                    nro_capas_dense=0, unidades_dense=[0], act_dense= ['linear'], drop_out=0, scaler='minmax', csv_file= 'evaluacion.csv'):
    '''
    Evalúa múltiples configuraciones de modelos de red neuronal utilizando la técnica de validación cruzada y devuelve un DataFrame con los resultados.

    Argumentos:
    - X: Array numpy que contiene los datos de entrada.
    - Y: Array numpy que contiene los datos de salida.
    - nro_pasos: Número entero que indica la cantidad de pasos de tiempo a considerar para la secuencia de entrada. Por defecto, se establece en 25.
    - nro_capas_lstm: Lista de enteros que especifica la cantidad de capas LSTM que se deben probar en cada configuración del modelo. Por defecto, se establece en [1].
    - unidades_capa: Lista de enteros que especifica el número de unidades en cada capa LSTM a probar en cada configuración del modelo. Por defecto, se establece en [50].
    - loss_: Lista de cadenas que indica las funciones de pérdida a probar en cada configuración del modelo. Por defecto, se establece en ['mse', 'mae'].
    - act_salida: Lista de cadenas que especifica las funciones de activación a probar en la capa de salida del modelo en cada configuración. Por defecto, se establece en ['linear'].
    - act_lstm: Lista de cadenas que indica las funciones de activación a probar en las capas LSTM del modelo en cada configuración. Por defecto, se establece en ['relu'].
    - nro_capas_dense: Número entero que indica la cantidad máxima de capas densas a probar en cada configuración del modelo. Por defecto, se establece en 0.
    - unidades_dense: Lista de enteros que especifica el número de unidades en cada capa densa a probar en cada configuración del modelo. Por defecto, se establece en [0].
    - act_dense: Lista de cadenas que especifica las funciones de activación a probar en las capas densas del modelo en cada configuración. Por defecto, se establece en ['linear'].
    - drop_out: Valor flotante que representa la tasa de abandono para las capas Dropout, utilizadas para regularizar el modelo y prevenir el sobreajuste. Por defecto, se establece en 0.
    - scaler: Cadena que indica el método de escalado a aplicar a los datos. Puede ser 'minmax' para escalamiento Min-Max o 'standard' para escalamiento estándar. Por defecto, se establece en 'minmax'.
    - csv_file: Nombre del archivo donde se guardarán los datos de la evaluación. Por defecto 'evaluacion.csv'

    Devuelve:
    - evaluacion: DataFrame que contiene los resultados de la evaluación de los modelos, incluyendo métricas de rendimiento, tiempos de ejecución y configuraciones de los modelos probados.
    '''

    columnas = ['combinacion','nro_pasos', 'nro_capas_lstm', 'unidades_capa', 'nro_capas_dense', 'unidades_dense',  
                'loss_function', 'act_salida', 'act_lstm', 'act_dense', 'loss_value(mse)', 'cv_scores_mean', 'tiempo(s)']

    evaluacion = pd.DataFrame(columns=columnas)
    i = 0
    if nro_capas_dense > 0:
        combinaciones = nro_pasos*len(nro_capas_lstm)*len(loss_)*len(act_salida)*len(act_lstm)*(nro_capas_dense+1)*len(act_dense)
    if nro_capas_dense == 0:
        combinaciones = nro_pasos*len(nro_capas_lstm)*len(loss_)*len(act_salida)*len(act_lstm)*len(act_dense)
    mejor_score = float('inf')
    mejor_combinacion = 0
    for pasos in range(1,nro_pasos+1):
        for capas in nro_capas_lstm:
            for loss_f in loss_:
                for act_s in act_salida:
                    for act_l in act_lstm:
                        for capas_dense in range(nro_capas_dense+1):
                            for act_d in act_dense:
                                start_time = time.time()
                                _, _, _, _, _, scores, _, loss = construir_modelo_cv(X, Y, pasos, capas, unidades_capa, loss_f, act_s, act_l,
                                                                                    drop_out=drop_out, nro_capas_dense=capas_dense, 
                                                                                    unidades_dense=unidades_dense, act_dense=act_d,
                                                                                    scaler=scaler)
                                end_time = time.time()
                                tiempo = end_time - start_time
                                unidades_capa_str = ', '.join(map(str, unidades_capa[0:capas]))
                                unidades_dense_str = ', '.join(map(str, unidades_dense[0:capas]))
                                scores_mean = np.mean(-scores)
                                evaluacion.loc[i] = [i+1, pasos, capas, unidades_capa_str, capas_dense, unidades_dense_str,
                                                    loss_f, act_s, act_l, act_d, loss, scores_mean, tiempo]
                                evaluacion.tail(1).to_csv(csv_file, index=False, mode='a', header=not i)
                                tiempo_promedio = np.mean(evaluacion['tiempo(s)'])
                                i+=1
                                if scores_mean < mejor_score:
                                    mejor_combinacion = i
                                    mejor_score = scores_mean
                                print(f'Finalizada combinación {i} de {combinaciones}. Score= {np.round(scores_mean, 3)}')
                                print(f'Mejor combinación: {mejor_combinacion}. Score: {np.round(mejor_score, 3)}')
                                tiempo_restante = tiempo_promedio * (combinaciones - i) / 60
                                print(f'Quedan {round(tiempo_restante, 2)} minutos.')
    
    return evaluacion

def graficar_curva_aprendizaje(history):
    '''
    Genera y muestra la curva de aprendizaje del modelo utilizando el historial de entrenamiento.

    Argumentos:
    - history: Objeto que contiene el historial de entrenamiento del modelo, incluyendo la pérdida en el conjunto de entrenamiento y validación en cada época.

    Acciones:
    - Extrae los valores de la pérdida del historial de entrenamiento y validación.
    - Grafica la curva de aprendizaje con la pérdida en el conjunto de entrenamiento y validación a lo largo de las épocas.
    - Muestra la gráfica.

    La curva de aprendizaje proporciona información sobre cómo la pérdida del modelo cambia a medida que avanza el entrenamiento, lo que puede ayudar a evaluar el rendimiento y la convergencia del modelo.
    '''

    # Extraemos los valores de la función de pérdida del historial de entrenamiento
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    final_loss = history.history['val_loss'][-1]
    print("Valor final de Loss:", final_loss)
    
    # Graficamos la curva de aprendizaje
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Curva de Aprendizaje')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def graficar_pred_test(X_test_reshaped, Y_test_reshaped, modelo, scaler_Y):
    '''
    Genera y muestra gráficas de las predicciones del modelo en comparación con los valores reales en el conjunto de prueba.

    Argumentos:
    - X_test_reshaped: Datos de entrada del conjunto de prueba, con formato 3D (número de muestras, número de pasos de tiempo, número de características).
    - Y_test_reshaped: Valores de salida reales del conjunto de prueba, con formato 2D (número de muestras, número de características).
    - modelo: Modelo entrenado utilizado para hacer predicciones.
    - scaler_Y: Objeto scaler utilizado para invertir la transformación aplicada a los valores de salida durante la normalización.

    Acciones:
    - Convierte los datos de salida del conjunto de prueba y las predicciones del modelo de nuevo a su escala original utilizando el scaler_Y.
    - Grafica las predicciones del modelo (línea de predicción) y los valores reales (línea real) para cada característica en el conjunto de prueba.
    - Muestra las gráficas para cada característica en el conjunto de prueba.

    Estas gráficas son útiles para visualizar visualmente cómo se comparan las predicciones del modelo con los valores reales en el conjunto de prueba, lo que permite evaluar el rendimiento del modelo.
    '''

    columnas_Y = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
    Y_test_df = pd.DataFrame(Y_test_reshaped, columns=columnas_Y)
    Y_test_original = scaler_Y.inverse_transform(Y_test_df)
    Y_test_original_df = pd.DataFrame(Y_test_original, columns=columnas_Y)

    Y_predict = modelo.predict(X_test_reshaped)
    Y_predict_df = pd.DataFrame(Y_predict, columns=columnas_Y)
    Y_predict_original = scaler_Y.inverse_transform(Y_predict_df)
    Y_predict_original_df = pd.DataFrame(Y_predict_original, columns=columnas_Y)

    for i in columnas_Y:
        sns.lineplot(x=Y_predict_original_df.index, y=Y_predict_original_df[i], label='Predicción')
        sns.lineplot(x=Y_test_original_df.index, y=Y_test_original_df[i], label= 'Real')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

def graficar_modelo(modelo, archivo= 'model_plot.png'):
    plot_model(modelo, to_file=archivo, show_shapes=True, show_layer_names=True, rankdir='TB')

def generador_X(X, p=5):
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