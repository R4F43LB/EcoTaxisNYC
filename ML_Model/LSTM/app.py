import streamlit as st
from joblib import load
import numpy as np

# Cargar el modelo de Joblib
modelo = load('test_model.joblib')
scaler_X = load('scaler_X.joblib')
scaler_Y = load('scaler_Y.joblib')

# Definir la interfaz de usuario
st.title('Aplicación de predicción de demanda de Taxis en NYC con modelo de ML')

# Agregar widgets para la entrada de datos
feature_1 = st.slider('Año', min_value=2023, max_value=9999, step=1)
feature_2 = st.slider('Mes', min_value=1, max_value=12, step=1)
feature_3 = st.slider('Dia', min_value=1, max_value=31, step=1)
feature_4 = st.slider('Hora', min_value=0, max_value=23, step=1)
feature_5 = st.slider('Temperatura', min_value=-40.0, max_value=50.0, step=0.01)
feature_6 = st.slider('Lluvia', min_value=0.0, max_value=100.0, step=0.01)
feature_7 = st.slider('Humedad', min_value=0.0, max_value=100.0, step=0.01)
feature_8 = st.slider('Nieve', min_value=0.0, max_value=100.0, step=0.01)

# Realizar la predicción
x = np.array([feature_1,	feature_2,	feature_3,	feature_4,	feature_5,	feature_6,	feature_7,	feature_8]).reshape((1, 8))
x_scaled = scaler_X.transform(x)
x_reshaped = x_scaled.reshape((1, 1, len(x_scaled[0])))

y_pred = modelo.predict(x_reshaped)
y_pred_scaled = scaler_Y.inverse_transform(y_pred)

# Mostrar el resultado
st.subheader('Predicción de demanda para cada barrio:')
for i, barrio in enumerate(['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']):
    st.write(f'{barrio}: {int(y_pred_scaled[0][i])}')