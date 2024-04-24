from utils.ensemble_functions import *
import streamlit as st
import plotly.graph_objs as go
import os


def main():

    # Inicialización de variables de sesión para almacenar el estado
    if 'predicciones' not in st.session_state:
        st.session_state['predicciones'] = None

    # Cargamos el modelo de ensemble
    ensemble_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/ensemble_1_complete.joblib'))
    ensemble_completo = load(ensemble_path)
    ensemble = ensemble_completo['ensemble']
    ponderacion = ensemble_completo['ponderacion']

    def graficar_predicciones_interactivas(pred, tipo_prediccion):
        '''
        Grafica las predicciones realizadas por el modelo de ensemble de manera interactiva.

        Argumentos:
        - pred (dict): Diccionario que contiene los DataFrames de predicciones para cada distrito.
        - tipo_prediccion (str): Tipo de predicción a mostrar ('Ensemble' o 'Modelos')

        Descripción detallada:
        La función recibe un diccionario de DataFrames de predicciones para cada distrito y grafica las predicciones realizadas por los modelos LSTM, RandomForest, XGBoost, LightGBM y el ensemble ponderado de manera interactiva.
        Cada gráfico muestra las predicciones de cada modelo por separado y la predicción del ensemble.
        '''
        traces = []  # Lista para almacenar las líneas de cada distrito

        # Crear una línea para cada distrito
        for district, preds in pred.items():
            if tipo_prediccion == 'Ensemble':
                values = preds['ensemble']
                trace = go.Scatter(
                    x=values.index,
                    y=values.values,
                    mode='lines',
                    name=f'{district} - Ensemble'
                )
                traces.append(trace)
            else:  # Mostrar predicciones de los modelos particulares
                for model, values in preds.items():
                    if model != 'ensemble':
                        trace = go.Scatter(
                            x=values.index,
                            y=values.values,
                            mode='lines',
                            name=f'{district} - {model}'
                        )
                        traces.append(trace)

        # Layout del gráfico
        layout = go.Layout(
            title=f'Predicciones de demanda por distrito ({tipo_prediccion})',
            xaxis=dict(title='Fecha-hora'),
            yaxis=dict(title='Demanda'),
            hovermode='closest'
        )

        # Crear la figura
        fig = go.Figure(data=traces, layout=layout)

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig)

    # Función para obtener la demanda para cada distrito en un día y hora específicos
    def obtener_demanda(pred, dia_seleccionado, hora_seleccionada):
        '''
        Obtiene la demanda para cada distrito en un día y hora específicos.

        Argumentos:
        - pred (dict): Diccionario que contiene los DataFrames de predicciones para cada distrito.
        - dia_seleccionado (str): Día seleccionado en formato 'YYYY-MM-DD'.
        - hora_seleccionada (str): Hora seleccionada en formato 'HH:MM'.

        Devuelve:
        - demanda (dict): Diccionario que contiene la demanda para cada distrito en el día y hora seleccionados.
        '''
        datetime_solicitada = str(dia_seleccionado) + ' ' + str(hora_seleccionada) 
        demanda = {}
        for district, preds in pred.items():
            demanda[district] = preds.loc[datetime_solicitada, 'ensemble']
        return demanda

    # Seleccionar el tipo de predicción a mostrar
    tipo_prediccion = st.selectbox("Selecciona el tipo de predicción a mostrar:",
                                options=['Ensemble', 'Modelos'])

    # Seleccionar la cantidad de días hacia adelante
    dias_prediccion = st.selectbox("Selecciona la cantidad de días hacia adelante:",
                                options=[1, 2, 3, 4, 5, 6, 7])

    # Botón para generar los gráficos
    if st.button("Generar predicciones"):
        # Realizar las predicciones
        predicciones = predecir(ensemble, cant_dias=dias_prediccion, ponderacion=ponderacion['tipo'], alpha=ponderacion['alpha'])
        # Actualizar la variable de sesión con las nuevas predicciones
        st.session_state['predicciones'] = predicciones
        # Llamar a la función para graficar las predicciones de manera interactiva
        graficar_predicciones_interactivas(predicciones, tipo_prediccion)

    # Verificar si ya existen predicciones antes de mostrar los selectores de fecha y hora
    if st.session_state['predicciones'] is not None:
        st.write("### Selecciona un día y una hora para ver la demanda:")
        dia_seleccionado = st.date_input("Selecciona un día:", min_value=pd.to_datetime('today').date(), max_value=pd.to_datetime('today').date() + pd.Timedelta(days=dias_prediccion-1))
        
        # Opciones de horas enteras
        horas = [f"{h:02d}:00" for h in range(24)]
        hora_seleccionada = st.selectbox("Selecciona una hora:", horas)
        
        # Obtener la demanda para cada distrito en el día y hora seleccionados
        if st.button("Mostrar demanda"):
            demanda = obtener_demanda(st.session_state['predicciones'], dia_seleccionado.strftime('%Y-%m-%d'), hora_seleccionada)
            # Mostrar la demanda para cada distrito
            st.write("### Demanda por distrito:")
            for district, value in demanda.items():
                st.write(f"- {district}: {value}")

if __name__ == "__main__":
    main()