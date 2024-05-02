from utils.ensemble_functions import *
import streamlit as st
import plotly.graph_objs as go
import os
import tarfile
import keras
from datetime import datetime

def main():

    # Inicialización de variables de sesión para almacenar el estado
    if 'predicciones' not in st.session_state:
        st.session_state['predicciones'] = None
    
    # Ruta del archivo ensemble_1_complete.tar.gz
    ensemble_path = os.path.join(os.path.dirname(__file__), '../../data/ensemble_2_sin_lstm.tar.gz')

    # Verificación de existencia del archivo
    if os.path.exists(ensemble_path):
        print(f"Archivo {ensemble_path} encontrado en Streamlit Sharing.")
    else:
        print(f"Error: Archivo {ensemble_path} no encontrado en Streamlit Sharing.")

    ensemble_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/ensemble_2_sin_lstm.tar.gz'))
    print('Dirección del ensemble: ',ensemble_path)
    # Directorio de extracción
    extracted_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/'))
    print('Dirección de extracción', extracted_dir)
    # Descomprimir el archivo tar.gz
    with tarfile.open(ensemble_path, 'r:gz') as tar:
        tar.extractall(path=extracted_dir)
    # Verificación de existencia del archivo extraído
    extracted_joblib_path = os.path.join(extracted_dir, 'ensemble_2_sin_lstm.joblib')
    if os.path.exists(extracted_joblib_path):
        print(f"Archivo {extracted_joblib_path} extraído correctamente.")
    else:
        print(f"Error: Archivo {extracted_joblib_path} no encontrado tras la extracción.")
    # Cargar el modelo desde el archivo descomprimido
    model_path = os.path.join(extracted_dir, 'ensemble_2_sin_lstm.joblib')
    print(f"Intentando cargar el modelo desde: {model_path}")
    ensemble_sin_lstm = load(model_path)
    print('Archivo cargado ensemble_sin_lstm cargado')
    #print('claves del diccionario: ', ensemble_sin_lstm.keys())
    
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Archivo {model_path} eliminado correctamente.")
    else:
        print(f"Error: Archivo {model_path} no encontrado.")

    lstm_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/lstm_model_2.h5'))
    lstm_model = keras.models.load_model(lstm_model_path)
    
    ensemble = ensemble_sin_lstm
    #print(f'claves de ensemble: {ensemble.keys()}')
    ensemble['ensemble']['models']['lstm'] = lstm_model
    ensemble_errors = ensemble['ensemble_errors']
    ensemble = ensemble['ensemble']

    # Agregar el cartel informativo
    st.markdown("""
        <div style="text-align: center;">
            <h2>Bienvenido al panel de predicción de demanda</h2>
        </div>
        """, unsafe_allow_html=True)
    st.write("### Nota:")
    st.write("Los datos de hora corresponden a la hora local en NYC.")

    # Agregar el casillero para el umbral de alerta de demanda
    umbral_alerta_demanda = st.selectbox("Umbral de alerta de demanda (porcentaje sobre la media)", options=[i for i in range(1, 101)] )
    hist_means_path = os.path.join(os.path.dirname(__file__), '../../data/hist_means.parquet')
    hist_means = pd.read_parquet(hist_means_path)

    def generar_archivo_excel(pred, tipo_prediccion):
    # Crear un DataFrame con las predicciones
        dfs = []
        for district, preds in pred.items():
            if tipo_prediccion == 'Ensemble':
                values = preds['ensemble']
                df = pd.DataFrame(values, columns=['ensemble'])
                df.rename(columns={'ensemble': 'Demanda'}, inplace=True)
            else:
                df = pd.DataFrame()
                for model, values in preds.items():
                    if model != 'ensemble':
                        df[model] = values
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Fecha'
            df.columns.name = 'Distrito'
            dfs.append((district, df))

        # Formatear el DataFrame para Excel
        with pd.ExcelWriter('predicciones.xlsx') as writer:
            for district, df in dfs:
                df.to_excel(writer, sheet_name=f'Predicciones_{district}')

        # Leer el contenido del archivo Excel generado
        with open('predicciones.xlsx', 'rb') as f:
            contenido = f.read()
            
        excel_filename = 'predicciones.xlsx'
        if os.path.exists(excel_filename):
            os.remove(excel_filename)
            print(f"Archivo {excel_filename} eliminado correctamente.")

        return contenido

    def graficar_predicciones_interactivas(pred, tipo_prediccion, generar_excel=False):
        '''
        Grafica las predicciones realizadas por el modelo de ensemble de manera interactiva.

        Argumentos:
        - pred (dict): Diccionario que contiene los DataFrames de predicciones para cada distrito.
        - tipo_prediccion (str): Tipo de predicción a mostrar ('Ensemble' o 'Modelos')
        - generar_excel (bool): Permite al usuario descargar excel con las predicciones. Predeterminado en False.

        Descripción detallada:
        La función recibe un diccionario de DataFrames de predicciones para cada distrito y grafica las predicciones realizadas por los modelos LSTM, RandomForest, XGBoost, LightGBM y el ensemble ponderado de manera interactiva.
        Cada gráfico muestra las predicciones de cada modelo por separado y la predicción del ensemble.
        '''
        if generar_excel:
            # Generar archivo Excel
            contenido_excel = generar_archivo_excel(pred, tipo_prediccion)

            # Mostrar el botón de descarga
            st.download_button(
                label='Descargar Predicciones',
                data=contenido_excel,
                file_name='predicciones.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

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

        try:
            datetime_solicitada = datetime.strptime(dia_seleccionado + ' ' + hora_seleccionada, '%Y-%m-%d %H:%M')
        except ValueError as e:
            st.error('Formato de fecha y hora inválido. Asegúrate de que esté en el formato "YYYY-MM-DD HH:MM".')
            raise e
        demanda = {}
        data_found = False  # Flag para verificar si se encontraron datos

        for district, preds in pred.items():
            # Comprobación utilizando datetime en el índice
            if datetime_solicitada in preds.index:
                demanda[district] = preds.loc[datetime_solicitada, 'ensemble']
                data_found = True
            else:
                demanda[district] = None  # indica falta de datos

        if not data_found:
            st.error('No hay datos de demanda para la fecha y hora seleccionados')

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
        best_preds = obtener_mejores_ponderaciones(ensemble_errors=ensemble_errors)
        predicciones = predecir(ensemble, cant_dias=dias_prediccion, multiple=True, mejores_pond_por_distrito=best_preds)
        # Actualizar la variable de sesión con las nuevas predicciones
        st.session_state['predicciones'] = predicciones
        # Llamar a la función para graficar las predicciones de manera interactiva

        graficar_predicciones_interactivas(predicciones, tipo_prediccion)

        contenido_excel = generar_archivo_excel(st.session_state['predicciones'], tipo_prediccion)

        st.download_button(
            label='Descargar Predicciones',
            data=contenido_excel,
            file_name='predicciones.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        alertas_demanda = comparar_con_media(st.session_state['predicciones'], hist_means, umbral=umbral_alerta_demanda)
        
        # Mostrar los resultados en una tabla
        for district, data in alertas_demanda.items():
            if not data.empty:
                st.write(f"### Alta demanda en {district}:")
                st.dataframe(data)
            else:
                st.write(f"No hay alta demanda predicha en {district} con el umbral seleccionado.")
    # Verificar si ya existen predicciones antes de mostrar los selectores de fecha y hora
    if st.session_state['predicciones'] is not None:

        #if st.button('Comparar demanda con media histórica'):
        
            #alertas_demanda = comparar_con_media(st.session_state['predicciones'], hist_means, umbral=umbral_alerta_demanda)
        
            # Mostrar los resultados en una tabla
            #for district, data in alertas_demanda.items():
             #   if not data.empty:
              #      st.write(f"### Alta demanda en {district}:")
               #     st.dataframe(data)
                #else:
                 #   st.write(f"No hay alta demanda predicha en {district} con el umbral seleccionado.")


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