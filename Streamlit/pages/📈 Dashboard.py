import streamlit as st

def main():
    st.title("Dashboard (Análisis de Datos):")
    
    # URL del informe de Power BI
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=d4d11ab5-1db8-43f8-8824-7124d0b944ee&autoAuth=true&ctid=17a02ccd-b8b8-4219-aba9-c06798c13ecd"
    
    # Incrustar el informe de Power BI utilizando un iframe
    st.components.v1.iframe(power_bi_url, width=900, height=550)

if __name__ == "__main__":
    main()

