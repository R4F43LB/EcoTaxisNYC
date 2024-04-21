import streamlit as st
import os


import streamlit as st

def main():
    # Agregar título centrado verticalmente
    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 10vh;">
            <h1>Proyecto EcoTaxisNYC</h1>
        </div>
    """, unsafe_allow_html=True)
        # Obtener la ruta absoluta del directorio actual y agregar el nombre del archivo de imagen
    logo_path = os.path.join(os.path.dirname(__file__), "Asset\Logo.png")
    st.image(logo_path)


       
    # Introducción
    st.header("Introducción")
    st.write("El proyecto EcoTaxisNYC tiene como misión ser parte de la transformación del transporte en la ciudad de Nueva York mediante la introducción de una flota de taxis sin emisiones.")
    st.write("Buscamos mejorar la eficiencia del transporte y reducir significativamente tanto la huella de carbono como la contaminación sonora, apoyando la visión de una ciudad más verde y sostenible.")
    st.write("Advisors on the sustainable Economic Transition (ASET Company) es una empresa líder en asesoramiento estratégico que ayuda a las empresas a navegar y prosperar en la transición hacia una economía sostenible.")
    st.write("Nuestro enfoque se basa en datos, brindando soluciones personalizadas respaldadas por análisis exhaustivos para impulsar la eficiencia, la innovación y el éxito a largo plazo en un mundo en evolución hacia la sostenibilidad.")

if __name__ == "__main__":
    main()

st.markdown("***")
st.markdown("## Contenido")
st.markdown("### ⏫ [Cargar Datos](Cargar_Datos)")
st.markdown("### 📈 [Dashboard](Dashboard)")