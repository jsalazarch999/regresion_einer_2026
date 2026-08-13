# Debe direccionar VS Code a la carpeta con los archivos:
# 1.- Archivo
# 2.- Abrir carpeta. Debe dar click en la carpeta que contiene los archivos de interés
#3.- A la izquierda, en el explorador deberá poder visualizar todos los archivos
#------------------------------------------------------------------------------------------------

# CÓDIGO STREAMLIT
# Ir a:   Ver/Terminal
# Crea un ambiente virtual (puedes usar otro nombre en lugar de 'venv'): coloca este código
#   python -m venv venv

#---------------------------------------------------------------------------------------
# Luego de crear el ambiente virtual, lo activas
#   .\venv\Scripts\activate   # En Windows
#---------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
# Cuando vuelva a iniciar sesión, debe volver a activar el ambiente virtual, ya no lo debe crear.
# En este caso debes abrir la carpeta con los archivos del caso.
#---------------------------------------------------------------------------------------------

# Instala la versión específica de scikit-learn
#   pip install scikit-learn==1.2.2
# Instala otras dependencias, incluyendo Streamlit
#  pip install streamlit pandas joblib
#-------------------------------------------------------------------------------------------------
# Desde la segunda vez: hacer:
# Si da error, debes ir a PowerShell de Window y:
#      Get-ExecutionPolicy                           Si es Restricted; ejecuta
#      Set-ExecutionPolicy RemoteSigned              Colocar Sí
# En consola de VSC:  .\venv\Scripts\activate

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load


# ============================================================
# CARGAR MODELO
# ============================================================

modelo = load(
    "modelo_ingresos_gradient_boosting.joblib"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Predicción de Ingresos",
    page_icon=":v",
    layout="centered"
)


# ============================================================
# TÍTULO
# ============================================================

st.title("Modelo Predictivo de Ingresos")

st.markdown(
    """
    Este aplicativo utiliza un modelo de **Gradient Boosting**
    para estimar los ingresos de un trabajador a partir de sus
    características sociodemográficas y laborales.
    """
)

st.markdown("---")


# ============================================================
# FORMULARIO
# ============================================================

with st.form("formulario_ingresos"):

    st.subheader("Características del trabajador")

    col1, col2 = st.columns(2)

    with col1:

        empleo = st.selectbox(
            "Tipo de empleo",
            ["Formal", "Informal"]
        )

        sexo = st.selectbox(
            "Sexo",
            ["Hombre", "Mujer"]
        )

        edad = st.number_input(
            "Edad",
            min_value=14,
            max_value=100,
            value=35,
            step=1
        )

        etnia = st.selectbox(
            "Etnia",
            [
                "Mestizo",
                "Afrodescendiente",
                "Blanco",
                "Otro"
            ]
        )

        lengua = st.selectbox(
            "Lengua",
            ["Castellano"]
        )

    with col2:

        area = st.selectbox(
            "Área",
            ["Urbano", "Rural"]
        )

        educacion = st.selectbox(
            "Educación",
            [
                "Primaria",
                "Secundaria",
                "Tecnico",
                "Universidad",
                "Posgrado"
            ]
        )

        ocupacion = st.selectbox(
            "Ocupación",
            [
                "Empleado",
                "Independiente",
                "Empleador",
                "Ayudante"
            ]
        )

        tam_empresa = st.selectbox(
            "Tamaño de empresa",
            [
                "Microempresa",
                "Grande"
            ]
        )

        horas_tr = st.number_input(
            "Horas trabajadas",
            min_value=1,
            max_value=100,
            value=48,
            step=1
        )

    boton = st.form_submit_button(
        "Predecir ingresos"
    )


# ============================================================
# PREDICCIÓN
# ============================================================

if boton:

    datos = pd.DataFrame({
        "Empleo": [empleo],
        "Sexo": [sexo],
        "Edad": [edad],
        "Etnia": [etnia],
        "Lengua": [lengua],
        "Area": [area],
        "Educacion": [educacion],
        "Ocupacion": [ocupacion],
        "Tam_empresa": [tam_empresa],
        "Horas_Tr": [horas_tr]
    })

    # Predicción en escala logarítmica
    ingreso_log = modelo.predict(datos)

    # Regresar a escala original
    ingreso_predicho = np.expm1(ingreso_log)[0]

    # Evitar valores negativos
    ingreso_predicho = max(0, ingreso_predicho)

    st.markdown("---")

    st.subheader("Resultado")

    st.metric(
        "Ingreso mensual estimado",
        f"S/ {ingreso_predicho:,.2f}"
    )