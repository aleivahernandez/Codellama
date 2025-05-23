import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm import HuggingFaceLLM

# --- Cargar base de datos principal ---
df = pd.read_csv("data/datos.csv")

# --- Cargar diccionario de datos y construir metadata ---
diccionario_df = pd.read_csv("data/diccionario.csv")

# Crear el diccionario de metadata desde el CSV
metadata = dict(zip(diccionario_df["Campo"], diccionario_df["Descripción"]))

# --- Inicializar LLM y SmartDataframe ---
llm = HuggingFaceLLM(model="codellama/CodeLlama-7b-Instruct-hf")
sdf = SmartDataframe(df, config={"llm": llm, "metadata": metadata})

# --- Interfaz Streamlit ---
st.title("📊 Análisis automático con lenguaje natural")

st.subheader("Base de datos cargada:")
st.dataframe(df.head())

st.subheader("Diccionario de datos:")
st.dataframe(diccionario_df)

pregunta = st.text_input("Haz tu pregunta sobre los datos:")

if pregunta:
    st.info(f"Procesando: {pregunta}")
    result = sdf.chat(pregunta)
    if isinstance(result, pd.DataFrame):
        st.dataframe(result)
    elif isinstance(result, plt.Figure):
        st.pyplot(result)
    else:
        st.write(result)
