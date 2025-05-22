import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm import HuggingFaceLLM

# Inicializar modelo
llm = HuggingFaceLLM(model="codellama/CodeLlama-7b-Instruct-hf")

st.title("App análisis con base de datos fija")

# Cargar CSV local dentro del repo (o ruta relativa)
df = pd.read_csv("data/datos.csv")

st.subheader("Datos cargados:")
st.dataframe(df.head())

metadata = {
    "ventas": "Ventas mensuales en pesos",
    "mes": "Mes del año",
    "region": "Región geográfica"
}

sdf = SmartDataframe(df, config={"llm": llm, "metadata": metadata})

# Entrada para preguntas en lenguaje natural
pregunta = st.text_input("Escribe tu pregunta o comando:")

if pregunta:
    result = sdf.chat(pregunta)
    if isinstance(result, pd.DataFrame):
        st.dataframe(result)
    elif isinstance(result, plt.Figure):
        st.pyplot(result)
    else:
        st.write(result)
