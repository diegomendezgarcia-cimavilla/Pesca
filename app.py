import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("🌊 IA Pesquera Cantábrico")

# Dataset simulado
np.random.seed(42)
temp = np.random.normal(18,2,100)
wave = np.random.normal(1.2,0.5,100)
species = np.random.choice(["pulpo","lubina","percebe"],100)
df = pd.DataFrame({"temp":temp,"wave":wave,"species":species})

# Entrenamiento IA simple
model = RandomForestClassifier()
model.fit(df[["temp","wave"]], df["species"])

# Predicción interactiva
t = st.slider("Temperatura del agua (°C)",10,24,18)
w = st.slider("Altura de ola (m)",0.0,4.0,1.0)
if st.button("Predecir"):
    pred = model.predict([[t,w]])[0]
    st.write(f"🎯 Especie más probable: {pred}")
