import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# -----------------------------
# DATASET SIMULADO CANTÁBRICO
# -----------------------------

np.random.seed(42)

dates = pd.date_range(start="2005-01-01", end="2024-12-31", freq="D")
n = len(dates)

temp_water = 14 + 4 * np.sin((dates.dayofyear-80)*2*np.pi/365) + np.random.normal(0,0.8,n)
wave_height = np.clip(np.random.normal(1.2,0.7,n),0,None)
wind_speed = np.clip(np.random.normal(15,6,n),0,None)
rain_mm = np.clip(np.random.exponential(2,n)-0.5,0,None)
pressure = np.random.normal(1015,7,n)

pulpo_prob = ((temp_water>16)&(temp_water<20)&(wave_height<1.2)).astype(int)
lubina_prob = ((temp_water>14)&(wave_height>1)&(wave_height<2.5)).astype(int)
percebe_prob = ((wave_height>2)).astype(int)

pulpo_catch = np.where(pulpo_prob,np.random.gamma(2,8,n),np.random.gamma(0.5,2,n))
lubina_catch = np.where(lubina_prob,np.random.gamma(2,5,n),np.random.gamma(0.5,1.5,n))
percebe_catch = np.where(percebe_prob,np.random.gamma(2,4,n),np.random.gamma(0.5,1,n))

df = pd.DataFrame({
"date":dates,
"water_temp":temp_water,
"wave_height":wave_height,
"wind_speed":wind_speed,
"rain":rain_mm,
"pressure":pressure,
"pulpo":pulpo_catch,
"lubina":lubina_catch,
"percebe":percebe_catch
})

df["species"] = df[["pulpo","lubina","percebe"]].idxmax(axis=1)

# -----------------------------
# ENTRENAR IA
# -----------------------------

features = ["water_temp","wave_height","wind_speed","rain","pressure"]

X = df[features]
y = df["species"]

model = RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(X,y)

# -----------------------------
# APP STREAMLIT
# -----------------------------

st.title("🌊 IA Pesquera del Cantábrico")

st.write("Predicción de pesca basada en condiciones del mar")

st.subheader("Condiciones del día")

temp = st.slider("Temperatura del agua (°C)",10.0,24.0,18.0)

wave = st.slider("Altura de ola (m)",0.0,4.0,1.0)

wind = st.slider("Viento (km/h)",0.0,60.0,15.0)

rain = st.slider("Lluvia (mm)",0.0,20.0,0.0)

pressure = st.slider("Presión atmosférica",990.0,1040.0,1015.0)

if st.button("Predecir pesca favorable"):

    pred = model.predict([[temp,wave,wind,rain,pressure]])[0]

    st.success(f"🎯 Especie más favorable hoy: {pred.upper()}")

# -----------------------------
# GRÁFICOS HISTÓRICOS
# -----------------------------

st.subheader("Capturas históricas")

species = st.selectbox("Selecciona especie",["pulpo","lubina","percebe"])

fig,ax = plt.subplots()

ax.plot(df["date"],df[species])

ax.set_title(f"Histórico de capturas: {species}")

ax.set_xlabel("Fecha")

ax.set_ylabel("Captura simulada")

st.pyplot(fig)
