import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Cargar el modelo y el escalador
model = load_model('residual_model.h5')
scaler = joblib.load('scaler.pkl')

# Crear un diccionario para las tipologías de la máquina
tipologia_dict = {1: "Elevación Diesel", 2: "Elevación Eléctrica", 3: "Transpaleta", 4: "Apilador", 5: "Preparapedidos", 6: "Preparapedidos altura", 7: "Retráctil", 8: "Carret. Diesel", 9: "Carret Elect Plomo", 10: "Carret. Elect Litio", 11: "Bigtruck", 12: "Manipulación y todoterreno"}

# Crear un diccionario para los estados de las máquinas
estado_dict = {1: "Bueno", 2: "Normal", 3: "Malo"}

st.title('Predicción del valor residual de una máquina')

# Solicitar datos al usuario
tipologia = st.selectbox("Seleccione la tipología de la máquina:", list(tipologia_dict.keys()), format_func=lambda x: tipologia_dict[x])
anos = st.number_input("Introduzca el número de años de uso:", min_value=1, max_value=30, value=10)
horas = st.number_input("Introduzca las horas de uso:", min_value=0, max_value=35000, value=5000)
estado = st.selectbox("Introduzca el estado de la máquina:", list(estado_dict.keys()), format_func=lambda x: estado_dict[x])
coste = st.number_input("Introduzca el coste de la máquina nueva:", min_value=3000.0, value=20000.0, step=100.0)

if st.button('Calcular valor residual'):
    # Crear un DataFrame con los datos del usuario
    user_data = pd.DataFrame(np.array([[tipologia, anos, horas, estado]]), columns=['Tipologia', 'anos', 'horas', 'estado'])
    
    # Escalar los datos del usuario
    user_data_scaled = scaler.transform(user_data)
    
    # Hacer la predicción
    prediction = model.predict(user_data_scaled)
    
    # Calcular el valor residual de la máquina
    residual_value = coste * prediction[0][0]
    
    st.write(f'El valor residual de la máquina es: {residual_value:.2f}')
