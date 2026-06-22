import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de la página (Debe ser la primera instrucción)
st.set_page_config(page_title="Predicción Médica", page_icon="🏥", layout="wide")

# 2. Cargar los modelos y herramientas
@st.cache_resource
def load_models():
    modelo = joblib.load('modelo_salud.pkl')
    scaler = joblib.load('scaler_salud.pkl')
    columnas = joblib.load('columnas_salud.pkl')
    return modelo, scaler, columnas

modelo, scaler, columnas = load_models()

# 3. Diseño de la Presentación Ejecutiva
st.title("🏥 Dashboard Ejecutivo: Predicción de Resultados Médicos")
st.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning (Random Forest)** optimizado para predecir 
si el resultado de las pruebas de un paciente será Normal, Anormal o Inconcluso, basándose en sus características clínicas y demográficas.
""")

st.divider()

# 4. Crear columnas para el formulario (Diseño limpio)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Datos Demográficos")
    age = st.slider("Edad del Paciente", 0, 100, 45)
    gender = st.selectbox("Género", ["Male", "Female"])
    blood_type = st.selectbox("Tipo de Sangre", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

with col2:
    st.subheader("🩺 Datos Clínicos")
    medical_condition = st.selectbox("Condición Médica", ["Cancer", "Diabetes", "Obesity", "Asthma", "Hypertension", "Arthritis"])
    admission_type = st.selectbox("Tipo de Ingreso", ["Emergency", "Urgent", "Elective"])

with col3:
    st.subheader("💳 Estancia y Facturación")
    days_stayed = st.number_input("Días de Internación", min_value=1, max_value=60, value=5)
    billing_amount = st.number_input("Monto Facturado ($)", min_value=0.0, max_value=100000.0, value=15000.0)

# Botón de predicción
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔍 Realizar Predicción", use_container_width=True, type="primary"):
    
    # 5. Preparar los datos tal como los espera el modelo
    # Creamos un diccionario con los datos ingresados
    input_dict = {
        'age': [age],
        'gender': [gender],
        'blood_type': [blood_type],
        'medical_condition': [medical_condition],
        'admission_type': [admission_type],
        'billing_amount': [billing_amount],
        'days_stayed': [days_stayed]
        # Las variables omitidas se rellenarán automáticamente con 0 en el reindex
    }
    
    # Convertir a DataFrame
    input_df = pd.DataFrame(input_dict)
    
    # Aplicar One-Hot Encoding
    input_dummies = pd.get_dummies(input_df)
    
    # Alinear las columnas con las que el modelo aprendió (Rellena con 0 lo que falte)
    input_dummies = input_dummies.reindex(columns=columnas, fill_value=0)
    
    # Escalar las numéricas
    cols_numericas = ['age', 'billing_amount', 'days_stayed']
    input_dummies[cols_numericas] = scaler.transform(input_dummies[cols_numericas])
    
    # 6. Hacer la predicción
    prediccion_num = modelo.predict(input_dummies)[0]
    
    # Mapear el número de vuelta a la palabra
    diccionario_resultados = {0: "Normal", 1: "Abnormal", 2: "Inconclusive"}
    resultado_final = diccionario_resultados.get(prediccion_num, "Desconocido")
    
    # 7. Mostrar el resultado con alertas visuales
    st.divider()
    st.subheader("📊 Resultado del Modelo Predictivo")
    
    if resultado_final == "Normal":
        st.success(f"**Pronóstico:** {resultado_final} ✅ - El paciente tiene alta probabilidad de resultados favorables.")
    elif resultado_final == "Abnormal":
        st.error(f"**Pronóstico:** {resultado_final} 🚨 - Se recomienda revisión exhaustiva y atención prioritaria.")
    else:
        st.warning(f"**Pronóstico:** {resultado_final} ⚠️ - Los resultados podrían no ser concluyentes. Se sugieren más pruebas.")
