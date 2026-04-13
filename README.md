# Data_Science_ML_Project

# 🏥 Proyecto de Machine Learning: Predicción de Resultados Médicos

## 📖 Descripción del Proyecto
El presente proyecto analiza un conjunto de datos de pacientes hospitalizados con diversas condiciones médicas (Cáncer, Diabetes, Obesidad, entre otras). El objetivo principal de esta primera etapa es realizar una limpieza exhaustiva (Data Wrangling) y un Análisis Exploratorio de Datos (EDA) para identificar patrones que influyan en los resultados de los exámenes médicos y los costos de facturación.

**Motivación:** Optimizar la gestión hospitalaria y entender qué factores demográficos o de diagnóstico correlacionan con resultados médicos "Anormales".
**Audiencia:** Directores de centros de salud, analistas de seguros médicos y profesionales de ciencia de datos.

## 🎯 Objetivos de la Primera Entrega
- [x] Importación de datos desde repositorio remoto.
- [x] Limpieza de datos (Data Wrangling): Eliminación de duplicados y transformación de tipos de datos (Fechas).
- [x] Verificación de Valores Nulos y Análisis de Outliers (IQR y Z-Score).
- [x] Análisis Exploratorio de Datos (EDA): Univariado, Bivariado y Multivariado.

## 📊 Sobre el Dataset
- **Fuente original:** Archivo CSV de atención médica.
- **Tamaño final:** 54,966 registros únicos y 15 columnas (tras eliminar 534 registros duplicados).
- **Variable Objetivo (Target):** `test_results` (Clasificación: Normal, Abnormal, Inconclusive).
- **Calidad de los datos:** 0 valores nulos y 0 valores atípicos (outliers) detectados en el monto de facturación.

## 🛠️ Herramientas y Librerías
- **Lenguaje:** Python 3
- **Entorno:** Google Colab
- **Manipulación de datos:** `pandas`, `numpy`
- **Visualización:** `matplotlib.pyplot`, `seaborn`

## 📌 Conclusiones Principales del EDA
1. **Variable Objetivo Equilibrada:** Las categorías a predecir están distribuidas de forma casi perfecta, lo cual es ideal para evitar sesgos en el futuro modelo predictivo.
2. **Ausencia de correlaciones lineales:** El mapa de calor multivariado demostró que variables como la Edad, el Monto de Facturación y los Días de Estancia no tienen una correlación lineal fuerte entre sí.
3. **Decisión de Modelado:** Debido a la falta de linealidad y la complejidad de las variables categóricas, se plantea como hipótesis para la segunda entrega utilizar algoritmos basados en árboles (ej. Random Forest o Decision Trees) para la tarea de Clasificación.
