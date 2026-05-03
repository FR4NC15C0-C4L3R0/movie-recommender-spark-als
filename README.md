# 🎬 Spark Cinema AI: Motor de Recomendación Big Data

![Status](https://img.shields.io/badge/Status-Desarrollo_Finalizado-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Spark](https://img.shields.io/badge/Apache_Spark-3.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)

**Spark Cinema AI** es una solución integral de ingeniería de datos que combina la potencia de procesamiento distribuido de **Apache Spark** con una interfaz moderna y fluida en **Streamlit**. El sistema analiza más de **25 millones de interacciones** (dataset MovieLens 25M) para predecir preferencias y recomendar contenido en tiempo real utilizando Machine Learning a gran escala.

---

## 🚀 Características Principales

- **Procesamiento de Big Data:** Manejo eficiente de 25 millones de ratings con Apache Spark MLlib.
- **Algoritmo ALS (Alternating Least Squares):** Implementación de filtrado colaborativo para predicciones de alta precisión.
- **Interfaz UI "Netflix-Style":** Diseñada en Streamlit con integración dinámica de la **API de TMDB** para visualizar pósters, sinopsis y valoraciones.
- **Evaluación del Modelo:** Cálculo del **RMSE** (Root Mean Squared Error) para validar la precisión de las predicciones.
- **Diseño Robusto (UX/UI):** Maquetación CSS personalizada para alineación de cuadrícula y gestión segura de credenciales mediante variables de entorno (`.env`).

---

## 🧠 ¿Cómo funciona? (Capas de Ingeniería)

El sistema se divide en tres niveles de ejecución:

1. **Capa de Inteligencia (Backend):** Spark carga el modelo ALS pre-entrenado. Al seleccionar un `userId`, el modelo genera una predicción de puntuación para miles de películas que el usuario no ha visto aún, seleccionando las *N* mejores recomendaciones.
2. **Capa de Integración (API REST):** La aplicación toma los IDs de Spark y consulta los metadatos visuales en la API de TMDB. Se utiliza un sistema de **placeholders** para garantizar la estabilidad visual si algún dato no está disponible.
3. **Capa de Presentación (Frontend):** Los resultados se renderizan en una cuadrícula de altura fija mediante contenedores HTML/CSS inyectados, asegurando una estética profesional similar a las plataformas comerciales.

---

## 🛠️ Requisitos del Sistema y Stack

### Requisitos de Infraestructura
- **Java Development Kit (JDK) 8+:** Spark se ejecuta en la JVM. Es necesario configurar la variable `JAVA_HOME`.
- **Apache Spark 3.x:** Configurado con la variable de entorno `SPARK_HOME` y añadido al `PATH`.
- **Python 3.8+:** Lenguaje base para la lógica de negocio y UI.

### Tecnologías Utilizadas
- **Procesamiento:** [Apache Spark](https://spark.apache.org/) (PySpark)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **API Externa:** [The Movie Database (TMDB)](https://www.themoviedb.org/)
- **Gestión de Entorno:** Virtualenv, Conda y `python-dotenv`.

---

## ⚙️ Instalación y Configuración

Sigue estos pasos para poner en marcha el proyecto:

### 1. Preparar el repositorio y el entorno
```bash
# Clonar el repositorio
git clone [https://github.com/tu-usuario/movie_recommender_project.git](https://github.com/tu-usuario/movie_recommender_project.git)
cd movie_recommender_project
```
```bash
# Crear y activar entorno virtual
python -m venv .venv
```

```bash
# En Windows: .venv\Scripts\activate | En Unix: source .venv/bin/activate
```

```bash
# Instalar dependencias
pip install pyspark findspark streamlit python-dotenv
```
### 2. Gestión de Datos y Secretos
Descarga el dataset MovieLens 25M y coloca los archivos ratings.csv y movies.csv en la carpeta ml-25m/.

Crea un archivo .env en la raíz del proyecto y añade tu Token de TMDB:

```bash
TMDB_API_TOKEN=tu_token_aqui_sin_comillas
```

### 3. Ejecución
Para entrenar/probar el motor Spark: 
```bash
python main.py
```

Para lanzar la interfaz de usuario: 
```bash 
streamlit run app.py
```

### 📊 Resultados y Análisis del Modelo:

El modelo ALS entrenado ha alcanzado un RMSE de aproximadamente 0.8066. Esto indica que, en promedio, la diferencia entre las calificaciones predichas y las reales es de solo 0.8 puntos en una escala de 0.5 a 5.0, un resultado robusto para un dataset de esta complejidad.

![IMPORTANTE](https://img.shields.io/badge/!-IMPORTANTE-red)

**Nota sobre rendimiento:** Para la demostración fluida en local, se utiliza un muestreo (sampling) del 5% del dataset. El sistema es totalmente escalable al 100% de los datos (25M de registros) ajustando el parámetro fraction a 1.0 en el código.



## 👤 Autor

**Francisco Calero Sánchez**

**· 🌐 [Mi Portafolio](https://portafolio-franciscocalero.onrender.com/)**

**· 🔗 [LinkedIn](https://www.linkedin.com/in/francisco-calero/)**

**· 📧 [Email](mailto:francisco.calero.sanchez@gmail.com)**

***-Este proyecto fue desarrollado como parte de mi especialización en Big Data en Granada, España.-***

---