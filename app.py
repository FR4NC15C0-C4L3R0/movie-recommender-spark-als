import os
import sys
from dotenv import load_dotenv
import platform

# --- 0. CONFIGURACIÓN CRÍTICA (DEBE IR ANTES QUE NADA) ---
load_dotenv()
TMDB_API_TOKEN = os.getenv("TMDB_API_TOKEN")

# Definimos las rutas exactas
#os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk1.8.0_251"

# Solo define JAVA_HOME manualmente si está en Windows
if platform.system() == "Windows":
    os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk1.8.0_251"
# En la nube (Linux), Streamlit encontrará Java automáticamente gracias al packages.txt

os.environ["HADOOP_HOME"] = r"C:\hadoop"
# Forzamos a Spark a usar el Python de tu entorno virtual
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# Añadimos al PATH
os.environ["PATH"] = os.path.join(os.environ["JAVA_HOME"], "bin") + os.pathsep + \
                     os.path.join(os.environ["HADOOP_HOME"], "bin") + os.pathsep + \
                     os.environ.get("PATH", "")

import streamlit as st
import requests
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode

st.set_page_config(
    page_title="Spark Cinema AI",
    page_icon="logo_favicon.png", # Asegúrate de que el nombre coincida con tu archivo exportado
    layout="wide"
)

# Carga las variables desde el archivo .env


# Ahora recuperamos el token de forma segura
TMDB_API_TOKEN = os.getenv("TMDB_API_TOKEN")


# --- 2. FUNCIONES DE APOYO ---
def get_movie_details(movie_title):
    """Busca poster, nota y sinopsis en la API de TMDB"""
    search_title = movie_title.split(' (')[0] if " (" in movie_title else movie_title
    url = f"https://api.themoviedb.org/3/search/movie?query={search_title}&language=es-ES"
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_TOKEN}"}

    detalles = {
        "poster": "https://via.placeholder.com/500x750?text=Sin+Imagen",
        "nota": "N/A",
        "resumen": "Sin descripción disponible."
    }

    try:
        response = requests.get(url, headers=headers).json()
        if response.get('results'):
            peli = response['results'][0]
            if peli.get('poster_path'):
                detalles["poster"] = f"https://image.tmdb.org/t/p/w500{peli['poster_path']}"
            detalles["nota"] = f"⭐ {peli.get('vote_average', 'N/A')}"
            # Cortamos la sinopsis si es muy larga para que no rompa el diseño
            resumen = peli.get('overview', 'Sin descripción.')
            detalles["resumen"] = (resumen[:120] + '...') if len(resumen) > 120 else resumen
    except Exception:
        pass
    return detalles
    # ... (tu código anterior de búsqueda) ...

    # Esta es la clave: una imagen elegante de "No disponible"
    # Puedes usar una URL de un diseño tuyo o este placeholder gris profesional
    imagen_error = "https://via.placeholder.com/500x750.png?text=Imagen+No+Disponible"

    detalles = {
        "poster": imagen_error,
        "nota": "N/A",
        "resumen": "Sin descripción disponible."
    }

    try:
        response = requests.get(url, headers=headers).json()
        if response.get('results'):
            peli = response['results'][0]
            # Solo si existe el poster_path, sobreescribimos el placeholder
            if peli.get('poster_path'):
                detalles["poster"] = f"https://image.tmdb.org/t/p/w500{peli['poster_path']}"

            # Lo mismo para la nota y el resumen...
            detalles["nota"] = f"⭐ {peli.get('vote_average', 'N/A')}"
            resumen = peli.get('overview', 'Sin descripción.')
            detalles["resumen"] = (resumen[:120] + '...') if len(resumen) > 120 else resumen
    except Exception:
        pass  # Si falla la API, el diccionario ya tiene los valores por defecto
    return detalles



@st.cache_resource
def iniciar_recursos():
    # Iniciamos Spark con configuraciones de estabilidad para Windows
    spark = SparkSession.builder \
        .appName("MovieRecommender") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.sql.shuffle.partitions", "5") \
        .master("local[*]") \
        .getOrCreate()

    modelo = ALSModel.load("models/als_movie_recommender_model")
    peliculas_df = spark.read.csv("ml-25m/movies.csv", header=True, inferSchema=True)
    return spark, modelo, peliculas_df


# --- 3. INTERFAZ DE USUARIO ---
st.set_page_config(page_title="MARS 📽️‍💻", layout="wide")
st.title("MARS 📽️💻")
st.subheader("📊‍Movie Analytics & Recommender With Spark⚙️")

st.sidebar.header("️🧭️Panel de Control️️")

# 1. Recogemos el ID como texto
user_input = st.sidebar.text_input("⌨️ ID de Usuario (1 - 162541):", value="1")

# 2. Intentamos convertirlo a número
try:
    user_id = int(user_input)
except ValueError:
    st.sidebar.error("⚠️ Introduce un número válido ⚠️")
    st.stop() # Esto detiene la ejecución si no es un número, ¡muy profesional!

# 3. Slider para cantidad
num_recs = st.sidebar.slider("🖱️¿Cuántas películas?", 4, 12, 6)

# Explicación técnica
# --- Explicación Detallada ---
with st.expander("ℹ️ ¿Cómo funciona este recomendador? (Explicación técnica)"):
    st.write(f"""
    Este sistema utiliza un algoritmo de **Mínimos Cuadrados Alternativos (ALS).**

    Utiliza **Apache Spark** para analizar 25 millones de votos. El modelo encuentra patrones entre usuarios similares para predecir qué le gustará. 

    1. **Entrenamiento:** El modelo ha analizado 25 millones de interacciones reales. Ha "aprendido" los gustos del **Usuario {user_id}** observando qué películas puntuó anteriormente.
    2. **Similitud:** El sistema busca otros miles de usuarios con patrones de voto similares a los del **Usuario {user_id}**.
    3. **Predicción:** Si a usuarios parecidos al **Usuario {user_id}** les gustó una película que aún no ha visto, el motor calcula una puntuación estimada para el **Usuario {user_id}**...
    4. **Resultado:** Las películas que muestra son las que tienen la mayor probabilidad de que gusten al **Usuario {user_id}**, calculadas en tiempo real mediante procesamiento distribuido.

    Para que la experiencia sea visual, se ha integrado una conexión con la API de **The Movie Database (TMDB)**:

    1. **Sincronización:** Una vez que Spark identifica las IDs de las películas recomendadas, extrae sus títulos originales.
    2. **Consulta en tiempo real:** Por cada recomendación, la aplicación lanza una petición HTTP a los servidores de TMDB usando el protocolo **REST API**.
    3. **Procesamiento de imagen:** Recibe una respuesta en formato JSON, extrae la ruta de la carátula y la renderiza dinámicamente en la interfaz, aplicando estilos CSS para asegurar que todos los pósters mantengan una estética uniforme y profesional.

    """)

# --- 5. ACCIÓN Y RESULTADOS (Solo ocurren al pulsar el botón) ---
if st.sidebar.button("🚀 Generar Recomendaciones"):
    with st.spinner('Spark está calculando tus gustos...'):
        try:
            # Iniciamos recursos
            spark, model, movies_df = iniciar_recursos()

            # Creamos el dataframe del usuario
            user_df = spark.createDataFrame([(user_id,)], ["userId"])

            # Generamos recomendaciones
            recommendations = model.recommendForUserSubset(user_df, num_recs)

            if recommendations.count() > 0:
                # Procesamos resultados
                recs_exploded = recommendations.select(explode("recommendations").alias("rec"))
                recs_final = recs_exploded.select("rec.movieId")

                # Unimos con los nombres de las pelis y pasamos a Pandas
                resultado = recs_final.join(movies_df, "movieId").select("title", "genres").toPandas()

                st.success(f"Sugerencias para el Usuario {user_id}:")
                st.divider()

                # --- MOSTRAR EN CUADRÍCULA BLINDADA ---
                columnas = st.columns(3)
                for i, row in resultado.iterrows():
                    with columnas[i % 3]:
                        info = get_movie_details(row['title'])

                        # 1. CONTENEDOR DE IMAGEN CON ALTURA FIJA
                        # Esto garantiza que el hueco del póster siempre mida 300px de alto
                        st.markdown(f"""
                                            <div style="height: 300px; width: 200px; background-color: #f0f0f0; 
                                                        border-radius: 10px; overflow: hidden; display: flex; 
                                                        align-items: center; justify-content: center; margin-bottom: 10px;">
                                                <img src="{info['poster']}" style="width: 100%; height: 100%; object-fit: cover;">
                                            </div>
                                        """, unsafe_allow_html=True)

                        # 2. TÍTULO CON ALTURA FIJA (45px)
                        st.markdown(f"""
                                            <div style='height: 45px; overflow: hidden; margin-bottom: 5px;'>
                                                <strong style='font-size: 1.1rem;'>{row['title']}</strong>
                                            </div>
                                        """, unsafe_allow_html=True)

                        # 3. NOTA
                        st.write(f"{info['nota']}")

                        # 4. SINOPSIS CON ALTURA FIJA (100px)
                        st.markdown(f"""
                                            <div style="height: 100px; overflow: hidden; font-size: 0.85rem; color: #555; line-height: 1.4;">
                                                {info['resumen']}
                                            </div>
                                        """, unsafe_allow_html=True)

                        st.divider()
            else:
                st.error("No hay datos para este usuario en el modelo.")
        except Exception as e:
            st.error(f"Error de Spark: {e}")