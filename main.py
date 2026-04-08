#import findspark
#findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import col, avg, count, lit, explode
import os
import sys # Importar sys para depuración de PATH
#=======================================================================================================================
# --- 1. CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ml-25m")

RATINGS_PATH = os.path.join(DATA_PATH, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_PATH, "movies.csv")

# Definimos la ruta donde se guardará/cargará el modelo ALS
MODEL_SAVE_LOAD_PATH = os.path.join(BASE_DIR, "models", "als_movie_recommender_model")

# --- Debugging: Verificar Variables de Entorno de Hadoop ---
print("\n--- Verificando Variables de Entorno (en Docker) ---")
# HADOOP_HOME no es relevante en esta configuración de Docker
hadoop_home_env = os.environ.get('HADOOP_HOME')
if hadoop_home_env:
    print(f"HADOOP_HOME: {hadoop_home_env}")
else:
    print("HADOOP_HOME: No definida (esperado en este setup Docker).")

# Debemos manejar el caso en que la variable no exista (devuelve None)
python_path_env = os.environ.get('PATH')
if python_path_env:
    print(f"PATH (primeros 200 chars): {python_path_env[:200]}...")
else:
    print("PATH: No definida en este entorno.")

java_home_env = os.environ.get('JAVA_HOME')
if java_home_env:
    print(f"JAVA_HOME: {java_home_env}")
else:
    print("JAVA_HOME: No definida (esperado que Spark la maneje internamente).")
print("--- Fin Verificación Variables de Entorno ---\n")

model_base_dir = os.path.dirname(MODEL_SAVE_LOAD_PATH)
if not os.path.exists(model_base_dir):
    print(f"Creando directorio para modelos: {model_base_dir}")
    os.makedirs(model_base_dir, exist_ok=True)
else:
    print(f"Directorio de modelos ya existe: {model_base_dir}")


# --- 2. INICIALIZAR SPARK SESSION ---
print("Inicializando Spark Session...")
spark = SparkSession.builder \
    .appName("MovieRecommendationSystem") \
    .master("local[*]") \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:log4j.properties") \
    .config("spark.driver.memory", "24g") \
    .config("spark.executor.memory", "24g") \
    .getOrCreate()
print("Spark Session inicializada.")

# --- 3. CONFIGURAR NIVEL DE LOGGING DE SPARK ---
spark.sparkContext.setLogLevel("WARN")
#spark.sparkContext.setLogLevel("INFO")

# --- 4. Cargar datos de películas (necesario para recomendaciones) ---
print(f"Cargando movies.csv desde: {MOVIES_PATH}")
movies_df = spark.read.csv(MOVIES_PATH, header=True, inferSchema=True)
movies_selected_df = movies_df.select(col("movieId"), col("title"), col("genres"))
print("movies.csv cargado exitosamente.")

# --- 5. COMPROBAR Y CARGAR/ENTRENAR EL MODELO ---
model = None # Inicializamos model a None
model_trained_now = False # Indicador para saber si entrenamos en esta ejecución

print("\nVerificando si el modelo ALS ya está entrenado...")
try:
    # Intenta cargar el modelo ALS.
    model = ALSModel.load(MODEL_SAVE_LOAD_PATH)
    print(f"Modelo ALS cargado exitosamente desde: {MODEL_SAVE_LOAD_PATH}")
    model_trained_now = False
except Exception as e:
    # Capturamos cualquier error, incluyendo UnsatisfiedLinkError o InvalidInputException
    print(f"No se encontró un modelo entrenado o hubo un error al cargar: {e}")
    print("Iniciando el proceso de entrenamiento del modelo ALS...")
    model_trained_now = True

    # --- Cargar y preprocesar datos de ratings solo si vamos a entrenar ---
    print(f"Cargando ratings.csv desde: {RATINGS_PATH}")
    ratings_df = spark.read.csv(RATINGS_PATH, header=True, inferSchema=True)
    print("ratings.csv cargado exitosamente.")
    
    #Muestreo reducido para ejecución en entornos locales; para producción se utiliza el 100% del dataset de 25M
    sampled_ratings_df = ratings_df.sample(withReplacement=False, fraction=0.05, seed=42)
    print(f"Dataset de ratings muestreado. Nuevo tamaño: {sampled_ratings_df.count()} registros.")

    
    ratings_cleaned_df = sampled_ratings_df.select( 
        col("userId"),
        col("movieId"),
        col("rating").alias("user_rating"),
        col("timestamp")
    )
    print("Ratings preprocesados.")

    # --- Preparación para ALS ---
    als_data = ratings_cleaned_df.select(
        col("userId"),
        col("movieId"),
        col("user_rating").alias("rating")
    )

    # Dividir los datos en conjuntos de entrenamiento y prueba.
    (training, test) = als_data.randomSplit([0.8, 0.2], seed=42)

    print(f"Tamaño del conjunto de entrenamiento: {training.count()}")
    print(f"Tamaño del conjunto de prueba: {test.count()}")

    # Construcción y entrenamiento del modelo ALS
    als = ALS(
        rank=10,
        maxIter=10,
        regParam=0.01,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        seed=42
    )

    model = als.fit(training)
    print("Modelo ALS entrenado exitosamente.")

    # --- Guardar el modelo entrenado ---
    print(f"Guardando el modelo ALS en: {MODEL_SAVE_LOAD_PATH}")
    try:
        # Usa .overwrite() para reemplazar si ya existe (si se entrena de nuevo)
        model.write().overwrite().save(MODEL_SAVE_LOAD_PATH)
        print("Modelo guardado exitosamente.")
    except Exception as e_save:
        print(f"ERROR: Fallo al guardar el modelo. Causa: {e_save}")
        print("Esto podría indicar un problema persistente con winutils o permisos de escritura.")


# --- 6. EVALUAR EL MODELO (SOLO SI SE ENTRENÓ AHORA) ---
# Si el modelo se entrenó en esta ejecución, evaluamos su rendimiento.
if model_trained_now:
    print("\nEvaluando el modelo ALS (recientemente entrenado)...")
    # Para la evaluación, regeneramos el test set para consistencia
    # (Si no se entrenó ahora, asumimos que ya se evaluó en su momento)
    full_ratings_df = spark.read.csv(RATINGS_PATH, header=True, inferSchema=True)
    full_ratings_cleaned_df = full_ratings_df.select(
        col("userId"),
        col("movieId"),
        col("rating").alias("user_rating")
    )
    full_als_data = full_ratings_cleaned_df.select(
        col("userId"),
        col("movieId"),
        col("user_rating").alias("rating")
    )
    _, test = full_als_data.randomSplit([0.8, 0.2], seed=42) # Regenera el test set

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Raíz del error cuadrático medio (RMSE) del modelo ALS: {rmse}")
else:
    print("\nEl modelo fue cargado, no re-evaluado en esta ejecución.")


# --- 7. GENERAR RECOMENDACIONES ---
print("\nGenerando recomendaciones para usuarios específicos...")

user_id_to_recommend = 1

# Aseguramos que ratings_df esté disponible para filtrar películas vistas
# Si el modelo fue cargado (no entrenado ahora), ratings_df no se cargó en el bloque de entrenamiento.
# Lo cargamos aquí si es necesario para evitar NameError.
try:
    _ = ratings_df # Intenta acceder a ratings_df
except NameError:
    print("Cargando ratings.csv para determinar películas vistas (necesario para filtrar recomendaciones)...")
    ratings_df = spark.read.csv(RATINGS_PATH, header=True, inferSchema=True)


movies_seen_by_user = ratings_df.filter(col("userId") == user_id_to_recommend) \
                                 .select("movieId") \
                                 .distinct()

# 2. Encontrar todas las películas que el usuario *no* ha visto.
all_movie_ids = movies_selected_df.select("movieId").distinct()
movies_not_seen_by_user = all_movie_ids.join(
    movies_seen_by_user,
    on="movieId",
    how="left_anti"
)

# 3. Crear un DataFrame de prueba para el usuario con las películas no vistas.
user_unrated_movies = movies_not_seen_by_user.withColumn("userId", lit(user_id_to_recommend))

# 4. Generar predicciones para el usuario en las películas no vistas.
user_predictions = model.transform(user_unrated_movies)

# 5. Unir las predicciones con los títulos de las películas y seleccionar las 10 mejores.
top_10_recs_user = user_predictions.join(
    movies_selected_df,
    on="movieId",
    how="inner"
).orderBy(
    col("prediction").desc()
).limit(10)

print(f"\nTop 10 recomendaciones para el usuario {user_id_to_recommend}:")
top_10_recs_user.show(truncate=False)

# --- 8. DETENER SPARK SESSION ---
print("\nDeteniendo Spark Session...")
spark.stop()
print("Spark Session detenida. Script finalizado.")