# 🎬 Sistema de Recomendación de Películas con Apache Spark

## 🌟 Visión General del Proyecto

Este proyecto implementa un sistema de recomendación de películas utilizando un enfoque de **Filtrado Colaborativo** basado en el algoritmo **ALS (Alternating Least Squares)**, potenciado por Apache Spark MLlib. El objetivo es predecir las calificaciones que un usuario daría a películas que no ha visto, y así recomendarle nuevos títulos de su interés.

El sistema ha sido desarrollado en Python y utiliza el dataset **MovieLens 25M**, uno de los conjuntos de datos de ratings de películas más grandes y completos disponibles, lo que demuestra la capacidad de Spark para manejar grandes volúmenes de datos de manera eficiente.

## ✨ Características Principales

* **Procesamiento de Big Data:** Maneja un dataset de 25 millones de ratings con Apache Spark.
* **Algoritmo ALS:** Utiliza el potente algoritmo ALS para el filtrado colaborativo.
* **Evaluación del Modelo:** Calcula el RMSE (Root Mean Squared Error) para evaluar la precisión de las predicciones del modelo.
* **Generación de Recomendaciones:** Genera recomendaciones personalizadas para usuarios y demuestra cómo obtener los títulos de las películas recomendadas.
* **Entorno Local:** Configurado para ejecutarse localmente, ideal para demostraciones y desarrollo.

## 🚀 Requisitos del Sistema

Antes de ejecutar el proyecto, asegúrate de tener instalados los siguientes requisitos:

* **Java Development Kit (JDK) 8 o superior:** Spark se ejecuta en la JVM.
    * Puedes descargarlo desde la [página oficial de Oracle JDK](https://www.oracle.com/java/technologies/downloads/) o [OpenJDK](https://openjdk.org/install/).
    * Asegúrate de configurar la variable de entorno `JAVA_HOME` para que apunte a tu instalación de JDK.
* **Python 3.8 o superior:** El script principal está escrito en Python.
    * Puedes descargarlo desde la [página oficial de Python](https://www.python.org/downloads/).
* **Apache Spark (versión 3.x recomendada):** El core del proyecto.
    * Descarga la versión pre-construida con Hadoop (ej., `spark-3.x.x-bin-hadoop3.2.tgz`) desde el [sitio de descargas de Apache Spark](https://spark.apache.org/downloads.html).
    * Descomprime el archivo en una ubicación accesible (ej., `C:\spark` en Windows o `/opt/spark` en Linux/macOS).
    * Configura la variable de entorno `SPARK_HOME` para que apunte a la carpeta raíz de Spark.
    * Añade `%SPARK_HOME%\bin` (Windows) o `$SPARK_HOME/bin` (Linux/macOS) a tu variable de entorno `PATH`.
* **pySpark:** La API de Python para Spark. Se instalará automáticamente con `pip`.

## 📦 Configuración del Entorno

Sigue estos pasos para configurar y ejecutar el proyecto:

1.  **Clonar el Repositorio (o descargar el código):**
    ```bash
    git clone [https://github.com/tu_usuario/movie_recommender_project.git](https://github.com/tu_usuario/movie_recommender_project.git)
    cd movie_recommender_project
    ```
    *(Si aún no lo tienes en GitHub, puedes simplemente crear la carpeta y poner los archivos allí.)*

2.  **Crear un Entorno Virtual (Recomendado):**
    ```bash
    python -m venv .venv
    ```

3.  **Activar el Entorno Virtual:**
    * **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Instalar Dependencias de Python:**
    ```bash
    pip install pyspark findspark
    ```

5.  **Descargar el Dataset MovieLens 25M:**
    * Ve a la página de [descargas de MovieLens](https://grouplens.org/datasets/movielens/).
    * Descarga el archivo `ml-25m.zip`.
    * Descomprime el contenido (`ratings.csv` y `movies.csv`) en una subcarpeta llamada `ml-25m` dentro de la carpeta raíz de tu proyecto.
        Tu estructura debería ser similar a:
        ```
        movie_recommender_project/
        ├── .venv/
        ├── ml-25m/
        │   ├── ratings.csv
        │   └── movies.csv
        ├── main.py
        ├── README.md
        └── requirements.txt  (Puedes crearlo con `pip freeze > requirements.txt` después de instalar pySpark)
        ```

6.  **Configurar `log4j.properties` (Opcional, para reducir el verbosidad de logs):**
    Para evitar el exceso de mensajes WARN/INFO de Spark, crea un archivo llamado `log4j.properties` en la raíz de tu proyecto con el siguiente contenido:
    ```properties
    log4j.rootCategory=WARN, console
    log4j.appender.console=org.apache.log4j.ConsoleAppender
    log4j.appender.console.layout=org.apache.log4j.PatternLayout
    log4j.appender.console.layout.ConversionPattern=%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n
    ```
    Y asegúrate de que tu configuración de SparkSession en `main.py` incluya:
    `.config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:log4j.properties")`

## 🏃‍♀️ Ejecución del Proyecto

Una vez que todo esté configurado:

1.  **Asegúrate de que tu entorno virtual esté activado.**
2.  **Ejecuta el script principal:**
    ```bash
    python main.py
    ```

El script cargará los datos, realizará las transformaciones necesarias, entrenará el modelo ALS, evaluará su rendimiento (RMSE) y finalmente mostrará las recomendaciones para los primeros 5 usuarios y un usuario específico (ID 1).

> [!IMPORTANT]
> **Nota sobre el rendimiento:** Para esta demostración, el script utiliza un **muestreo (sampling) del 5%** del dataset original (`fraction=0.05`). Esto permite una ejecución fluida en entornos locales y pruebas rápidas de la interfaz. El modelo es totalmente compatible con el procesamiento de los **25 millones de registros** ajustando dicho parámetro a `1.0`.

## 🛠️ Estructura del Código (`main.py`)

El archivo `main.py` contiene toda la lógica del sistema de recomendación:

* **Inicialización de SparkSession:** Configura el entorno Spark, asignando recursos de memoria (`spark.driver.memory`, `spark.executor.memory`) para manejar el gran volumen de datos.
* **Carga de Datos:** Lee los datasets `ratings.csv` y `movies.csv` en DataFrames de Spark.
* **Preprocesamiento y Exploración:** Realiza la limpieza de datos, unión de DataFrames y algunas operaciones exploratorias básicas (conteo, esquema, ratings por usuario, etc.).
* **Preparación para ALS:** Transforma los datos al formato requerido por el algoritmo ALS (UserID, MovieID, Rating).
* **División de Datos:** Divide el dataset en conjuntos de entrenamiento y prueba.
* **Entrenamiento del Modelo ALS:** Configura y entrena el modelo de recomendación ALS.
* **Evaluación del Modelo:** Utiliza `RegressionEvaluator` para calcular el RMSE del modelo entrenado.
* **Generación y Procesamiento de Recomendaciones:**
    * Genera recomendaciones crudas para un conjunto de usuarios.
    * Filtra las películas ya vistas por el usuario.
    * Une las recomendaciones con el DataFrame de películas para obtener los títulos y géneros.
    * Muestra las recomendaciones finales.
* **Cierre de SparkSession:** Detiene la sesión de Spark para liberar recursos.

## 📊 Resultados y Análisis del RMSE

El modelo ALS entrenado ha alcanzado un **RMSE de aproximadamente 0.8066**. Este valor indica que, en promedio, la diferencia entre las calificaciones predichas por el modelo y las calificaciones reales es de aproximadamente 0.8 puntos en una escala de 0.5 a 5.0. Para un dataset de esta magnitud y complejidad, este es un resultado inicial bastante robusto, demostrando la capacidad del modelo para predecir las preferencias de los usuarios con una precisión razonable.

## 🤝 Contribuciones

Si tienes sugerencias o mejoras, no dudes en abrir un *issue* o enviar un *pull request*.

## 📄 Licencia

Este proyecto es para fines de demostración y educación.

## 📧 Contacto

[Francisco J. Calero Sánchez] - [franciscocalero.coder@gmail.com] - [www.linkedin.com/in/francisco-calero]