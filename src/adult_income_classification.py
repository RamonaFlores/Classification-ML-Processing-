"""
DataPros - Adult Income Classification Model
============================================

Este script implementa un modelo de clasificación binaria usando Spark ML
para predecir si una persona gana más de 50K al año basándose en características
demográficas y laborales.

Autor: DataPros Team
Fecha: 2024
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count, isnull
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os

class AdultIncomeClassifier:
    """
    Clase principal para el modelo de clasificación de ingresos adultos
    """
    
    def __init__(self, app_name="AdultIncomeClassification"):
        """
        Inicializa la sesión de Spark y configura el clasificador
        
        Args:
            app_name (str): Nombre de la aplicación Spark
        """
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        self.model = None
        self.pipeline = None
        
    def load_data(self, file_path):
        """
        Carga los datos desde el archivo CSV
        
        Args:
            file_path (str): Ruta al archivo CSV
            
        Returns:
            DataFrame: DataFrame de Spark con los datos cargados
        """
        print("=" * 60)
        print("CARGANDO DATOS")
        print("=" * 60)
        
        # Definir el esquema para optimizar la carga
        schema = StructType([
            StructField("age", IntegerType(), True),
            StructField("sex", StringType(), True),
            StructField("workclass", StringType(), True),
            StructField("fnlwgt", IntegerType(), True),
            StructField("education", StringType(), True),
            StructField("hours_per_week", IntegerType(), True),
            StructField("label", StringType(), True)
        ])
        
        # Cargar datos con esquema definido
        df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "false") \
            .schema(schema) \
            .csv(file_path)
        
        print(f"✓ Datos cargados exitosamente desde: {file_path}")
        print(f"✓ Número total de registros: {df.count()}")
        print(f"✓ Número de columnas: {len(df.columns)}")
        
        return df
    
    def explore_data(self, df):
        """
        Explora y analiza los datos cargados
        
        Args:
            df (DataFrame): DataFrame de Spark con los datos
        """
        print("\n" + "=" * 60)
        print("EXPLORACIÓN DE DATOS")
        print("=" * 60)
        
        # Mostrar esquema
        print("\n📋 ESQUEMA DE DATOS:")
        df.printSchema()
        
        # Mostrar primeras filas
        print("\n📊 PRIMERAS 10 FILAS:")
        df.show(10, truncate=False)
        
        # Estadísticas descriptivas
        print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
        df.describe().show()
        
        # Verificar valores nulos
        print("\n🔍 ANÁLISIS DE VALORES NULOS:")
        null_counts = df.select([count(when(isnan(c) | isnull(c), c)).alias(c) for c in df.columns])
        null_counts.show()
        
        # Distribución de la variable objetivo
        print("\n🎯 DISTRIBUCIÓN DE LA VARIABLE OBJETIVO:")
        df.groupBy("label").count().orderBy("count", ascending=False).show()
        
        # Distribución por sexo
        print("\n👥 DISTRIBUCIÓN POR SEXO:")
        df.groupBy("sex", "label").count().orderBy("sex", "label").show()
        
        # Distribución por educación
        print("\n🎓 DISTRIBUCIÓN POR EDUCACIÓN:")
        df.groupBy("education", "label").count().orderBy("education", "label").show()
        
        # Distribución por clase de trabajo
        print("\n💼 DISTRIBUCIÓN POR CLASE DE TRABAJO:")
        df.groupBy("workclass", "label").count().orderBy("workclass", "label").show()
    
    def preprocess_data(self, df):
        """
        Preprocesa los datos para el entrenamiento del modelo
        
        Args:
            df (DataFrame): DataFrame original
            
        Returns:
            DataFrame: DataFrame preprocesado
        """
        print("\n" + "=" * 60)
        print("PREPROCESAMIENTO DE DATOS")
        print("=" * 60)
        
        # Crear una copia del DataFrame
        df_processed = df
        
        # Convertir la variable objetivo a binaria (0 para <=50K, 1 para >50K)
        df_processed = df_processed.withColumn(
            "income_binary", 
            when(col("label") == ">50K", 1).otherwise(0)
        )
        
        print("✓ Variable objetivo convertida a binaria")
        print("✓ 0 = <=50K, 1 = >50K")
        
        # Mostrar distribución de la nueva variable
        print("\n📊 DISTRIBUCIÓN DE LA VARIABLE OBJETIVO BINARIA:")
        df_processed.groupBy("income_binary").count().show()
        
        return df_processed
    
    def create_feature_pipeline(self):
        """
        Crea el pipeline de transformación de características
        
        Returns:
            Pipeline: Pipeline de Spark ML
        """
        print("\n" + "=" * 60)
        print("CONSTRUCCIÓN DEL PIPELINE DE CARACTERÍSTICAS")
        print("=" * 60)
        
        # Indexadores para variables categóricas
        sex_indexer = StringIndexer(inputCol="sex", outputCol="sex_indexed")
        workclass_indexer = StringIndexer(inputCol="workclass", outputCol="workclass_indexed")
        education_indexer = StringIndexer(inputCol="education", outputCol="education_indexed")
        
        # Codificador one-hot para variables categóricas
        sex_encoder = OneHotEncoder(inputCol="sex_indexed", outputCol="sex_encoded")
        workclass_encoder = OneHotEncoder(inputCol="workclass_indexed", outputCol="workclass_encoded")
        education_encoder = OneHotEncoder(inputCol="education_indexed", outputCol="education_encoded")
        
        # Ensamblador de características
        feature_columns = [
            "age", "fnlwgt", "hours_per_week",
            "sex_encoded", "workclass_encoded", "education_encoded"
        ]
        
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features"
        )
        
        # Crear pipeline
        pipeline = Pipeline(stages=[
            sex_indexer,
            workclass_indexer,
            education_indexer,
            sex_encoder,
            workclass_encoder,
            education_encoder,
            assembler
        ])
        
        print("✓ Pipeline de características creado")
        print("✓ Variables numéricas: age, fnlwgt, hours_per_week")
        print("✓ Variables categóricas codificadas: sex, workclass, education")
        
        return pipeline
    
    def train_model(self, df, pipeline):
        """
        Entrena el modelo de regresión logística
        
        Args:
            df (DataFrame): DataFrame preprocesado
            pipeline (Pipeline): Pipeline de transformación de características
            
        Returns:
            PipelineModel: Modelo entrenado
        """
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO DEL MODELO")
        print("=" * 60)
        
        # Aplicar transformaciones de características
        df_transformed = pipeline.fit(df).transform(df)
        
        # Seleccionar solo las columnas necesarias
        df_final = df_transformed.select("features", "income_binary")
        
        # Dividir datos en entrenamiento y prueba
        train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)
        
        print(f"✓ Datos de entrenamiento: {train_data.count()} registros")
        print(f"✓ Datos de prueba: {test_data.count()} registros")
        
        # Crear modelo de regresión logística
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="income_binary",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.8
        )
        
        # Crear pipeline completo con el modelo
        full_pipeline = Pipeline(stages=[pipeline, lr])
        
        # Entrenar el modelo
        print("\n🔄 Entrenando modelo de Regresión Logística...")
        self.model = full_pipeline.fit(df)
        
        print("✓ Modelo entrenado exitosamente")
        
        return self.model, test_data
    
    def evaluate_model(self, model, test_data):
        """
        Evalúa el rendimiento del modelo
        
        Args:
            model (PipelineModel): Modelo entrenado
            test_data (DataFrame): Datos de prueba
            
        Returns:
            dict: Métricas de evaluación
        """
        print("\n" + "=" * 60)
        print("EVALUACIÓN DEL MODELO")
        print("=" * 60)
        
        # Hacer predicciones
        predictions = model.transform(test_data)
        
        # Evaluador para clasificación binaria
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="income_binary",
            rawPredictionCol="rawPrediction"
        )
        
        # Evaluador para métricas multiclase
        multi_evaluator = MulticlassClassificationEvaluator(
            labelCol="income_binary",
            predictionCol="prediction"
        )
        
        # Calcular métricas
        auc = binary_evaluator.evaluate(predictions)
        accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
        precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
        recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
        f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
        
        metrics = {
            "AUC": auc,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
        
        # Mostrar métricas
        print("\n📊 MÉTRICAS DE RENDIMIENTO:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric:12}: {value:.4f}")
        
        # Mostrar matriz de confusión
        print("\n🔢 MATRIZ DE CONFUSIÓN:")
        predictions.groupBy("income_binary", "prediction").count().orderBy("income_binary", "prediction").show()
        
        # Mostrar algunas predicciones de ejemplo
        print("\n🔍 EJEMPLOS DE PREDICCIONES:")
        predictions.select("income_binary", "prediction", "probability").show(10, truncate=False)
        
        return metrics, predictions
    
    def save_model(self, model, output_path):
        """
        Guarda el modelo entrenado
        
        Args:
            model (PipelineModel): Modelo entrenado
            output_path (str): Ruta donde guardar el modelo
        """
        print(f"\n💾 Guardando modelo en: {output_path}")
        model.write().overwrite().save(output_path)
        print("✓ Modelo guardado exitosamente")
    
    def run_complete_pipeline(self, data_path, model_output_path="models/adult_income_model"):
        """
        Ejecuta el pipeline completo de clasificación
        
        Args:
            data_path (str): Ruta al archivo de datos
            model_output_path (str): Ruta donde guardar el modelo
        """
        try:
            # 1. Cargar datos
            df = self.load_data(data_path)
            
            # 2. Explorar datos
            self.explore_data(df)
            
            # 3. Preprocesar datos
            df_processed = self.preprocess_data(df)
            
            # 4. Crear pipeline de características
            feature_pipeline = self.create_feature_pipeline()
            
            # 5. Entrenar modelo
            model, test_data = self.train_model(df_processed, feature_pipeline)
            
            # 6. Evaluar modelo
            metrics, predictions = self.evaluate_model(model, test_data)
            
            # 7. Guardar modelo
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
            self.save_model(model, model_output_path)
            
            print("\n" + "=" * 60)
            print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            print("=" * 60)
            
            return model, metrics
            
        except Exception as e:
            print(f"\n❌ Error durante la ejecución: {str(e)}")
            raise
    
    def stop_spark(self):
        """
        Detiene la sesión de Spark
        """
        if self.spark:
            self.spark.stop()
            print("✓ Sesión de Spark detenida")


def main():
    """
    Función principal para ejecutar el clasificador
    """
    # Configurar rutas
    data_path = "data/adult_income_sample.csv"
    model_path = "models/adult_income_model"
    
    # Crear instancia del clasificador
    classifier = AdultIncomeClassifier()
    
    try:
        # Ejecutar pipeline completo
        model, metrics = classifier.run_complete_pipeline(data_path, model_path)
        
        print("\n🎉 ¡Modelo de clasificación completado exitosamente!")
        print(f"📈 AUC obtenido: {metrics['AUC']:.4f}")
        print(f"🎯 Precisión obtenida: {metrics['Accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Error en la ejecución: {str(e)}")
    finally:
        # Detener Spark
        classifier.stop_spark()


if __name__ == "__main__":
    main()
