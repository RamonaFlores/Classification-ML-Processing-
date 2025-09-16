#!/usr/bin/env python3
"""
DataPros - Demo Simplificado de Clasificación de Ingresos
=========================================================

Este script ejecuta una versión simplificada del análisis de clasificación
de ingresos adultos con Spark ML.

Autor: DataPros Team
Fecha: 2024
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan, isnull
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import os

def main():
    """
    Función principal para ejecutar el demo simplificado
    """
    print("🚀 DataPros - Demo Simplificado de Clasificación de Ingresos")
    print("=" * 60)
    
    # Inicializar Spark
    spark = SparkSession.builder \
        .appName("AdultIncomeSimpleDemo") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # 1. Cargar datos
        print("\n📊 CARGANDO DATOS...")
        schema = StructType([
            StructField("age", IntegerType(), True),
            StructField("sex", StringType(), True),
            StructField("workclass", StringType(), True),
            StructField("fnlwgt", IntegerType(), True),
            StructField("education", StringType(), True),
            StructField("hours_per_week", IntegerType(), True),
            StructField("label", StringType(), True)
        ])
        
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "false") \
            .schema(schema) \
            .csv("data/adult_income_sample.csv")
        
        print(f"✅ Datos cargados: {df.count()} registros")
        
        # 2. Mostrar información básica
        print("\n📋 INFORMACIÓN BÁSICA:")
        df.printSchema()
        print("\n📊 PRIMERAS 5 FILAS:")
        df.show(5, truncate=False)
        
        # 3. Análisis de distribución de la variable objetivo
        print("\n🎯 DISTRIBUCIÓN DE LA VARIABLE OBJETIVO:")
        df.groupBy("label").count().orderBy("count", ascending=False).show()
        
        # 4. Preprocesamiento
        print("\n🔧 PREPROCESAMIENTO...")
        
        # Convertir variable objetivo a binaria
        df_processed = df.withColumn(
            "income_binary", 
            when(col("label") == ">50K", 1).otherwise(0)
        )
        
        # Indexadores para variables categóricas
        sex_indexer = StringIndexer(inputCol="sex", outputCol="sex_indexed")
        workclass_indexer = StringIndexer(inputCol="workclass", outputCol="workclass_indexed")
        education_indexer = StringIndexer(inputCol="education", outputCol="education_indexed")
        
        # Codificadores one-hot
        sex_encoder = OneHotEncoder(inputCol="sex_indexed", outputCol="sex_encoded")
        workclass_encoder = OneHotEncoder(inputCol="workclass_indexed", outputCol="workclass_encoded")
        education_encoder = OneHotEncoder(inputCol="education_indexed", outputCol="education_encoded")
        
        # Ensamblador de características
        assembler = VectorAssembler(
            inputCols=["age", "fnlwgt", "hours_per_week", "sex_encoded", "workclass_encoded", "education_encoded"],
            outputCol="features"
        )
        
        # 5. Crear pipeline
        print("\n🏗️ CREANDO PIPELINE...")
        pipeline = Pipeline(stages=[
            sex_indexer,
            workclass_indexer,
            education_indexer,
            sex_encoder,
            workclass_encoder,
            education_encoder,
            assembler
        ])
        
        # Aplicar pipeline
        df_transformed = pipeline.fit(df_processed).transform(df_processed)
        
        # Seleccionar columnas necesarias
        df_final = df_transformed.select("features", "income_binary")
        
        # 6. Dividir datos
        print("\n📊 DIVIDIENDO DATOS...")
        train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)
        print(f"   • Entrenamiento: {train_data.count()} registros")
        print(f"   • Prueba: {test_data.count()} registros")
        
        # 7. Entrenar modelo
        print("\n🤖 ENTRENANDO MODELO...")
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="income_binary",
            maxIter=50,
            regParam=0.01
        )
        
        model = lr.fit(train_data)
        print("✅ Modelo entrenado exitosamente")
        
        # 8. Evaluar modelo
        print("\n📈 EVALUANDO MODELO...")
        predictions = model.transform(test_data)
        
        # Calcular métricas
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="income_binary",
            rawPredictionCol="rawPrediction"
        )
        
        multi_evaluator = MulticlassClassificationEvaluator(
            labelCol="income_binary",
            predictionCol="prediction"
        )
        
        auc = binary_evaluator.evaluate(predictions)
        accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
        precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
        recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
        f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
        
        print("\n📊 MÉTRICAS DE RENDIMIENTO:")
        print("-" * 40)
        print(f"AUC         : {auc:.4f}")
        print(f"Accuracy    : {accuracy:.4f}")
        print(f"Precision   : {precision:.4f}")
        print(f"Recall      : {recall:.4f}")
        print(f"F1-Score    : {f1:.4f}")
        
        # 9. Mostrar algunas predicciones
        print("\n🔍 EJEMPLOS DE PREDICCIONES:")
        predictions.select("income_binary", "prediction", "probability").show(10, truncate=False)
        
        # 10. Crear datos de prueba
        print("\n🎯 CREANDO DATOS DE PRUEBA...")
        test_cases = [
            (35, "Male", "Private", 200000, "Bachelors", 40, ">50K"),
            (25, "Female", "Gov", 150000, "Masters", 35, "<=50K"),
            (45, "Male", "Self-emp", 300000, "HS-grad", 50, ">50K"),
            (30, "Female", "Private", 180000, "Some-college", 30, "<=50K"),
            (55, "Male", "Gov", 250000, "Masters", 40, ">50K")
        ]
        
        test_df = spark.createDataFrame(test_cases, schema)
        test_df_processed = test_df.withColumn(
            "income_binary", 
            when(col("label") == ">50K", 1).otherwise(0)
        )
        
        # Aplicar pipeline completo
        full_pipeline = Pipeline(stages=[
            sex_indexer,
            workclass_indexer,
            education_indexer,
            sex_encoder,
            workclass_encoder,
            education_encoder,
            assembler,
            lr
        ])
        
        full_model = full_pipeline.fit(df_processed)
        test_predictions = full_model.transform(test_df_processed)
        
        print("\n📊 PREDICCIONES EN DATOS NUEVOS:")
        test_predictions.select(
            "age", "sex", "workclass", "education", "hours_per_week",
            "label", "prediction", "probability"
        ).show(truncate=False)
        
        # 11. Guardar modelo
        print("\n💾 GUARDANDO MODELO...")
        os.makedirs("models", exist_ok=True)
        full_model.write().overwrite().save("models/adult_income_model")
        print("✅ Modelo guardado en: models/adult_income_model")
        
        print("\n🎉 ¡Demo completado exitosamente!")
        print(f"📈 AUC obtenido: {auc:.4f}")
        print(f"🎯 Precisión obtenida: {accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
    finally:
        spark.stop()
        print("✅ Spark Session detenida")

if __name__ == "__main__":
    main()
