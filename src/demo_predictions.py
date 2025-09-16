"""
DataPros - Demo de Predicciones con el Modelo Entrenado
=======================================================

Este script demuestra cómo cargar el modelo entrenado y hacer predicciones
sobre nuevos datos.

Autor: DataPros Team
Fecha: 2024
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when

def load_trained_model(model_path):
    """
    Carga el modelo entrenado desde el archivo guardado
    
    Args:
        model_path (str): Ruta al modelo guardado
        
    Returns:
        PipelineModel: Modelo cargado
    """
    print(f"🔄 Cargando modelo desde: {model_path}")
    model = PipelineModel.load(model_path)
    print("✅ Modelo cargado exitosamente")
    return model

def create_sample_data(spark):
    """
    Crea datos de muestra para demostrar las predicciones
    
    Args:
        spark (SparkSession): Sesión de Spark
        
    Returns:
        DataFrame: Datos de muestra
    """
    # Definir esquema
    schema = StructType([
        StructField("age", IntegerType(), True),
        StructField("sex", StringType(), True),
        StructField("workclass", StringType(), True),
        StructField("fnlwgt", IntegerType(), True),
        StructField("education", StringType(), True),
        StructField("hours_per_week", IntegerType(), True),
        StructField("label", StringType(), True)  # No se usará para predicción
    ])
    
    # Crear datos de muestra
    sample_data = [
        (35, "Male", "Private", 215646, "Bachelors", 40, ">50K"),      # Caso 1
        (28, "Female", "Gov", 185432, "Masters", 35, "<=50K"),         # Caso 2
        (45, "Male", "Self-emp", 320000, "HS-grad", 50, ">50K"),       # Caso 3
        (22, "Female", "Private", 120000, "Some-college", 30, "<=50K"), # Caso 4
        (55, "Male", "Gov", 280000, "Masters", 45, ">50K"),            # Caso 5
    ]
    
    df = spark.createDataFrame(sample_data, schema)
    print("✅ Datos de muestra creados")
    return df

def make_predictions(model, data):
    """
    Hace predicciones usando el modelo cargado
    
    Args:
        model (PipelineModel): Modelo entrenado
        data (DataFrame): Datos para predecir
        
    Returns:
        DataFrame: Datos con predicciones
    """
    print("🔄 Haciendo predicciones...")
    predictions = model.transform(data)
    print("✅ Predicciones completadas")
    return predictions

def interpret_predictions(predictions):
    """
    Interpreta y muestra las predicciones de manera legible
    
    Args:
        predictions (DataFrame): DataFrame con predicciones
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType, DoubleType
    
    # Función para interpretar predicciones
    def interpret_prediction(pred_idx):
        return ">50K" if pred_idx == 1.0 else "<=50K"
    
    def interpret_label(label_idx):
        return ">50K" if label_idx == 1.0 else "<=50K"
    
    def extract_prob_high(probability_vector):
        return float(probability_vector[1])  # Probabilidad de >50K
    
    # Crear UDFs
    interpret_pred_udf = udf(interpret_prediction, StringType())
    interpret_label_udf = udf(interpret_label, StringType())
    extract_prob_udf = udf(extract_prob_high, DoubleType())
    
    # Agregar columnas interpretables
    results = predictions.withColumn(
        "prediction_interpreted", interpret_pred_udf("prediction")
    ).withColumn(
        "label_interpreted", interpret_label_udf("label_indexed")
    ).withColumn(
        "prob_>50K", extract_prob_udf("probability")
    )
    
    print("\n" + "=" * 80)
    print("📊 RESULTADOS DE PREDICCIONES")
    print("=" * 80)
    
    # Mostrar resultados detallados
    results.select(
        "age", "sex", "workclass", "education", "hours_per_week",
        "label_interpreted", "prediction_interpreted", "prob_>50K"
    ).show(truncate=False)
    
    # Análisis de precisión
    correct_predictions = results.filter(
        col("label_indexed") == col("prediction")
    ).count()
    
    total_predictions = results.count()
    accuracy = correct_predictions / total_predictions * 100
    
    print(f"\n📈 ANÁLISIS DE PRECISIÓN:")
    print(f"   • Predicciones correctas: {correct_predictions}/{total_predictions}")
    print(f"   • Precisión: {accuracy:.2f}%")
    
    # Mostrar casos incorrectos
    incorrect = results.filter(col("label_indexed") != col("prediction"))
    if incorrect.count() > 0:
        print(f"\n❌ CASOS INCORRECTOS:")
        incorrect.select(
            "age", "sex", "workclass", "education", "hours_per_week",
            "label_interpreted", "prediction_interpreted", "prob_>50K"
        ).show(truncate=False)

def main():
    """
    Función principal para ejecutar la demostración
    """
    # Inicializar Spark
    spark = SparkSession.builder \
        .appName("AdultIncomePredictionDemo") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        print("🚀 DEMO DE PREDICCIONES - DataPros Adult Income Classification")
        print("=" * 70)
        
        # Cargar modelo
        model_path = "../models/adult_income_model"
        model = load_trained_model(model_path)
        
        # Crear datos de muestra
        sample_data = create_sample_data(spark)
        
        # Mostrar datos de entrada
        print("\n📋 DATOS DE ENTRADA:")
        sample_data.select("age", "sex", "workclass", "education", "hours_per_week").show(truncate=False)
        
        # Hacer predicciones
        predictions = make_predictions(model, sample_data)
        
        # Interpretar y mostrar resultados
        interpret_predictions(predictions)
        
        print("\n🎉 ¡Demo completado exitosamente!")
        print("\n💡 NOTAS:")
        print("   • El modelo predice si una persona gana >50K o <=50K")
        print("   • Las probabilidades indican la confianza del modelo")
        print("   • Valores altos de prob_>50K sugieren mayor probabilidad de ingresos altos")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        print("💡 Asegúrate de que el modelo esté entrenado y guardado correctamente")
    finally:
        spark.stop()
        print("✅ Spark Session detenida")

if __name__ == "__main__":
    main()
