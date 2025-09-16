#!/usr/bin/env python3
"""
DataPros - Extractor de Métricas para Dashboard
==============================================

Este script ejecuta el análisis completo y extrae todas las métricas
importantes para el dashboard, incluyendo findings y insights.

Autor: DataPros Team
Fecha: 2024
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan, isnull, avg, stddev, min as spark_min, max as spark_max
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import json
import os
from datetime import datetime

class MetricsExtractor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("DataProsMetricsExtractor") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        self.metrics = {}
        self.findings = []
        
    def load_and_analyze_data(self):
        """Cargar y analizar los datos"""
        print("📊 Cargando y analizando datos...")
        
        # Cargar datos
        schema = StructType([
            StructField("age", IntegerType(), True),
            StructField("sex", StringType(), True),
            StructField("workclass", StringType(), True),
            StructField("fnlwgt", IntegerType(), True),
            StructField("education", StringType(), True),
            StructField("hours_per_week", IntegerType(), True),
            StructField("label", StringType(), True)
        ])
        
        df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "false") \
            .schema(schema) \
            .csv("data/adult_income_sample.csv")
        
        # Métricas básicas de datos
        total_records = df.count()
        self.metrics["data_overview"] = {
            "total_records": total_records,
            "total_features": len(df.columns),
            "data_quality": "high" if total_records > 0 else "low"
        }
        
        # Análisis de distribución de la variable objetivo
        label_dist = df.groupBy("label").count().collect()
        label_dist_dict = {row["label"]: row["count"] for row in label_dist}
        
        high_income_count = label_dist_dict.get(">50K", 0)
        low_income_count = label_dist_dict.get("<=50K", 0)
        
        self.metrics["target_distribution"] = {
            "high_income_count": high_income_count,
            "low_income_count": low_income_count,
            "high_income_percentage": (high_income_count / total_records) * 100,
            "low_income_percentage": (low_income_count / total_records) * 100,
            "class_balance": "balanced" if abs(high_income_count - low_income_count) < total_records * 0.1 else "imbalanced"
        }
        
        # Análisis por género
        gender_analysis = df.groupBy("sex", "label").count().collect()
        gender_metrics = {}
        for row in gender_analysis:
            sex = row["sex"]
            label = row["label"]
            count = row["count"]
            if sex not in gender_metrics:
                gender_metrics[sex] = {}
            gender_metrics[sex][label] = count
        
        self.metrics["gender_analysis"] = gender_metrics
        
        # Análisis por educación
        education_analysis = df.groupBy("education", "label").count().collect()
        education_metrics = {}
        for row in education_analysis:
            edu = row["education"]
            label = row["label"]
            count = row["count"]
            if edu not in education_metrics:
                education_metrics[edu] = {}
            education_metrics[edu][label] = count
        
        self.metrics["education_analysis"] = education_metrics
        
        # Análisis por clase de trabajo
        workclass_analysis = df.groupBy("workclass", "label").count().collect()
        workclass_metrics = {}
        for row in workclass_analysis:
            work = row["workclass"]
            label = row["label"]
            count = row["count"]
            if work not in workclass_metrics:
                workclass_metrics[work] = {}
            workclass_metrics[work][label] = count
        
        self.metrics["workclass_analysis"] = workclass_metrics
        
        # Estadísticas numéricas
        numeric_stats = df.select("age", "fnlwgt", "hours_per_week").describe().collect()
        numeric_metrics = {}
        for stat in numeric_stats:
            metric = stat["summary"]
            if metric in ["mean", "stddev", "min", "max"]:
                numeric_metrics[metric] = {
                    "age": float(stat["age"]),
                    "fnlwgt": float(stat["fnlwgt"]),
                    "hours_per_week": float(stat["hours_per_week"])
                }
        
        self.metrics["numeric_statistics"] = numeric_metrics
        
        return df
    
    def train_and_evaluate_model(self, df):
        """Entrenar y evaluar el modelo"""
        print("🤖 Entrenando y evaluando modelo...")
        
        # Preprocesamiento
        df_processed = df.withColumn(
            "income_binary", 
            when(col("label") == ">50K", 1).otherwise(0)
        )
        
        # Pipeline de características
        sex_indexer = StringIndexer(inputCol="sex", outputCol="sex_indexed")
        workclass_indexer = StringIndexer(inputCol="workclass", outputCol="workclass_indexed")
        education_indexer = StringIndexer(inputCol="education", outputCol="education_indexed")
        
        sex_encoder = OneHotEncoder(inputCol="sex_indexed", outputCol="sex_encoded")
        workclass_encoder = OneHotEncoder(inputCol="workclass_indexed", outputCol="workclass_encoded")
        education_encoder = OneHotEncoder(inputCol="education_indexed", outputCol="education_encoded")
        
        assembler = VectorAssembler(
            inputCols=["age", "fnlwgt", "hours_per_week", "sex_encoded", "workclass_encoded", "education_encoded"],
            outputCol="features"
        )
        
        # Modelo
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="income_binary",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.8
        )
        
        # Pipeline completo
        pipeline = Pipeline(stages=[
            sex_indexer, workclass_indexer, education_indexer,
            sex_encoder, workclass_encoder, education_encoder,
            assembler, lr
        ])
        
        # Entrenar modelo
        model = pipeline.fit(df_processed)
        
        # Evaluar modelo
        df_transformed = model.transform(df_processed)
        train_data, test_data = df_transformed.select("features", "income_binary").randomSplit([0.8, 0.2], seed=42)
        
        # Reentrenar solo el modelo de regresión logística
        lr_model = LogisticRegression(
            featuresCol="features",
            labelCol="income_binary",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.8
        ).fit(train_data)
        
        predictions = lr_model.transform(test_data)
        
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
        
        # Matriz de confusión
        confusion_matrix = predictions.groupBy("income_binary", "prediction").count().collect()
        cm_dict = {}
        for row in confusion_matrix:
            actual = int(row["income_binary"])
            predicted = int(row["prediction"])
            count = row["count"]
            cm_dict[f"{actual}_{predicted}"] = count
        
        # Métricas del modelo
        self.metrics["model_performance"] = {
            "auc": float(auc),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm_dict,
            "train_records": train_data.count(),
            "test_records": test_data.count()
        }
        
        # Análisis de confianza
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DoubleType
        
        def extract_prob_high(probability_vector):
            return float(probability_vector[1])
        
        extract_prob_udf = udf(extract_prob_high, DoubleType())
        predictions_with_prob = predictions.withColumn("prob_high_income", extract_prob_udf("probability"))
        
        # Estadísticas de confianza
        confidence_stats = predictions_with_prob.select("prob_high_income").describe().collect()
        confidence_metrics = {}
        for stat in confidence_stats:
            metric = stat["summary"]
            if metric in ["mean", "stddev", "min", "max"]:
                confidence_metrics[metric] = float(stat["prob_high_income"])
        
        self.metrics["confidence_analysis"] = confidence_metrics
        
        # Análisis de errores
        correct_predictions = predictions.filter(col("income_binary") == col("prediction")).count()
        total_predictions = predictions.count()
        error_rate = (total_predictions - correct_predictions) / total_predictions
        
        self.metrics["error_analysis"] = {
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "error_rate": float(error_rate),
            "success_rate": float(correct_predictions / total_predictions)
        }
        
        return model, predictions
    
    def generate_findings(self):
        """Generar findings y insights"""
        print("🔍 Generando findings y insights...")
        
        findings = []
        
        # Finding 1: Distribución de datos
        target_dist = self.metrics["target_distribution"]
        if target_dist["class_balance"] == "balanced":
            findings.append({
                "category": "Data Quality",
                "title": "Distribución Balanceada de Clases",
                "description": f"Las clases están bien balanceadas: {target_dist['high_income_percentage']:.1f}% >50K vs {target_dist['low_income_percentage']:.1f}% <=50K",
                "impact": "positive",
                "recommendation": "No se requiere balanceo de clases"
            })
        else:
            findings.append({
                "category": "Data Quality",
                "title": "Distribución Desbalanceada de Clases",
                "description": f"Las clases están desbalanceadas: {target_dist['high_income_percentage']:.1f}% >50K vs {target_dist['low_income_percentage']:.1f}% <=50K",
                "impact": "warning",
                "recommendation": "Considerar técnicas de balanceo de clases"
            })
        
        # Finding 2: Rendimiento del modelo
        model_perf = self.metrics["model_performance"]
        if model_perf["auc"] > 0.8:
            findings.append({
                "category": "Model Performance",
                "title": "Excelente Rendimiento del Modelo",
                "description": f"AUC de {model_perf['auc']:.3f} indica excelente capacidad de discriminación",
                "impact": "positive",
                "recommendation": "Modelo listo para producción"
            })
        elif model_perf["auc"] > 0.7:
            findings.append({
                "category": "Model Performance",
                "title": "Buen Rendimiento del Modelo",
                "description": f"AUC de {model_perf['auc']:.3f} indica buena capacidad de discriminación",
                "impact": "positive",
                "recommendation": "Considerar optimizaciones menores"
            })
        else:
            findings.append({
                "category": "Model Performance",
                "title": "Rendimiento Moderado del Modelo",
                "description": f"AUC de {model_perf['auc']:.3f} indica capacidad de discriminación moderada",
                "impact": "warning",
                "recommendation": "Revisar características y parámetros del modelo"
            })
        
        # Finding 3: Análisis de género
        gender_analysis = self.metrics["gender_analysis"]
        if "Male" in gender_analysis and "Female" in gender_analysis:
            male_high = gender_analysis["Male"].get(">50K", 0)
            male_total = sum(gender_analysis["Male"].values())
            female_high = gender_analysis["Female"].get(">50K", 0)
            female_total = sum(gender_analysis["Female"].values())
            
            male_rate = (male_high / male_total) * 100 if male_total > 0 else 0
            female_rate = (female_high / female_total) * 100 if female_total > 0 else 0
            
            if abs(male_rate - female_rate) > 10:
                findings.append({
                    "category": "Bias Analysis",
                    "title": "Diferencias Significativas por Género",
                    "description": f"Tasa de ingresos altos: Hombres {male_rate:.1f}% vs Mujeres {female_rate:.1f}%",
                    "impact": "warning",
                    "recommendation": "Revisar posibles sesgos en el modelo"
                })
        
        # Finding 4: Análisis de educación
        education_analysis = self.metrics["education_analysis"]
        education_rates = {}
        for edu, counts in education_analysis.items():
            total = sum(counts.values())
            high_income = counts.get(">50K", 0)
            rate = (high_income / total) * 100 if total > 0 else 0
            education_rates[edu] = rate
        
        # Encontrar la educación con mayor tasa de ingresos altos
        best_education = max(education_rates.items(), key=lambda x: x[1])
        findings.append({
            "category": "Feature Analysis",
            "title": "Educación con Mayor Impacto en Ingresos",
            "description": f"'{best_education[0]}' tiene la mayor tasa de ingresos >50K: {best_education[1]:.1f}%",
            "impact": "insight",
            "recommendation": "Considerar esta variable como predictor clave"
        })
        
        # Finding 5: Análisis de horas trabajadas
        numeric_stats = self.metrics["numeric_statistics"]
        avg_hours = numeric_stats["mean"]["hours_per_week"]
        if avg_hours > 45:
            findings.append({
                "category": "Work Pattern",
                "title": "Alto Promedio de Horas Trabajadas",
                "description": f"Promedio de {avg_hours:.1f} horas por semana sugiere alta dedicación laboral",
                "impact": "insight",
                "recommendation": "Analizar correlación con ingresos altos"
            })
        
        # Finding 6: Análisis de confianza
        confidence_analysis = self.metrics["confidence_analysis"]
        avg_confidence = confidence_analysis["mean"]
        if avg_confidence > 0.7:
            findings.append({
                "category": "Model Confidence",
                "title": "Alta Confianza del Modelo",
                "description": f"Confianza promedio de {avg_confidence:.3f} indica predicciones seguras",
                "impact": "positive",
                "recommendation": "Modelo confiable para decisiones automáticas"
            })
        elif avg_confidence < 0.6:
            findings.append({
                "category": "Model Confidence",
                "title": "Baja Confianza del Modelo",
                "description": f"Confianza promedio de {avg_confidence:.3f} indica predicciones inciertas",
                "impact": "warning",
                "recommendation": "Revisar modelo y características"
            })
        
        self.findings = findings
    
    def generate_dashboard_data(self):
        """Generar datos para el dashboard"""
        print("📊 Generando datos para dashboard...")
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "project": "DataPros Adult Income Classification",
            "version": "1.0.0",
            "metrics": self.metrics,
            "findings": self.findings,
            "summary": {
                "total_records": self.metrics["data_overview"]["total_records"],
                "model_auc": self.metrics["model_performance"]["auc"],
                "model_accuracy": self.metrics["model_performance"]["accuracy"],
                "total_findings": len(self.findings),
                "critical_findings": len([f for f in self.findings if f["impact"] == "warning"]),
                "positive_findings": len([f for f in self.findings if f["impact"] == "positive"])
            }
        }
        
        return dashboard_data
    
    def save_metrics(self, dashboard_data):
        """Guardar métricas en archivos JSON"""
        print("💾 Guardando métricas...")
        
        # Crear directorio de salida
        os.makedirs("dashboard_data", exist_ok=True)
        
        # Guardar datos completos
        with open("dashboard_data/complete_metrics.json", "w") as f:
            json.dump(dashboard_data, f, indent=2)
        
        # Guardar solo métricas
        with open("dashboard_data/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        # Guardar solo findings
        with open("dashboard_data/findings.json", "w") as f:
            json.dump(self.findings, f, indent=2)
        
        # Guardar resumen
        with open("dashboard_data/summary.json", "w") as f:
            json.dump(dashboard_data["summary"], f, indent=2)
        
        print("✅ Métricas guardadas en dashboard_data/")
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("🚀 Iniciando análisis completo para dashboard...")
        
        try:
            # 1. Cargar y analizar datos
            df = self.load_and_analyze_data()
            
            # 2. Entrenar y evaluar modelo
            model, predictions = self.train_and_evaluate_model(df)
            
            # 3. Generar findings
            self.generate_findings()
            
            # 4. Generar datos para dashboard
            dashboard_data = self.generate_dashboard_data()
            
            # 5. Guardar métricas
            self.save_metrics(dashboard_data)
            
            # 6. Mostrar resumen
            self.print_summary(dashboard_data)
            
            return dashboard_data
            
        except Exception as e:
            print(f"❌ Error durante el análisis: {str(e)}")
            raise
        finally:
            self.spark.stop()
    
    def print_summary(self, dashboard_data):
        """Imprimir resumen de resultados"""
        print("\n" + "="*60)
        print("📊 RESUMEN DE MÉTRICAS PARA DASHBOARD")
        print("="*60)
        
        summary = dashboard_data["summary"]
        print(f"📈 Total de registros: {summary['total_records']:,}")
        print(f"🎯 AUC del modelo: {summary['model_auc']:.4f}")
        print(f"🎯 Precisión del modelo: {summary['model_accuracy']:.4f}")
        print(f"🔍 Total de findings: {summary['total_findings']}")
        print(f"⚠️  Findings críticos: {summary['critical_findings']}")
        print(f"✅ Findings positivos: {summary['positive_findings']}")
        
        print(f"\n📁 Archivos generados:")
        print(f"   • dashboard_data/complete_metrics.json")
        print(f"   • dashboard_data/metrics.json")
        print(f"   • dashboard_data/findings.json")
        print(f"   • dashboard_data/summary.json")
        
        print(f"\n🔍 PRINCIPALES FINDINGS:")
        for i, finding in enumerate(self.findings[:3], 1):
            print(f"   {i}. {finding['title']} ({finding['category']})")
            print(f"      {finding['description']}")
            print(f"      Impacto: {finding['impact']}")
            print()

def main():
    """Función principal"""
    extractor = MetricsExtractor()
    dashboard_data = extractor.run_complete_analysis()
    return dashboard_data

if __name__ == "__main__":
    main()
