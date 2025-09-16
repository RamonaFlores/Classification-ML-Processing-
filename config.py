"""
DataPros - Configuración del Proyecto
=====================================

Archivo de configuración central para el proyecto de clasificación de ingresos adultos.

Autor: DataPros Team
Fecha: 2024
"""

import os
from pathlib import Path

# Configuración de rutas
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Archivos de datos
DATA_FILE = DATA_DIR / "adult_income_sample.csv"
MODEL_PATH = MODELS_DIR / "adult_income_model"

# Configuración de Spark
SPARK_CONFIG = {
    "app_name": "AdultIncomeClassification",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
}

# Configuración del modelo
MODEL_CONFIG = {
    "algorithm": "LogisticRegression",
    "max_iter": 100,
    "reg_param": 0.01,
    "elastic_net_param": 0.8,
    "train_test_split": 0.8,
    "random_seed": 42
}

# Configuración de características
FEATURE_CONFIG = {
    "numerical_features": ["age", "fnlwgt", "hours_per_week"],
    "categorical_features": ["sex", "workclass", "education"],
    "target_column": "label",
    "target_mapping": {">50K": 1, "<=50K": 0}
}

# Configuración de evaluación
EVALUATION_CONFIG = {
    "metrics": ["auc", "accuracy", "precision", "recall", "f1"],
    "confidence_thresholds": {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    }
}

# Configuración de logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "classification.log"
}

# Crear directorios necesarios
def create_directories():
    """
    Crea los directorios necesarios para el proyecto
    """
    directories = [DATA_DIR, SRC_DIR, MODELS_DIR, DOCS_DIR, LOGGING_CONFIG["file"].parent]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Directorios del proyecto creados/verificados")

# Configuración de datos de prueba
TEST_DATA = [
    # (age, sex, workclass, fnlwgt, education, hours_per_week, expected_label)
    (29, "Male", "Private", 180000, "Masters", 45, ">50K"),
    (42, "Female", "Gov", 220000, "Bachelors", 40, ">50K"),
    (58, "Male", "Self-emp", 350000, "HS-grad", 55, ">50K"),
    (24, "Female", "Private", 95000, "11th", 25, "<=50K"),
    (47, "Male", "Private", 280000, "Bachelors", 50, ">50K"),
    (31, "Female", "Gov", 160000, "Some-college", 35, "<=50K"),
    (52, "Female", "Self-emp", 320000, "Masters", 48, ">50K"),
    (38, "Male", "Private", 200000, "Assoc", 42, "<=50K"),
    (61, "Male", "Gov", 400000, "Masters", 40, ">50K")
]

# Configuración de visualización
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "style": "seaborn-v0_8",
    "palette": "husl",
    "dpi": 300
}

# Configuración de exportación
EXPORT_CONFIG = {
    "formats": ["csv", "json", "parquet"],
    "include_probabilities": True,
    "include_confidence": True
}

if __name__ == "__main__":
    # Crear directorios al ejecutar el archivo
    create_directories()
    print("📋 Configuración del proyecto cargada correctamente")
