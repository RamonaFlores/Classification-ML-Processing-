# DataPros - Clasificación de Ingresos Adultos con Spark ML

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de **clasificación binaria** utilizando **Apache Spark ML** y **Regresión Logística** para predecir si una persona gana más de 50K al año basándose en características demográficas y laborales.

### 🎯 Objetivo
Construir un modelo de machine learning que permita a DataPros predecir la categoría de ingresos (>50K o <=50K) de una persona utilizando datos demográficos y laborales.

## 📊 Datos

- **Archivo**: `adult_income_sample.csv`
- **Registros**: 2,000 registros simulados
- **Características**:
  - `age`: Edad de la persona
  - `sex`: Sexo (Male/Female)
  - `workclass`: Clase de trabajo (Private, Gov, Self-emp)
  - `fnlwgt`: Peso final (variable de ponderación)
  - `education`: Nivel educativo
  - `hours_per_week`: Horas trabajadas por semana
  - `label`: Variable objetivo (>50K o <=50K)

## 🏗️ Arquitectura del Proyecto

```
Classification-ML-Processing-/
├── data/
│   └── adult_income_sample.csv          # Datos de entrada
├── src/
│   ├── adult_income_classification.py   # Script principal
│   ├── notebook_analysis.ipynb          # Análisis interactivo
│   ├── simple_demo.py                   # Demo simplificado
│   └── metrics_extractor.py             # Extractor de métricas para dashboard
├── dashboard_data/                      # Métricas y datos para dashboard
│   ├── complete_metrics.json            # Métricas completas
│   ├── kpi_metrics.json                 # KPIs principales
│   ├── chart_data.json                  # Datos para gráficos
│   ├── findings.json                    # Findings y insights
│   ├── executive_summary.json           # Resumen ejecutivo
│   └── README_DASHBOARD.md              # Documentación del dashboard
├── Dashboard/                           # Dashboard web (React/TypeScript)
│   ├── src/
│   │   └── App.tsx                      # Componente principal
│   ├── package.json                     # Dependencias del dashboard
│   └── public/                          # Archivos estáticos
├── models/                              # Modelos entrenados (generado)
├── requirements.txt                     # Dependencias Python
└── README.MD                           # Documentación
```

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- Java 8 o superior (requerido para Spark)
- Apache Spark 3.5.0

### Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd Classification-ML-Processing-
```

2. **Crear entorno virtual**:
```bash
python3 -m venv .venv
```

3. **Activar entorno virtual**:
```bash
# Opción 1: Script automático
./activate_env.sh

# Opción 2: Manual
source .venv/bin/activate
```

4. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

5. **Verificar instalación**:
```bash
python -c "import pyspark; print('PySpark:', pyspark.__version__)"
```

## 💻 Uso

### Opción 1: Script Principal
```bash
python src/adult_income_classification.py
```

### Opción 2: Jupyter Notebook (Recomendado)
```bash
jupyter notebook src/notebook_analysis.ipynb
```

### Opción 3: Demo Simplificado
```bash
python src/simple_demo.py
```

### Opción 4: Generar Métricas para Dashboard
```bash
python src/metrics_extractor.py
```

## 🔧 Proceso de Machine Learning

### 1. Carga de Datos
- Lectura del archivo CSV con esquema optimizado
- Inspección del esquema y estadísticas descriptivas
- Análisis de valores nulos y distribuciones

### 2. Preprocesamiento de Variables Categóricas

#### StringIndexer
Convierte variables categóricas de tipo string a índices numéricos:
```python
sex_indexer = StringIndexer(inputCol="sex", outputCol="sex_indexed")
workclass_indexer = StringIndexer(inputCol="workclass", outputCol="workclass_indexed")
education_indexer = StringIndexer(inputCol="education", outputCol="education_indexed")
label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
```

#### OneHotEncoder
Convierte las variables categóricas indexadas en vectores binarios para evitar interpretaciones de orden:
```python
sex_encoder = OneHotEncoder(inputCol="sex_indexed", outputCol="sex_encoded")
workclass_encoder = OneHotEncoder(inputCol="workclass_indexed", outputCol="workclass_encoded")
education_encoder = OneHotEncoder(inputCol="education_indexed", outputCol="education_encoded")
```

### 3. Ensamblaje de Características

#### VectorAssembler
Combina todas las características en un vector único:
```python
feature_columns = [
    "age",                    # Variable numérica
    "fnlwgt",                # Variable numérica  
    "hours_per_week",        # Variable numérica
    "sex_encoded",           # Variable categórica codificada
    "workclass_encoded",     # Variable categórica codificada
    "education_encoded"      # Variable categórica codificada
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)
```

### 4. Definición y Entrenamiento del Modelo

#### Logistic Regression
```python
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label_indexed",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.8
)
```

#### Pipeline Completo
```python
full_pipeline = Pipeline(stages=[
    sex_indexer,
    workclass_indexer,
    education_indexer,
    label_indexer,
    sex_encoder,
    workclass_encoder,
    education_encoder,
    assembler,
    lr
])
```

## 📈 Evaluación del Modelo

### Métricas Utilizadas
- **AUC (Area Under Curve)**: Medida de la capacidad del modelo para distinguir entre clases
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Proporción de predicciones positivas que son correctas
- **Recall**: Proporción de casos positivos que fueron identificados correctamente
- **F1-Score**: Media armónica entre precision y recall

### División de Datos
- **Entrenamiento**: 80% de los datos
- **Prueba**: 20% de los datos
- **Semilla aleatoria**: 42 (para reproducibilidad)

## 🎨 Visualizaciones

El notebook incluye visualizaciones para:
- Distribución de la variable objetivo
- Análisis por sexo y educación
- Histogramas de variables numéricas
- Matriz de confusión
- Importancia de características (coeficientes)

## 📁 Estructura del Código

### Clase Principal: `AdultIncomeClassifier`

```python
class AdultIncomeClassifier:
    def __init__(self, app_name="AdultIncomeClassification")
    def load_data(self, file_path)
    def explore_data(self, df)
    def preprocess_data(self, df)
    def create_feature_pipeline(self)
    def train_model(self, df, pipeline)
    def evaluate_model(self, model, test_data)
    def save_model(self, model, output_path)
    def run_complete_pipeline(self, data_path, model_output_path)
```

### Pipeline de ML

1. **Carga de datos** con esquema optimizado
2. **Exploración** y análisis estadístico
3. **Preprocesamiento** de características
4. **Entrenamiento** del modelo
5. **Evaluación** con múltiples métricas
6. **Persistencia** del modelo

## 🔍 Análisis de Características

### Variables Numéricas
- `age`: Edad (impacto positivo en ingresos altos)
- `fnlwgt`: Peso final (variable de ponderación)
- `hours_per_week`: Horas trabajadas (correlación positiva)

### Variables Categóricas
- `sex`: Sexo (diferencias por género)
- `workclass`: Clase de trabajo (sector público vs privado)
- `education`: Nivel educativo (factor clave)

## 🚀 Ejecución en Producción

### Cargar Modelo Entrenado
```python
from pyspark.ml import PipelineModel

# Cargar modelo
model = PipelineModel.load("models/adult_income_model")

# Hacer predicciones
predictions = model.transform(new_data)
```

### API REST (Opcional)
El modelo puede ser integrado en una API REST usando Flask/FastAPI para predicciones en tiempo real.

## 📊 Monitoreo y Mantenimiento

### Métricas de Monitoreo
- Drift de datos
- Degradación de rendimiento
- Distribución de predicciones

### Retrenamiento
- Frecuencia recomendada: Mensual
- Trigger: Degradación de métricas > 5%

## 🔧 Configuración de Spark

El proyecto utiliza las siguientes configuraciones optimizadas de Spark:

```python
spark = SparkSession.builder \
    .appName("AdultIncomeClassification") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()
```

## 📋 Dependencias

```
pyspark==3.5.0
pandas==2.1.4
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
scikit-learn==1.3.2
```

## 🤝 Contribuciones

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👥 Equipo

- **DataPros Team** - Desarrollo y análisis
- **Machine Learning Engineers** - Implementación del modelo

## 📞 Contacto

Para preguntas o soporte, contactar al equipo de DataPros.

---

**Nota**: Este proyecto es un ejemplo educativo de clasificación binaria con Spark ML. Para uso en producción, se recomienda validación adicional y optimización de hiperparámetros.

## 🎯 Resultados Esperados

El modelo debería alcanzar:
- **AUC**: > 0.80
- **Accuracy**: > 0.75
- **F1-Score**: > 0.70

## 📚 Referencias

- [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Logistic Regression in Spark](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression)
- [Feature Engineering with Spark](https://spark.apache.org/docs/latest/ml-features.html)
