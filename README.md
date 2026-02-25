# 🌳 Árbol de Decisión - Predicción de Medicamentos

Análisis completo y profesional de un **Árbol de Decisión** usando el dataset de Kaggle: **pablomgomez21/drugs-a-b-c-x-y-for-decision-trees**

## 📋 Descripción

Este proyecto implementa un modelo de clasificación basado en **Decision Tree** para predecir qué medicamento (A, B, C, X, o Y) debe prescribirse a un paciente basándose en características como:
- Edad
- Género  
- Presión arterial
- Colesterol
- Índice de sodio/potasio en sangre

## 🎯 Objetivos

- ✅ Exploración y análisis de datos (EDA)
- ✅ Preprocesamiento y codificación de variables
- ✅ Construcción de modelo base y optimizado
- ✅ Optimización de hiperparámetros con GridSearchCV
- ✅ Validación cruzada (5-fold)
- ✅ Evaluación detallada con múltiples métricas
- ✅ Visualización de resultados
- ✅ Extracción de reglas del árbol

## 📁 Estructura del Proyecto

```
Arbol-Decision/
├── Arbol_de_Decision.ipynb    # Notebook principal con análisis completo
├── requirements.txt            # Dependencias del proyecto
├── kaggle.json                 # Credenciales de Kaggle (no incluido)
├── README.md                   # Este archivo
└── LICENSE                     # Licencia del proyecto
```

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/Arbol-Decision.git
cd Arbol-Decision
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar Kaggle

1. Ve a tu cuenta de Kaggle → Settings → API → "Create New API Token"
2. Descarga el archivo `kaggle.json`
3. Coloca `kaggle.json` en el directorio raíz del proyecto

### 4. Ejecutar el notebook

```bash
jupyter notebook Arbol_de_Decision.ipynb
```

El notebook descargará automáticamente el dataset y ejecutará todo el análisis.

## 📚 Contenido del Notebook

El notebook está organizado en **16 secciones didácticas**:

1. **Introducción** - ¿Qué es un Árbol de Decisión?
2. **Importación de Librerías** - Herramientas necesarias
3. **Descarga y Carga del Dataset** - Desde Kaggle
4. **Análisis Exploratorio (EDA)** - Exploración de datos
5. **Preprocesamiento** - Codificación de variables
6. **División del Dataset** - Train/Test split (80/20)
7. **Modelo Base** - DecisionTree sin optimizar
8. **Optimización de Hiperparámetros** - GridSearchCV
9. **Validación Cruzada** - Evaluación robusta
10. **Evaluación Detallada** - Múltiples métricas
11. **Matriz de Confusión** - Análisis de errores
12. **Visualización del Árbol** - Gráfico del modelo
13. **Importancia de Features** - Variables más relevantes
14. **Comparación de Modelos** - Base vs Optimizado
15. **Reglas del Árbol** - Interpretabilidad
16. **Resumen Final** - Resultados y archivos generados

## 📊 Resultados

El notebook generará automáticamente:

### Modelos
- `decision_tree_model.pkl` - Modelo optimizado entrenado
- `label_encoders.pkl` - Codificadores de variables

### Visualizaciones
- `arbol_decision.png` - Visualización del árbol de decisión
- `importancia_features.png` - Gráfico de importancia de variables
- `matriz_confusion.png` - Heatmap de la matriz de confusión
- `comparacion_modelos.png` - Comparación Base vs Optimizado

### Análisis
- `arbol_reglas.txt` - Reglas del árbol en formato texto

### 📁 Ubicación de los archivos

Todos los archivos se guardan en el **directorio raíz del proyecto**, es decir:
```
\\wsl.localhost\Ubuntu\home\alex\proyects\Arbol-Decision\
```

Durante la ejecución, el notebook muestra la ruta completa donde se guardan los archivos.

### Métricas Esperadas
```
Accuracy:   ~0.95-1.00
Precision:  ~0.95-1.00
Recall:     ~0.95-1.00
F1-Score:   ~0.95-1.00
```

## 🤔 ¿Es necesaria la optimización?

El notebook incluye optimización de hiperparámetros con GridSearchCV. Sin embargo:

**Cuando el modelo base tiene >95% de accuracy:**
- La optimización puede no mostrar mejoras significativas
- Esto es **normal y positivo**: indica que el dataset tiene patrones claros
- El modelo base ya captura bien la estructura de los datos

**El notebook detecta esto automáticamente y:**
- ✅ Muestra un mensaje indicando que el rendimiento ya es excelente
- ✅ Continúa con la optimización para fines didácticos
- ✅ Explica por qué no hay mejora significativa en la comparación

**Beneficios de mantener la optimización:**
- 📚 Aprendizaje: Demuestra el proceso completo
- 🔍 Validación: Confirma que los parámetros por defecto son óptimos
- 🛡️ Prevención de overfitting: Puede regularizar mejor el modelo

## 🔧 Hiperparámetros Optimizados

El GridSearchCV explora:
```python
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}
```
**Total:** 192 combinaciones evaluadas con validación cruzada 5-fold

## 💡 Cómo usar el modelo guardado

```python
import joblib
import pandas as pd

# Cargar modelo y encoders
modelo = joblib.load('decision_tree_model.pkl')
encoders = joblib.load('label_encoders.pkl')

# Hacer predicciones con nuevos datos
# (primero codificar las variables categóricas con los encoders)
prediccion = modelo.predict(nuevos_datos)
probabilidades = modelo.predict_proba(nuevos_datos)
```

## 🛠️ Dependencias Principales

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- kaggle

Ver `requirements.txt` para todas las dependencias y versiones.

## 📝 Características del Proyecto

- ✅ Código profesional y documentado
- ✅ Explicaciones didácticas en cada sección
- ✅ Reproducible con random_state=42
- ✅ Validación cruzada implementada
- ✅ Múltiples métricas de evaluación
- ✅ Visualizaciones de calidad profesional
- ✅ Modelo interpretable y explicable
- ✅ Listo para GitHub y portafolio

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para más detalles.

---
**Última actualización:** Febrero 2026  
**Estado:** ✅ Completado
