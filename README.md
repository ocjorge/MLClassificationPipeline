# 🌳 Machine Learning Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10-blue)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-up%20to%20date-brightgreen)](https://pypi.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositorio contiene un pipeline completo de Machine Learning para resolver problemas de clasificación multiclase. Utiliza técnicas tradicionales como Random Forest, PCA, SMOTE y análisis de importancia de características para proporcionar una solución robusta y explicativa.

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Características Principales](#características-principales)
3. [Requisitos](#requisitos)
4. [Instalación](#instalación)
5. [Uso](#uso)
6. [Resultados](#resultados)
7. [Contribuciones](#contribuciones)
8. [Licencia](#licencia)

---

## 📚 Descripción del Proyecto

Este proyecto implementa un pipeline de Machine Learning supervisado para abordar problemas de clasificación multiclase con datos desbalanceados. Se incluyen las siguientes etapas:

- **Preprocesamiento**: Normalización de características, balanceo de clases con SMOTE y codificación de etiquetas.
- **Análisis Exploratorio**: Visualización de la estructura de los datos mediante PCA.
- **Modelado**: Entrenamiento de un modelo Random Forest con validación cruzada estratificada.
- **Evaluación**: Métricas detalladas (accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC) y matrices de confusión.
- **Interpretación**: Análisis de importancia de características basado en impureza y permutación.

---

## ✨ Características Principales

- **Balanceo de Clases**: Manejo de datos desbalanceados con SMOTE.
- **Visualización PCA**: Proyección en 2D y análisis de contribución de características.
- **Random Forest**: Modelo robusto con ajuste de hiperparámetros básicos.
- **Métricas Detalladas**: Evaluación exhaustiva con gráficos ROC, PR y matrices de confusión.
- **Guardado de Resultados**: Exporta modelos, métricas y gráficos para su uso posterior.

---

## 🔧 Requisitos

Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes dependencias:

- Python 3.8+
- Bibliotecas:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `imbalanced-learn`
  - `matplotlib`
  - `seaborn`

Puedes instalar todas las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

> **Nota**: El archivo `requirements.txt` debe contener las bibliotecas necesarias. Si no está disponible, puedes crearlo manualmente con las dependencias listadas arriba.

---

## ▶️ Instalación

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu-usuario/tu-repositorio.git
   cd tu-repositorio
   ```

2. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Coloca tus datos en el directorio raíz o actualiza la ruta en el script (`cinco.csv`).

---

## 🚀 Uso

Ejecuta el script principal para entrenar el modelo y generar resultados:

```bash
python main.py
```

### Archivos Generados

El script genera los siguientes archivos en el directorio actual:

- **Modelos**:
  - `modelo_random_forest.pkl`: Modelo entrenado.
  - `scaler.pkl`: Escalador utilizado para normalizar los datos.
  - `label_encoder.pkl`: Codificador de etiquetas.

- **Gráficos**:
  - `pca_explained_variance.png`: Varianza explicada por componentes principales.
  - `pca_projection.png`: Proyección PCA en 2D.
  - `pca_loadings.png`: Contribución de características a los componentes principales.
  - `confusion_matrix.png`: Matriz de confusión.
  - `roc_curves.png`: Curvas ROC.
  - `precision_recall_curves.png`: Curvas Precision-Recall.

- **Métricas**:
  - `metricas_resumen.csv`: Resumen de métricas globales.
  - `metricas_por_clase.csv`: Métricas por clase.
  - `feature_importance_gini.csv`: Importancia de características basada en impureza.
  - `feature_importance_permutation.csv`: Importancia de características basada en permutación.

---

## 📊 Resultados

El pipeline genera visualizaciones y métricas que facilitan la interpretación del modelo. Algunos ejemplos incluyen:

- **PCA**: Proyecciones en 2D para entender la estructura de los datos.
- **Matrices de Confusión**: Evaluación del rendimiento del modelo.
- **Curvas ROC y PR**: Análisis de la capacidad del modelo para distinguir entre clases.
- **Importancia de Características**: Identificación de las variables más relevantes.

---

## 👥 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una rama para tu nueva funcionalidad: `git checkout -b feature/nueva-funcionalidad`.
3. Realiza tus cambios y haz commit: `git commit -m "Añade nueva funcionalidad"`.
4. Envía tus cambios: `git push origin feature/nueva-funcionalidad`.
5. Abre un Pull Request.

---

## 📝 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE). Esto significa que puedes usar, modificar y distribuir el código libremente, siempre que incluyas la licencia original.

---

## 🙏 Agradecimientos

- Inspirado en las mejores prácticas de Machine Learning.
- Gracias a las bibliotecas de Python que hacen posible este trabajo.

---

Si tienes alguna pregunta o sugerencia, no dudes en abrir un issue o contactarme directamente. ¡Gracias por visitar este repositorio! 🚀

