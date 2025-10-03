# 🏥 Predicción de Ausentismo en Citas Médicas con LightGBM

## 🌟 Visión General del Proyecto

Este proyecto desarrolla un modelo de Machine Learning (ML) para predecir la **inasistencia** (No-Show) a citas médicas. El objetivo principal es optimizar la gestión de recursos de la clínica [Nombre de la Clínica/Centro], priorizando los esfuerzos de contacto y recordatorio en los pacientes con mayor riesgo de ausentismo.

La solución utiliza un enfoque de clasificación, priorizando la métrica **Recall** para maximizar la detección de casos de riesgo.

## 🎯 Resultados Clave

| Métrica | Valor (Umbral de Negocio: 0.36) | Interpretación |
| :--- | :--- | :--- |
| **AUC-ROC** | ~0.67 | Moderada capacidad de discriminación general. |
| **Recall (Detección)** | ~0.82 | El modelo detecta correctamente **8 de cada 10** casos de inasistencia real. |
| **Precisión (Costo)** | ~0.45 | Aproximadamente **45%** de las alertas son inasistencias reales (el resto son falsos positivos). |

## ⚙️ Estructura del Modelo y Herramientas

* **Modelo Final:** LightGBM (LGBMClassifier)
* **Técnica de Balanceo:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Umbral de Decisión:** 0.36 (Seleccionado para maximizar el Recall)
* **Variables Clave de Inasistencia (SHAP):** `dias_espera`, `especialidad_kinesiología`, `grupo_etario_[18-44]`.

## 📂 Contenido del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `Analisis_dataset.ipynb` | Notebook principal con el ETL, preprocesamiento, EDA, modelado (LGBM, SMOTE) y análisis de resultados (SHAP, Segmentación). |
| `funciones.py` | Módulo Python con las funciones de preprocesamiento, evaluación y visualización (separando la lógica del notebook). |
| `dataset.csv` | Conjunto de datos anonimizado utilizado para el entrenamiento. |
| `requirements.txt` | Lista de dependencias de Python para replicar el entorno. |

## 🚀 Cómo Replicar el Entorno

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/ClaudioRomeroG/proyecto_ausentismo_ml
    cd proyecto_ausentismo_ml
    ```
2.  **Crear y activar el entorno virtual**:
     ```bash
    conda create -n proyecto_ausentismo python=3.10
    conda activate proyecto_ausentismo
    ```
3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  Ejecutar el notebook `Analisis_dataset.ipynb` en orden.

---

## ⏫ Paso 3: Subir todos los archivos a GitHub

Una vez que tengas `README.md` y `requirements.txt` listos, vuelve a tu terminal y ejecuta los comandos actualizados para incluir los nuevos archivos:

```bash
# Asegúrate de estar en la rama correcta (main) y en la carpeta del proyecto
git add . 

# El comando 'add .' incluye automáticamente los archivos README.md y requirements.txt

# Crea un nuevo commit con un mensaje más completo
git commit -m "Final: Añadir README, requirements.txt y visualizaciones finales."

# Sube todos los cambios al repositorio remoto
git push