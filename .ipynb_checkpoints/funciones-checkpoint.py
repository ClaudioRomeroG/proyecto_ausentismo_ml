import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, average_precision_score,
    precision_score, precision_recall_curve, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, GridSearchCV) 
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE 
import re
import shap
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


def preprocesamiento_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza, crea nuevas variables (inasistencia, edad, espera, turno),
    el filtrado IQR, identifica los NAs y selecciona las columnas finales para el modelo.
    """
    df = df.copy()
    
   # 1. Inicializacion de variables
    df['inasistencia'] = np.nan
    df['grupo_etario'] = 'missing'
    df['dias_espera'] = -1
    df['turno_cita'] = 'missing'
    
    # 2. Variable inasistencia (derivada de 'Estado Cita')
    if 'Estado Cita' in df.columns:
        ausentismo_estados = [
            'No asiste', 'Ausente', 'Anulado', 'Anulada', 'Cambio de fecha',
            'No confirmado', 'Anulado por pcte. via email', 'Anulado vía validación',
            'Anulado por sesiones en conflicto', 
        ]
        df['inasistencia'] = df['Estado Cita'].isin(ausentismo_estados).astype(int)

    # 3. Grupo Etario (derivada de varaible 'Edad', creada con la fecha de nacimiento y fecha de hoy)
    if 'Fecha de nac.' in df.columns:
        df['Fecha de nac.'] = pd.to_datetime(df['Fecha de nac.'], errors='coerce', format='%Y-%m-%d')
        hoy = pd.to_datetime('today')
        df['Edad'] = ((hoy - df['Fecha de nac.']).dt.days / 365.25).fillna(0).astype(int)
        bins = [0, 12, 18, 45, 65, np.inf]
        labels = ['<12', '13-17', '18-44', '45-64', '>65']
        df['grupo_etario'] = pd.cut(df['Edad'], bins=bins, labels=labels, right=False).astype(str) 

    # 4. Dias_espera (derivada de 'fecha cita' y 'fecha creación cita')
    if 'Fecha Cita' in df.columns and 'Fecha de creación de cita' in df.columns:
        df['Fecha Cita'] = pd.to_datetime(df['Fecha Cita'])
        df['Fecha de creación de cita'] = pd.to_datetime(df['Fecha de creación de cita'])
        df['dias_espera'] = (df['Fecha Cita'] - df['Fecha de creación de cita']).dt.days
        df['dias_espera'] = np.maximum(0, df['dias_espera'])

    # 5. Turno Cita (derivada de 'Hora Inicio Cita')
    if 'Hora Inicio Cita' in df.columns:
        H_1000 = pd.to_datetime('10:00', format='%H:%M').time()
        H_1300 = pd.to_datetime('13:00', format='%H:%M').time()
        H_1700 = pd.to_datetime('17:00', format='%H:%M').time()
        
        try:
            df['hora_comparacion'] = pd.to_datetime(df['Hora Inicio Cita'].astype(str).str[:5], format='%H:%M', errors='coerce').dt.time
            
            def asignar_turno(hora):
                if pd.isna(hora): return 'Fuera de Horario'
                if hora < H_1000: return 'Mañana'
                elif hora < H_1300: return 'Medio día'
                elif hora < H_1700: return 'Tarde'
                elif hora >= H_1700: return 'Noche'
                return 'Fuera de Horario'
    
            df['turno_cita'] = df['hora_comparacion'].apply(asignar_turno)
            df = df.drop(['hora_comparacion'], axis=1, errors='ignore')
        except Exception:
            pass
            
    # 6. Renombre de variables
    df = df.rename(columns={
        'Especialidad Profesional': 'especialidad',
        'Convenio Paciente': 'convenio'
    })

    # 7. Rango intercuartílico
    if 'dias_espera' in df.columns:
        
        # Copia para cálculo IQR solo con datos válidos de espera (>= 0)
        df_iqr = df[df['dias_espera'] >= 0].copy()
        
        if not df_iqr.empty:
            Q1 = df_iqr['dias_espera'].quantile(0.25)
            Q3 = df_iqr['dias_espera'].quantile(0.75)
            IQR = Q3 - Q1
            limite_superior = Q3 + 1.5 * IQR
    
            filas_antes_iqr = len(df_iqr)
            
            df_iqr = df_iqr[df_iqr['dias_espera'] <= limite_superior]
            
            filas_despues_iqr = len(df_iqr)
            
            print(f"📊 Filtro IQR aplicado a 'dias_espera'. Límite Superior: {limite_superior:.0f} días.")
            print(f"   Filas eliminadas por IQR: {filas_antes_iqr - filas_despues_iqr} ({((filas_antes_iqr - filas_despues_iqr) / filas_antes_iqr)*100:.2f}%)")
        
            df = df_iqr # Reasignar el DataFrame filtrado
            
    # 8. Preparacion e identificación de nulos 
    columnas_finales_modelo = [
        'especialidad', 'grupo_etario', 'convenio', 
        'dias_espera', 'turno_cita', 'inasistencia'
    ]
    
    df_temp = df[columnas_finales_modelo].copy()
    
    filas_con_na_antes_drop = len(df_temp) - len(df_temp.dropna())
    
    if filas_con_na_antes_drop > 0:
        resumen_na = df_temp.isna().sum().rename('NA Count')
        resumen_na = resumen_na[resumen_na > 0].sort_values(ascending=False)
        
        print("\n⚠️ ALERTA DE DATOS NULOS (Filas a eliminar en el Dropna final):")
        print(f"   Filas a eliminar por NA: {filas_con_na_antes_drop}")
        print("   Columnas responsables de los NAs (por conteo):")
        print(resumen_na)
        
    df_final = df_temp.dropna()

    print("\n✅ Preprocesamiento completado. El DataFrame final 'df_final' (limpio y preprocesado) está listo.")
    print("\nColumnas finales y sus tipos de datos (Listas para el modelo):")
    df_final.info()
    print("\nConteo de clases en el DataFrame con inasistencias:")
    print(df_final['inasistencia'].value_counts())
    
    return df_final

def aplicar_transformaciones_finales(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, random_state=42):
    """
    Aplica el ColumnTransformer (Escalado + OHE) a los datos de entrenamiento y prueba,
    y luego aplica SMOTE para el balanceo del set de entrenamiento.
    
    IMPORTANTE: Incluye limpieza ROBUSTA de nombres de columnas para LightGBM.
    """
    # Columnas a transformar
    columnas_categoricas = ['especialidad', 'grupo_etario', 'convenio', 'turno_cita']
    columnas_numericas = ['dias_espera'] 

    # 1. Definir ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas), 
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), columnas_categoricas)
        ],
        remainder='drop' 
    )

    # 2. Aplicar Transformación
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 3. Reconstrucción y corrección de nombres
    feature_names = preprocessor.get_feature_names_out()
    
    safe_feature_names = []
    for name in feature_names:
        
        # 3a. Limpiar prefijos del ColumnTransformer ('num__' o 'cat__')
        cleaned_name = name.replace('num__', '').replace('cat__', '')
        
        # 3b. Reemplazar cualquier caracter que NO sea alfanumérico o guion bajo (\w) por un guion bajo
        cleaned_name = re.sub(r'[^\w]+', '_', cleaned_name)
        
        # 3c. Eliminar guiones bajos sobrantes al inicio o final de la cadena
        cleaned_name = cleaned_name.strip('_')
        
        # 3d. Forzar minúsculas
        cleaned_name = cleaned_name.lower()

        # 3e. Anadir prefijo si inicia con un dígito
        if cleaned_name and cleaned_name[0].isdigit():
             cleaned_name = 'f_' + cleaned_name
        
        safe_feature_names.append(cleaned_name)

    # Reconstruir DataFrames con los nombres limpios
    X_train_ohe = pd.DataFrame(X_train_transformed, columns=safe_feature_names, index=X_train.index)
    X_test_ohe = pd.DataFrame(X_test_transformed, columns=safe_feature_names, index=X_test.index)
    
    # 4. Aplicar SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_ohe, y_train)

    return X_train_smote, X_test_ohe, y_train_smote
    
def entrenar_modelo(X_train: pd.DataFrame, y_train: pd.Series, modelo_base):
    """Entrena un modelo base clonado."""
    from sklearn.base import clone
    modelo = clone(modelo_base)
    modelo.fit(X_train, y_train)
    return modelo

def evaluar_modelo_general(modelo, X_test, y_test, umbral=0.5):
    """
    Evalúa un modelo y calcula métricas clave para la clase 1 (Inasistencia).
    """
    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= umbral).astype(int)

    # Métricas
    report = classification_report(y_test, y_pred, output_dict=True)
    
    auc_roc = roc_auc_score(y_test, y_prob)
    
    # Extracción para la clase 1 (Inasistencia)
    metrics = {
        'AUC_ROC': auc_roc,
        'Precision (1)': report['1']['precision'],
        'Recall (1)': report['1']['recall'],
        'F1-Score (1)': report['1']['f1-score'],
    }
    return metrics

def evaluar_segmento(modelo, X_test_ohe, y_test, X_test_original, nombre_columna, umbral=0.36):
    """
    Evalúa el modelo ya entrenado por cada valor único de una columna categórica 
    usando las métricas de clasificación con un umbral predefinido.
    
    NOTA: X_test_ohe se usa para predecir, X_test_original se usa para filtrar.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    resultados_segmento = []
    
    # 1. Obtener valores únicos del segmento desde el DataFrame original
    valores_unicos = X_test_original[nombre_columna].unique()

    for valor in valores_unicos:
        # 2. Filtrar por índice usando el DataFrame original (X_test_original)
        indices_segmento = X_test_original[X_test_original[nombre_columna] == valor].index
        
        # Filtrar los datos OHE y el target usando los índices
        X_seg = X_test_ohe.loc[indices_segmento]
        y_seg = y_test.loc[indices_segmento]
        
        if len(y_seg) == 0:
            continue

        # 3. Generar predicciones de probabilidad y aplicar el umbral
        proba = modelo.predict_proba(X_seg)[:, 1]
        preds = (proba >= umbral).astype(int)
        
        # 4. Calcular métricas
        total_citas = len(y_seg)
        total_ausencias = y_seg.sum()
        total_alertas = preds.sum()
        
        precision = precision_score(y_seg, preds, zero_division=0)
        recall = recall_score(y_seg, preds, zero_division=0)
        f1 = f1_score(y_seg, preds, zero_division=0)

        resultados_segmento.append({
            'Segmento': nombre_columna,
            'Valor': valor,
            'Total_Citas': total_citas,
            'Total_Ausentes_Reales': total_ausencias,
            'Total_Alertas': total_alertas,
            'Precisión (P)': precision,
            'Recall (R)': recall,
            'F1-Score': f1
        })
        
    return resultados_segmento

def generar_curva_roc_auc(y_true, y_prob, label='Modelo'):
    """Calcula y grafica una curva ROC AUC."""
    if len(y_true.unique()) < 2:
        print(f"⚠️ Datos insuficientes para curva ROC.")
        return 0
        
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.plot(fpr, tpr, 
             label=f'{label} (AUC = {roc_auc:.2f})', 
             linewidth=3)
    
    return roc_auc

def crear_grafico_segmento(df_segmentos, segmento_col, umbral_valor, metrica='Recall (R)', color='Reds_d'):
    """
    Genera un gráfico de barras comparativo para un segmento específico.
    Acepta 'umbral_valor' para incluirlo correctamente en el título.
    """
    # 1. Filtrar el DataFrame por la columna del segmento y ordenar
    df_plot = df_segmentos[df_segmentos['Segmento'] == segmento_col].sort_values(metrica, ascending=False)
    
    # 2. Ajustar el tamaño de la figura dinámicamente
    fig_height = max(5, len(df_plot) * 0.4) 
    plt.figure(figsize=(10, fig_height))
    
    # 3. Generar el gráfico de barras comparativo
    ax = sns.barplot(
        x=metrica, 
        y='Valor', 
        data=df_plot, 
        palette=color,
        hue='Valor', 
        legend=False 
    )
    
    # 4. Añadir etiquetas de valor
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.005, # x position
                 p.get_y() + p.get_height() / 2, # y position
                 f'{width:.2f}', 
                 color='black', ha="left", va="center")

    # 5. Título para usar el valor de 'umbral_valor'
    plt.title(f'Rendimiento por {segmento_col.replace("_", " ").title()} - Métrica: {metrica} (Umbral {umbral_valor})', fontsize=14)
    plt.xlabel(metrica)
    plt.ylabel(segmento_col.replace('_', ' ').title())
    plt.xlim(0, df_plot[metrica].max() + 0.1) 
    plt.tight_layout()
    plt.show()
    
def optimizar_modelo_con_grid_search(modelo_base, X_train, y_train, param_grid, cv=5, scoring='roc_auc'):
    """
    Realiza una búsqueda en grilla (Grid Search) para encontrar los mejores 
    hiperparámetros para el modelo y la data.
    """
    print(f"Iniciando Grid Search para {modelo_base.__class__.__name__}...")
    
    # 1. Clonar el modelo base para el proceso
    modelo = clone(modelo_base)
    
    # 2. Configurar Grid Search
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        scoring=scoring,  
        cv=cv,           
        verbose=1,
        n_jobs=-1        
    )
    
    # 3. Ejecutar la búsqueda
    grid_search.fit(X_train, y_train)
    
    # 4. Extraer resultados clave
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    
    print("\n--- Resultados del Grid Search ---")
    print(f"Mejor score ({scoring}): {best_score:.4f}")
    print(f"Mejores hiperparámetros: {best_params}")
    
    return best_model, best_params

def graficar_curvas_umbral(modelo, X_test_ohe, y_test, titulo=""):
    """
    Calcula y grafica las curvas de Precision y Recall vs Umbral.
    Retorna el umbral que maximiza F1.
    """
    from sklearn.metrics import precision_recall_curve
    
    y_pred_proba = modelo.predict_proba(X_test_ohe)[:, 1]
    
    # Calcular Precision, Recall y Umbrales
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calcular F1-score para cada umbral
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    # Encontrar el umbral que maximiza el F1-score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], label='Precisión', color='blue')
    plt.plot(thresholds, recall[:-1], label='Recall', color='red')
    plt.axvline(optimal_threshold, color='green', linestyle='--', 
                label=f'Umbral Máximo F1: {optimal_threshold:.2f}')
    
    plt.title(f'Curvas de Precisión/Recall vs. Umbral ({titulo})', fontsize=14)
    plt.xlabel('Umbral de Clasificación')
    plt.ylabel('Métrica')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"El umbral que maximiza el F1-Score es: {optimal_threshold:.2f}")
    
    return optimal_threshold

def generar_alertas_clinicas(modelo, X_test, y_test, X_base_df, umbral_operacional: float):
    """
    Genera un DataFrame con las alertas de alto riesgo (prob > umbral)
    y añade las variables originales (especialidad, grupo_etario, etc.) para la acción clínica.
    """
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    # Crear un DataFrame de resultados con las variables transformadas
    df_resultados = pd.DataFrame({
        'Prob_Inasistencia': y_prob,
        'Inasistencia_Real': y_test,
        'Riesgo_Alto': (y_prob >= umbral_operacional).astype(int)
    }, index=X_test.index)
    
    df_final = X_base_df.loc[X_test.index].copy()
    df_final = df_final.merge(df_resultados, left_index=True, right_index=True)
    
    # Filtrar solo las citas que el modelo marcó como Alto Riesgo (Alerta)
    df_alertas = df_final[df_final['Riesgo_Alto'] == 1].sort_values('Prob_Inasistencia', ascending=False)
    
    return df_alertas

def analisis_importancia_shap(modelo, X_test):
    """
    Calcula los valores SHAP y grafica la importancia de las variables 
    (Feature Importance) para el modelo LightGBM.
    """
    print("Calculando valores SHAP...")
    
    # Aseguramos que X_test sea un DataFrame de al menos dos dimensiones
    X_test_df = X_test.copy()
    if X_test_df.ndim == 1:
        X_test_df = X_test_df.to_frame().T # Lo convierte a 2D

    # 1. CÁLCULO DE VALORES SHAP
    explainer = shap.TreeExplainer(modelo)
    
    # SHAP para clasificadores binarios devuelve una lista de dos arrays. Clase positiva es [1]
    # Usamos try/except para capturar el caso inestable de la versión de SHAP/LGBM
    try:
        shap_values = explainer.shap_values(X_test_df)
        # Si devuelve lista, tomamos el valor para la clase 1 (inasistencia)
        if isinstance(shap_values, list):
            shap_values_clase1 = shap_values[1]
        else:
            shap_values_clase1 = shap_values
            
    except Exception as e:
        print(f"Error al calcular valores SHAP: {e}. Continuando solo con Feature Importance de LGBM.")
        shap_values_clase1 = None 

    # 2. Gráfico de BARRAS (Importancia Global) - Usa el método nativo de LightGBM (Gain)
    feature_importances = pd.Series(modelo.feature_importances_, index=X_test_df.columns)
    plt.figure(figsize=(10, 6))
    feature_importances.sort_values(ascending=False).head(10).plot(kind='barh')
    plt.title('Top 10 Importancia de Variables (Feature Importances de LGBM - Gain)', fontsize=14)
    plt.xlabel('Importancia (Gain)')
    plt.gca().invert_yaxis()
    plt.show() 
    
    # 3. Retorno del DataFrame SHAP (Solo si se calculó correctamente)
    if shap_values_clase1 is not None and shap_values_clase1.shape[1] == len(X_test_df.columns):
         # Solo si las dimensiones coinciden, devolvemos el DataFrame SHAP
         return pd.DataFrame(shap_values_clase1, columns=X_test_df.columns, index=X_test_df.index)
    else:
         return feature_importances.sort_values(ascending=False) # Retorna la importancia de LGBM si SHAP falla.