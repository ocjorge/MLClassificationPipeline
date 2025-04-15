import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import seaborn as sns

# Cargar datos
df = pd.read_csv('cinco.csv', encoding='cp1252')

# Preparación de los datos
X = df.drop(['Species'], axis=1)  # Todas las columnas excepto 'Species'
y = df['Species']

# Guardar nombres de características para usar más tarde
feature_names = X.columns.tolist()

# Codificar la variable objetivo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Ver la distribución de clases
print("Distribución de clases original:")
for i, label in enumerate(label_encoder.classes_):
    count = np.sum(y_encoded == i)
    print(f"{label}: {count} ({count/len(y_encoded)*100:.2f}%)")

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Aplicar SMOTE para balancear las clases con k_neighbors ajustado
# Dado que tenemos clases muy minoritarias con pocas muestras (SEVERA tiene solo 2),
# necesitamos establecer k_neighbors=1 o usar otra estrategia
try:
    # Intenta con k_neighbors=1 (el mínimo posible)
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
except ValueError as e:
    print(f"Error en SMOTE: {e}")
    print("Usando los datos originales sin oversampling...")
    X_train_resampled, y_train_resampled = X_train, y_train

# Ver la distribución después de SMOTE/oversampling
unique, counts = np.unique(y_train_resampled, return_counts=True)
print("\nDistribución después del procesamiento:")
for i, count in zip(unique, counts):
    print(f"{label_encoder.classes_[i]}: {count}")
    
# Verificar si hay suficientes muestras para cada clase para el análisis
min_samples = min(counts)
if min_samples < 5:
    print(f"\n⚠️ Advertencia: Hay clases con pocas muestras ({min_samples}). Algunos análisis pueden ser menos confiables.")

# Calcular pesos de clase (inverso de la frecuencia)
class_weights = {}
for i, label in enumerate(label_encoder.classes_):
    class_weights[i] = len(y_encoded) / (len(label_encoder.classes_) * np.sum(y_encoded == i))

print("\nPesos de clase:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label}: {class_weights[i]:.2f}")

# =============================================================================
# 1. Análisis PCA para entender la estructura de los datos
# =============================================================================
print("\n=== Análisis PCA ===")
pca = PCA()
pca.fit(X_scaled)

# Varianza explicada
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Graficar varianza explicada
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Varianza Individual')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Varianza Acumulada')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Varianza')
plt.xlabel('Componentes Principales')
plt.ylabel('Ratio de Varianza Explicada')
plt.title('Varianza Explicada por Componentes Principales')
plt.legend()
plt.tight_layout()
plt.savefig('pca_explained_variance.png')
plt.show()

# Determinar número de componentes para explicar 95% de varianza
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Número de componentes para explicar 95% de varianza: {n_components}")

# Proyectar en 2D para visualización
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

# Graficar proyección PCA
plt.figure(figsize=(12, 10))
for i, label in enumerate(label_encoder.classes_):
    # Verificar si hay puntos para esta clase
    points = X_pca[y_encoded == i]
    if len(points) > 0:
        plt.scatter(points[:, 0], points[:, 1], label=f"{label} ({len(points)})", alpha=0.7)
    else:
        print(f"Advertencia: No hay puntos para la clase {label} en la visualización PCA")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Proyección PCA de los datos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_projection.png')
plt.show()

# Analizar contribuciones de características a los primeros componentes
loadings = pca.components_
component_names = [f'PC{i+1}' for i in range(loadings.shape[0])]
loadings_df = pd.DataFrame(loadings.T, columns=component_names, index=feature_names)

# Visualizar loadings para los primeros 2 componentes
plt.figure(figsize=(12, 8))
loadings_heatmap = sns.heatmap(loadings_df.iloc[:, :min(5, len(loadings_df.columns))], 
                               cmap='coolwarm', annot=True, fmt=".2f", 
                               linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Contribución de características a los principales componentes')
plt.tight_layout()
plt.savefig('pca_loadings.png')
plt.show()

# =============================================================================
# 2. Entrenar modelo Random Forest para análisis de importancia de características
# =============================================================================
print("\n=== Entrenando Random Forest para análisis de características ===")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight=class_weights,
    random_state=42
)

# Validación cruzada estratificada
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_scores = []

# Crear arrays para almacenar métricas por fold
fold_accuracy = []
fold_precision_macro = []
fold_recall_macro = []
fold_f1_macro = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_resampled, y_train_resampled)):
    print(f"\nEntrenando fold {fold+1}/{n_splits}")
    
    X_fold_train, X_fold_val = X_train_resampled[train_idx], X_train_resampled[val_idx]
    y_fold_train, y_fold_val = y_train_resampled[train_idx], y_train_resampled[val_idx]
    
    rf_model.fit(X_fold_train, y_fold_train)
    
    # Evaluar en el conjunto de validación
    y_fold_pred = rf_model.predict(X_fold_val)
    
    # Calcular métricas
    acc = accuracy_score(y_fold_val, y_fold_pred)
    prec = precision_score(y_fold_val, y_fold_pred, average='macro', zero_division=0)
    rec = recall_score(y_fold_val, y_fold_pred, average='macro', zero_division=0)
    f1 = f1_score(y_fold_val, y_fold_pred, average='macro', zero_division=0)
    
    # Almacenar métricas
    fold_accuracy.append(acc)
    fold_precision_macro.append(prec)
    fold_recall_macro.append(rec)
    fold_f1_macro.append(f1)
    
    print(f"Fold {fold+1} - Validación:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision macro: {prec:.4f}")
    print(f"  Recall macro: {rec:.4f}")
    print(f"  F1 macro: {f1:.4f}")

# Mostrar métricas promedio de CV
print("\n=== Métricas de validación cruzada (promedio) ===")
print(f"Accuracy: {np.mean(fold_accuracy):.4f} ± {np.std(fold_accuracy):.4f}")
print(f"Precision macro: {np.mean(fold_precision_macro):.4f} ± {np.std(fold_precision_macro):.4f}")
print(f"Recall macro: {np.mean(fold_recall_macro):.4f} ± {np.std(fold_recall_macro):.4f}")
print(f"F1 macro: {np.mean(fold_f1_macro):.4f} ± {np.std(fold_f1_macro):.4f}")

# Entrenar modelo final con todos los datos de entrenamiento
final_rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight=class_weights,
    random_state=42
)

final_rf_model.fit(X_train_resampled, y_train_resampled)

# Evaluar en conjunto de prueba
y_pred = final_rf_model.predict(X_test)
y_pred_proba = final_rf_model.predict_proba(X_test)

# =============================================================================
# 3. Métricas de clasificación detalladas
# =============================================================================
print("\n=== EVALUACIÓN DEL MODELO EN CONJUNTO DE PRUEBA ===")

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Reporte de clasificación detallado
print("\nReporte de clasificación:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

# Calcular métricas por clase
precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

# Crear DataFrame para visualizar métricas por clase
metrics_df = pd.DataFrame({
    'Clase': label_encoder.classes_,
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class
})

print("\nMétricas por clase:")
print(metrics_df)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Matriz de confusión normalizada (por fila)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión (Normalizada)')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig('confusion_matrix_normalized.png')
plt.show()

# AUC-ROC para clasificación multiclase (one-vs-rest)
n_classes = len(label_encoder.classes_)

# Preparar gráfico ROC por clase
plt.figure(figsize=(12, 8))

# Para calcular ROC AUC promedio
roc_auc = dict()
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for i in range(n_classes):
    # Para cada clase, calcular ROC
    fpr, tpr, _ = roc_curve(
        (y_test == i).astype(int), 
        y_pred_proba[:, i]
    )
    roc_auc[i] = auc(fpr, tpr)
    
    # Graficar cada curva ROC
    plt.plot(
        fpr, tpr, lw=2,
        label=f'ROC clase {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})'
    )
    
    # Interpolar para calcular promedio después
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc[i])

# Calcular y graficar ROC promedio
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(
    mean_fpr, mean_tpr, color='b', lw=2,
    label=f'ROC promedio (AUC = {mean_auc:.2f} ± {std_auc:.2f})'
)

# Línea diagonal
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC para cada clase (one-vs-rest)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.show()

# Precision-Recall curve para cada clase (especialmente útil con clases desbalanceadas)
plt.figure(figsize=(12, 8))

# Para calcular PR AUC promedio
pr_auc = dict()
precisions = []
aucs_pr = []
mean_recall = np.linspace(0, 1, 100)

for i in range(n_classes):
    # Para cada clase, calcular PR curve
    precision, recall, _ = precision_recall_curve(
        (y_test == i).astype(int), 
        y_pred_proba[:, i]
    )
    pr_auc[i] = average_precision_score(
        (y_test == i).astype(int), 
        y_pred_proba[:, i]
    )
    
    # Graficar cada curva PR
    plt.plot(
        recall, precision, lw=2,
        label=f'PR clase {label_encoder.classes_[i]} (AUC = {pr_auc[i]:.2f})'
    )
    
    # Interpolar para calcular promedio después
    interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
    precisions.append(interp_precision)
    aucs_pr.append(pr_auc[i])

# Calcular y graficar PR promedio
mean_precision = np.mean(precisions, axis=0)
mean_pr_auc = np.mean(aucs_pr)
std_pr_auc = np.std(aucs_pr)
plt.plot(
    mean_recall, mean_precision, color='b', lw=2,
    label=f'PR promedio (AUC = {mean_pr_auc:.2f} ± {std_pr_auc:.2f})'
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curvas Precision-Recall para cada clase')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curves.png')
plt.show()

# =============================================================================
# 4. Análisis de importancia de características
# =============================================================================
print("\n=== Análisis de importancia de características ===")

# Método 1: Importancia basada en impureza (Gini/Entropy)
feature_importances = final_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nImportancia de características (basada en impureza):")
print(feature_importance_df)

# Visualizar importancia de características
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Importancia de características (Gini/Entropy)')
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.tight_layout()
plt.savefig('feature_importance_gini.png')
plt.show()

# Método 2: Importancia basada en permutación
print("\nCalculando importancia de características basada en permutación (puede tardar)...")
perm_importance = permutation_importance(
    final_rf_model, X_test, y_test, 
    n_repeats=10, 
    random_state=42
)

perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values(by='Importance', ascending=False)

print("\nImportancia de características (basada en permutación):")
print(perm_importance_df[['Feature', 'Importance']])

# Visualizar importancia de características por permutación
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=perm_importance_df)
plt.title('Importancia de características (Permutación)')
plt.xlabel('Reducción en precisión')
plt.ylabel('Característica')
plt.tight_layout()
plt.savefig('feature_importance_permutation.png')
plt.show()

# =============================================================================
# 5. Análisis de correlación de características
# =============================================================================
print("\n=== Análisis de correlación de características ===")
correlation_matrix = pd.DataFrame(data=X, columns=feature_names).corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de correlación de características')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

# =============================================================================
# 6. Guardar resultados y métricas en archivos
# =============================================================================
# Guardar el modelo
import pickle
with open('modelo_random_forest.pkl', 'wb') as f:
    pickle.dump(final_rf_model, f)
print("\nModelo guardado como 'modelo_random_forest.pkl'")

# Guardar el escalador y codificador para uso futuro
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Scaler y Label Encoder guardados para uso futuro")

# Guardar resultados de importancia de características en CSV
feature_importance_df.to_csv('feature_importance_gini.csv', index=False)
perm_importance_df.to_csv('feature_importance_permutation.csv', index=False)

# Guardar métricas en un archivo CSV
metrics_summary = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision Macro', 'Recall Macro', 'F1 Macro', 
                'AUC-ROC Promedio', 'PR-AUC Promedio'],
    'Valor': [acc, 
              precision_score(y_test, y_pred, average='macro', zero_division=0),
              recall_score(y_test, y_pred, average='macro', zero_division=0),
              f1_score(y_test, y_pred, average='macro', zero_division=0),
              mean_auc,
              mean_pr_auc]
})
metrics_summary.to_csv('metricas_resumen.csv', index=False)

# Guardar métricas por clase
metrics_df.to_csv('metricas_por_clase.csv', index=False)

print("Resultados y métricas guardadas en archivos CSV")
