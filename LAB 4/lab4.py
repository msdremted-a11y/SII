# -*- coding: utf-8 -*-
"""
Лабораторная работа №4: Исследование инструментов классификации Scikit-learn
Тема: Linear Discriminant Analysis и кластеризация на медицинских данных
Датасет: Pima Indians Diabetes Database (загрузка из публичного источника)
Запуск в VS Code (или обычном Python-окружении)
"""

# 1. Импорт библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Настройка стилей графиков
sns.set_style("whitegrid")


# Данные взяты из публичного репозитория UCI (через GitHub)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# Указываем названия столбцов (из описания датасета)
columns = [
    'Pregnancies',            # Количество беременностей
    'Glucose',                # Уровень глюкозы через 2 часа после нагрузки
    'BloodPressure',          # Диастолическое давление (мм рт.ст.)
    'SkinThickness',          # Толщина кожной складки трицепса (мм)
    'Insulin',                # 2-часовой сывороточный инсулин (мкЕд/мл)
    'BMI',                    # Индекс массы тела (вес кг/(рост м)^2)
    'DiabetesPedigreeFunction', # Функция наследственности диабета
    'Age',                    # Возраст (лет)
    'Outcome'                 # Целевая переменная: 0 = нет диабета, 1 = диабет
]

df = pd.read_csv(url, names=columns)

# Отделим признаки (8 столбцов) и целевую переменную
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Быстрый просмотр данных
print("="*60)
print("ИНФОРМАЦИЯ О ДАННЫХ (Pima Indians Diabetes)")
print("="*60)
print(f"Размер матрицы признаков: {X.shape}")
print("Целевая переменная: 0 - нет диабета, 1 - диабет")
print("\nРаспределение классов:")
print(y.value_counts())

# Проверка типов и пропусков
print("\nТипы данных признаков:")
print(X.dtypes.value_counts())
print(f"\nОбщее количество пропущенных значений: {X.isnull().sum().sum()}")


X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Сохраняем копии сырых данных
X_train_raw_copy = X_train_raw.copy()
X_test_raw_copy  = X_test_raw.copy()

print("\n" + "="*60)
print("КЛАССИФИКАЦИЯ: LDA НА СЫРЫХ ДАННЫХ")
print("="*60)

lda_raw = LinearDiscriminantAnalysis()
lda_raw.fit(X_train_raw_copy, y_train)
y_pred_raw = lda_raw.predict(X_test_raw_copy)

acc_raw = accuracy_score(y_test, y_pred_raw)
print(f"Точность на тестовой выборке (сырые данные): {acc_raw:.4f}")
print("\nОтчёт классификации:")
print(classification_report(y_test, y_pred_raw, target_names=['Нет диабета', 'Диабет']))

# Матрица ошибок
ConfusionMatrixDisplay.from_estimator(
    lda_raw, X_test_raw_copy, y_test,
    display_labels=['Нет диабета', 'Диабет'],
    cmap='Blues'
)
plt.title("Матрица ошибок LDA (сырые данные)")
plt.show()



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw_copy)
X_test_scaled  = scaler.transform(X_test_raw_copy)


print("\n" + "="*60)
print("КЛАССИФИКАЦИЯ: LDA ПОСЛЕ СТАНДАРТИЗАЦИИ")
print("="*60)

lda_scaled = LinearDiscriminantAnalysis()
lda_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = lda_scaled.predict(X_test_scaled)

acc_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"Точность на тестовой выборке (стандартизированные данные): {acc_scaled:.4f}")
print("\nОтчёт классификации:")
print(classification_report(y_test, y_pred_scaled, target_names=['Нет диабета', 'Диабет']))

ConfusionMatrixDisplay.from_estimator(
    lda_scaled, X_test_scaled, y_test,
    display_labels=['Нет диабета', 'Диабет'],
    cmap='Greens'
)
plt.title("Матрица ошибок LDA (стандартизированные данные)")
plt.show()


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*60)
print("КРОСС-ВАЛИДАЦИЯ LDA")
print("="*60)

# Сырые данные
scores_raw_cv = cross_val_score(lda_raw, X, y, cv=cv, scoring='accuracy')
print(f"Сырые данные: средняя точность = {scores_raw_cv.mean():.4f} ± {scores_raw_cv.std():.4f}")

# Стандартизированные данные (масштабирование внутри фолдов)
pipeline = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
scores_scaled_cv = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
print(f"Стандартизированные: средняя точность = {scores_scaled_cv.mean():.4f} ± {scores_scaled_cv.std():.4f}")


results_df = pd.DataFrame({
    'Метод оценки': ['Hold-out (сырые)', 'Hold-out (стандарт.)',
                     '5-Fold CV (сырые)', '5-Fold CV (стандарт.)'],
    'Средняя точность': [
        acc_raw, acc_scaled,
        scores_raw_cv.mean(), scores_scaled_cv.mean()
    ],
    'Стандартное отклонение': [
        0, 0,
        scores_raw_cv.std(), scores_scaled_cv.std()
    ]
})
print("\nТАБЛИЦА СРАВНЕНИЯ РЕЗУЛЬТАТОВ КЛАССИФИКАЦИИ:")
print(results_df.to_string(index=False))


# Boxplot точности
plt.figure(figsize=(8,6))
sns.boxplot(data=[scores_raw_cv, scores_scaled_cv], width=0.5)
plt.xticks([0, 1], ['Сырые данные', 'Стандартизированные'])
plt.ylabel('Accuracy')
plt.title('Распределение точности LDA при кросс-валидации')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Проекция LDA (только для стандартизированных данных, чтобы увидеть разделение)
lda_proj = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda_proj.fit_transform(X_train_scaled, y_train)

plt.figure(figsize=(8,5))
for label, color, name in zip([0,1], ['blue', 'orange'], ['Нет диабета', 'Диабет']):
    plt.hist(X_lda[y_train == label], bins=20, alpha=0.6, color=color, label=name)
plt.xlabel('Значение LDA-компоненты')
plt.ylabel('Количество объектов')
plt.title('Проекция обучающих данных на первую LDA-компоненту')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


print("\n" + "="*60)
print("КЛАСТЕРИЗАЦИЯ: K-MEANS")
print("="*60)

# На сырых данных
kmeans_raw = KMeans(n_clusters=2, random_state=1, n_init=10)
clusters_raw = kmeans_raw.fit_predict(X)

# На стандартизированных данных
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)
kmeans_scaled = KMeans(n_clusters=2, random_state=1, n_init=10)
clusters_scaled = kmeans_scaled.fit_predict(X_scaled_full)

# Внешние и внутренние метрики качества
ari_raw = adjusted_rand_score(y, clusters_raw)
nmi_raw = normalized_mutual_info_score(y, clusters_raw)
sil_raw = silhouette_score(X, clusters_raw)

ari_scaled = adjusted_rand_score(y, clusters_scaled)
nmi_scaled = normalized_mutual_info_score(y, clusters_scaled)
sil_scaled = silhouette_score(X_scaled_full, clusters_scaled)

clust_results = pd.DataFrame({
    'Метрика': ['Adjusted Rand Index', 'Normalized Mutual Information', 'Silhouette Score'],
    'Сырые данные': [f"{ari_raw:.4f}", f"{nmi_raw:.4f}", f"{sil_raw:.4f}"],
    'Стандартизированные': [f"{ari_scaled:.4f}", f"{nmi_scaled:.4f}", f"{sil_scaled:.4f}"]
})
print("\nРезультаты кластеризации:")
print(clust_results.to_string(index=False))

# Визуализация кластеров через PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled_full)

fig, axes = plt.subplots(1, 2, figsize=(14,6))
# Истинные классы
sc1 = axes[0].scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='Set1', alpha=0.7, edgecolor='k')
axes[0].set_title('Истинные классы (PCA)')
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
leg1 = axes[0].legend(*sc1.legend_elements(), title="Класс")
axes[0].add_artist(leg1)

# Кластеры K-Means
sc2 = axes[1].scatter(X_pca[:,0], X_pca[:,1], c=clusters_scaled, cmap='Set1', alpha=0.7, edgecolor='k')
axes[1].set_title('Кластеры K-Means (стандартизированные)')
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
leg2 = axes[1].legend(*sc2.legend_elements(), title="Кластер")
axes[1].add_artist(leg2)

plt.tight_layout()
plt.show()

