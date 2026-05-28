# -*- coding: utf-8 -*-
"""
Лабораторная работа №5
Исследование инструментов регрессии и кластеризации библиотеки Scikit-learn
С замерами времени выполнения обучения и тестирования
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


CSV_FILE_PATH = r"C:\ml_moscow_flats.csv"

column_names = [
    'wallsMaterial', 'floorNumber', 'floorsTotal', 'totalArea',
    'kitchenArea', 'latitude', 'longitude', 'price'
]

data = pd.read_csv(
    CSV_FILE_PATH,
    encoding='utf-8-sig',
    sep=',',
    names=column_names,
    header=0,
    skipinitialspace=True,
    engine='c'
)

print(f"Размер датасета: {data.shape[0]} строк, {data.shape[1]} столбцов")
print("\nПервые 5 строк:")
print(data.head())


print("\n--- Предобработка данных ---")
initial_rows = len(data)
data = data.drop_duplicates()
print(f"Удалено дубликатов: {initial_rows - len(data)}")
data = data.dropna()
print(f"Удалено строк с пропусками: {initial_rows - len(data)}")
print(f"Итоговое количество строк: {len(data)}")


target = 'price'
features = data.columns.drop(target)
categorical_features = data[features].select_dtypes(include=['object', 'category']).columns.tolist()
numeric_features = data[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nЦелевая переменная: {target}")
print(f"Категориальные признаки: {categorical_features}")
print(f"Числовые признаки: {numeric_features}")


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])


X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

print("\n" + "="*60)
print("ОБУЧЕНИЕ НА СЫРЫХ ДАННЫХ (только OneHot-кодирование)")
print("="*60)

raw_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

X_train_raw = raw_preprocessor.fit_transform(X_train)
X_test_raw = raw_preprocessor.transform(X_test)

def evaluate_model_with_time(model, X_train, X_test, y_train, y_test, model_name):
    # Замер времени обучения
    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit
    
    # Замер времени предсказания
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name}:")
    print(f"  Время обучения: {fit_time:.4f} сек")
    print(f"  Время предсказания: {pred_time:.4f} сек")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R2:  {r2:.4f}")
    
    # Кросс-валидация (без замеров времени)
    if model_name == 'MLPRegressor':
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    else:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        print(f"  CV R2 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    except Exception as e:
        print(f"  CV не выполнен: {e}")
    return y_pred, fit_time, pred_time

models = {
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000),
    'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                                 max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
}

print("\n--- Результаты на сырых данных ---")
times_raw = {}
for name, model in models.items():
    _, fit_t, pred_t = evaluate_model_with_time(model, X_train_raw, X_test_raw, y_train, y_test, name)
    times_raw[name] = {'fit': fit_t, 'predict': pred_t}


print("\n" + "="*60)
print("ОБУЧЕНИЕ НА ОЧИЩЕННЫХ ДАННЫХ (масштабирование + OneHot)")
print("="*60)

X_train_clean = preprocessor.fit_transform(X_train)
X_test_clean = preprocessor.transform(X_test)

print("\n--- Результаты на очищенных данных ---")
times_clean = {}
for name, model in models.items():
    if name == 'ElasticNet':
        model_clean = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000)
    else:
        model_clean = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                                   max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
    _, fit_t, pred_t = evaluate_model_with_time(model_clean, X_train_clean, X_test_clean, y_train, y_test, name)
    times_clean[name] = {'fit': fit_t, 'predict': pred_t}


print("\n" + "="*60)
print("ИСПОЛЬЗОВАНИЕ РАЗНЫХ РАЗБИЕНИЙ (с замерами времени)")
print("="*60)
for rs in [0, 123]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=rs)
    X_tr_clean = preprocessor.fit_transform(X_tr)
    X_te_clean = preprocessor.transform(X_te)
    
    for name in models.keys():
        if name == 'ElasticNet':
            model_tmp = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000)
        else:
            model_tmp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                                     max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
        start = time.time()
        model_tmp.fit(X_tr_clean, y_tr)
        fit_time = time.time() - start
        start_pred = time.time()
        y_pr = model_tmp.predict(X_te_clean)
        pred_time = time.time() - start_pred
        r2 = r2_score(y_te, y_pr)
        print(f"{name} (random_state={rs}): R2={r2:.4f}, fit={fit_time:.4f}s, pred={pred_time:.4f}s")


print("\n" + "="*60)
print("СВОДНАЯ ТАБЛИЦА (очищенные данные, random_state=42)")
print("="*60)
results = []
for name in models.keys():
    if name == 'ElasticNet':
        model_final = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000)
    else:
        model_final = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                                   max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
    start_fit = time.time()
    model_final.fit(X_train_clean, y_train)
    fit_time = time.time() - start_fit
    start_pred = time.time()
    y_pred_final = model_final.predict(X_test_clean)
    pred_time = time.time() - start_pred
    results.append({
        'Модель': name,
        'MSE': mean_squared_error(y_test, y_pred_final),
        'MAE': mean_absolute_error(y_test, y_pred_final),
        'R2': r2_score(y_test, y_pred_final),
        'Время обучения (сек)': fit_time,
        'Время предсказания (сек)': pred_time
    })
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))



fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name in zip(axes, models.keys()):
    # Переобучаем модели для получения предсказаний для графика (можно взять уже готовые)
    if name == 'ElasticNet':
        m = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000)
    else:
        m = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                         max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
    m.fit(X_train_clean, y_train)
    y_pred = m.predict(X_test_clean)
    ax.scatter(y_test, y_pred, alpha=0.5, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Реальная цена')
    ax.set_ylabel('Предсказанная цена')
    ax.set_title(f'{name}: предсказания vs реальность')
    ax.grid(True)
plt.tight_layout()
plt.show()


print("\n" + "="*60)
print("ЗАДАЧА КЛАСТЕРИЗАЦИИ (KMeans)")
print("="*60)

X_cluster = preprocessor.fit_transform(X)
sample_size = min(10000, X_cluster.shape[0])
np.random.seed(42)
idx = np.random.choice(X_cluster.shape[0], sample_size, replace=False)
X_sample = X_cluster[idx]

inertias = []
K_range = range(1, 9)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_sample)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Число кластеров k')
plt.ylabel('Инерция')
plt.title('Метод локтя для выбора k')
plt.grid(True)
plt.show()

k_opt = 3
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster)
data['Cluster'] = clusters

print(f"\nРаспределение объектов по кластерам (k={k_opt}):")
print(data['Cluster'].value_counts().sort_index())

# Визуализация кластеров через PCA
pca = PCA(n_components=2, random_state=42)
X_pca_sample = pca.fit_transform(X_sample)
clusters_sample = clusters[idx]
plt.figure(figsize=(10, 6))
plt.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=clusters_sample, cmap='viridis', alpha=0.5, s=5)
plt.colorbar(label='Кластер')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title(f'Визуализация кластеров (k={k_opt})')
plt.grid(True)
plt.show()

# Анализ кластеров
cluster_analysis = data.groupby('Cluster')[numeric_features + [target]].mean()
print("\nСредние значения числовых признаков и цены по кластерам:")
print(cluster_analysis.round(2))

if categorical_features:
    mode_cat = data.groupby('Cluster')[categorical_features[0]].agg(lambda x: x.mode()[0] if len(x.mode())>0 else 'unknown')
    print(f"\nНаиболее частое значение категориального признака '{categorical_features[0]}' в кластере:")
    print(mode_cat)

