import numpy as np
import time
import random
from sklearn.cluster import KMeans

# Генерация случайных городов
def generate_cities(n, x_range=(0, 100), y_range=(0, 100), seed=None):
    """
    Генерирует словарь городов со случайными координатами.
    Вход: n - количество городов, диапазоны координат.
    Выход: словарь { "Городi": (x, y) }
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    cities = {}
    for i in range(1, n+1):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        cities[f"Город{i}"] = (x, y)
    return cities

# Вспомогательная функция для извлечения координат и названий в порядке возрастания
def extract_coords(cities_dict):
    """
    Преобразует словарь городов в массив координат и список названий,
    отсортированных по ключу (чтобы сохранить соответствие индексов).
    """
    items = sorted(cities_dict.items())  # сортировка по названию (Город1, Город2, ...)
    names = [item[0] for item in items]
    coords = np.array([item[1] for item in items])
    return coords, names

# Построение выходного словаря по меткам кластеров и центрам
def build_result(labels, centers, cities_dict, coords, names):
    """
    labels: массив меток кластеров для каждой точки (индекс соответствует names)
    centers: массив координат центров кластеров
    cities_dict: исходный словарь городов (для полноты, не используется напрямую)
    coords: массив координат городов
    names: список названий городов в порядке coords
    Возвращает словарь:
        ключ: название города, ближайшего к центру кластера
        значение: список названий городов в этом кластере
    """
    clusters = {}
    unique_labels = set(labels)
    for lab in unique_labels:
        # индексы городов, принадлежащих кластеру
        idx = np.where(labels == lab)[0]
        cluster_names = [names[i] for i in idx]
        # центроид кластера
        centroid = centers[lab]
        # среди городов кластера ищем ближайший к центроиду
        cluster_coords = coords[idx]
        distances = np.linalg.norm(cluster_coords - centroid, axis=1)
        nearest_idx_local = np.argmin(distances)
        nearest_city_name = cluster_names[nearest_idx_local]
        clusters[nearest_city_name] = cluster_names
    return clusters

# Вариант 1: с использованием sklearn KMeans
def kmeans_sklearn(cities_dict, K, random_state=42):
    """
    Кластеризация через sklearn.cluster.KMeans.
    Возвращает словарь в заданном формате.
    """
    coords, names = extract_coords(cities_dict)
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    kmeans.fit(coords)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return build_result(labels, centers, cities_dict, coords, names)

# Вариант 2: ручная реализация K-means (максимально приближенная к результату)
def kmeans_manual(cities_dict, K, max_iters=300, tol=1e-4, seed=None):
    """
    Кластеризация самостоятельно реализованным алгоритмом K-средних.
    Инициализация: случайный выбор K различных городов.
    Возвращает словарь в заданном формате.
    """
    if seed is not None:
        np.random.seed(seed)
    coords, names = extract_coords(cities_dict)
    n = len(coords)
    
    # Инициализация центроидов: случайные K точек из набора данных
    init_idx = np.random.choice(n, size=K, replace=False)
    centers = coords[init_idx].copy()
    
    for iteration in range(max_iters):
        # Шаг 1: присвоение каждой точки ближайшему центру
        # Вычисляем расстояния от всех точек до всех центров
        # diff: (n, K, 2)
        diff = coords[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))  # (n, K)
        labels = np.argmin(distances, axis=1)
        
        # Шаг 2: пересчёт центров как среднее точек в кластере
        new_centers = np.zeros_like(centers)
        for j in range(K):
            cluster_points = coords[labels == j]
            if len(cluster_points) > 0:
                new_centers[j] = cluster_points.mean(axis=0)
            else:
                # Если кластер пуст, оставляем старый центр (можно переинициализировать, но для простоты так)
                new_centers[j] = centers[j]
        
        # Проверка сходимости
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break
    
    # Финальное присвоение меток (на случай, если центры менялись после последнего пересчёта)
    diff = coords[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    labels = np.argmin(distances, axis=1)
    
    return build_result(labels, centers, cities_dict, coords, names)

# Бенчмарк: автоматический запуск обоих вариантов с разными N и K
def benchmark(N_values, K_values, num_runs=3):
    """
    Запускает оба варианта для каждой комбинации N и K,
    измеряет среднее время выполнения (для усреднения шума).
    Выводит результаты в табличном виде.
    """
    print("Бенчмарк кластеризации городов (время в секундах)")
    print("="*70)
    print(f"{'N':>6} {'K':>4} {'sklearn':>10} {'manual':>10} {'разница':>10}")
    print("-"*70)
    
    for N in N_values:
        for K in K_values:
            if K > N:  # не может быть кластеров больше городов
                continue
            # Генерируем один и тот же набор городов для всех запусков
            base_seed = N * 100 + K  # фиксированный seed для воспроизводимости
            cities = generate_cities(N, seed=base_seed)
            
            # Прогрев (не замеряем)
            _ = kmeans_sklearn(cities, K, random_state=base_seed)
            _ = kmeans_manual(cities, K, seed=base_seed)
            
            # Замер времени sklearn (среднее за num_runs)
            sk_times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = kmeans_sklearn(cities, K, random_state=base_seed+_)
                end = time.perf_counter()
                sk_times.append(end - start)
            sk_avg = np.mean(sk_times)
            
            # Замер времени manual (среднее за num_runs)
            man_times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = kmeans_manual(cities, K, seed=base_seed+_)
                end = time.perf_counter()
                man_times.append(end - start)
            man_avg = np.mean(man_times)
            
            diff = man_avg / sk_avg if sk_avg > 0 else float('inf')
            print(f"{N:6d} {K:4d} {sk_avg:10.5f} {man_avg:10.5f} {diff:10.2f}x")
    
    print("="*70)

# Пример запуска
if __name__ == "__main__":
    # Задаём диапазоны N и K для тестирования
    N_list = [100, 200, 500, 1000]
    K_list = [2, 3, 5, 10]
    benchmark(N_list, K_list, num_runs=3)
    
    # Демонстрация работы на маленьком примере
    print("\nДемонстрация на 10 городах, K=3:")
    cities_demo = generate_cities(10, seed=42)
    print("Исходные города:")
    for name, coord in cities_demo.items():
        print(f"  {name}: {coord}")
    
    result_sk = kmeans_sklearn(cities_demo, K=3, random_state=42)
    print("\nРезультат sklearn:")
    for center, cities in result_sk.items():
        print(f"Центр {center}: {cities}")
    
    result_man = kmeans_manual(cities_demo, K=3, seed=42)
    print("\nРезультат ручной реализации:")
    for center, cities in result_man.items():
        print(f"Центр {center}: {cities}")