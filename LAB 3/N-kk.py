import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time  # для замера времени
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter


class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)


def multiclass_tp_fp_fn_tn(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    data = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        data.append([label, tp, fp, fn, tn])
    return pd.DataFrame(data, columns=['Класс', 'TP', 'FP', 'FN', 'TN'])


data = {
    'продукт': ['Яблоко', 'салат', 'бекон', 'банан', 'орехи', 'рыба', 'сыр', 'виноград', 'морковь', 'апельсин'],
    'сладость': [7, 2, 1, 9, 1, 1, 1, 8, 2, 6],
    'хруст':    [7, 5, 2, 1, 5, 1, 1, 1, 8, 1],
    'класс':    ['Фрукт', 'Овощ', 'Протеин', 'Фрукт', 'Протеин', 'Протеин', 'Протеин', 'Фрукт', 'Овощ', 'Фрукт']
}
df = pd.DataFrame(data)
print("Исходный датасет:")
print(df)


def augment_dataset(df, n_augment=20, noise_std=0.5, random_seed=42):
    np.random.seed(random_seed)
    augmented_rows = []
    for _, row in df.iterrows():
        for _ in range(n_augment):
            new_sweet = row['сладость'] + np.random.normal(0, noise_std)
            new_crunch = row['хруст'] + np.random.normal(0, noise_std)
            new_sweet = np.clip(new_sweet, 0, 10)
            new_crunch = np.clip(new_crunch, 0, 10)
            augmented_rows.append([new_sweet, new_crunch, row['класс']])
    aug_df = pd.DataFrame(augmented_rows, columns=['сладость', 'хруст', 'класс'])
    full_df = pd.concat([df[['сладость', 'хруст', 'класс']], aug_df], ignore_index=True)
    return full_df

# Расширяем датасет (3 класса)
full_df = augment_dataset(df, n_augment=30, noise_std=0.7)
print(f"\nРазмер расширенного датасета: {full_df.shape[0]} примеров")

# Визуализация расширенного датасета
plt.figure(figsize=(8, 6))
classes = full_df['класс'].unique()
colors = {'Фрукт': 'red', 'Овощ': 'green', 'Протеин': 'blue'}
for cls in classes:
    subset = full_df[full_df['класс'] == cls]
    plt.scatter(subset['сладость'], subset['хруст'], c=colors[cls], label=cls, alpha=0.6, edgecolors='k')
plt.xlabel('Сладость')
plt.ylabel('Хруст')
plt.title('Расширенный датасет (3 класса)')
plt.legend()
plt.grid(True)
plt.show()


X = full_df[['сладость', 'хруст']].values
y = full_df['класс'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


custom_knn = CustomKNN(k=5)

# Замер времени обучения
start_fit = time.time()
custom_knn.fit(X_train, y_train)
time_fit_custom = time.time() - start_fit

# Замер времени предсказания
start_pred = time.time()
y_pred_custom = custom_knn.predict(X_test)
time_pred_custom = time.time() - start_pred

total_time_custom = time_fit_custom + time_pred_custom
acc_custom = accuracy_score(y_test, y_pred_custom)

print(f"\n=== Собственный k-NN (k=5) ===")
print(f"Точность: {acc_custom:.4f}")
print(f"Время обучения: {time_fit_custom:.6f} сек")
print(f"Время предсказания: {time_pred_custom:.6f} сек")
print(f"Общее время: {total_time_custom:.6f} сек")
print(f"Отношение точность/общее_время: {acc_custom/total_time_custom:.2f} (1/с)")
print("Отчёт классификации:")
print(classification_report(y_test, y_pred_custom))
print("Матрица TP/FP/FN/TN:")
print(multiclass_tp_fp_fn_tn(y_test, y_pred_custom, labels=np.unique(y_test)))


sklearn_knn = KNeighborsClassifier(n_neighbors=5)

start_fit = time.time()
sklearn_knn.fit(X_train, y_train)
time_fit_sklearn = time.time() - start_fit

start_pred = time.time()
y_pred_sklearn = sklearn_knn.predict(X_test)
time_pred_sklearn = time.time() - start_pred

total_time_sklearn = time_fit_sklearn + time_pred_sklearn
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"\n=== sklearn k-NN (k=5) ===")
print(f"Точность: {acc_sklearn:.4f}")
print(f"Время обучения: {time_fit_sklearn:.6f} сек")
print(f"Время предсказания: {time_pred_sklearn:.6f} сек")
print(f"Общее время: {total_time_sklearn:.6f} сек")
print(f"Отношение точность/общее_время: {acc_sklearn/total_time_sklearn:.2f} (1/с)")
print("Отчёт классификации:")
print(classification_report(y_test, y_pred_sklearn))
print("Матрица TP/FP/FN/TN:")
print(multiclass_tp_fp_fn_tn(y_test, y_pred_sklearn, labels=np.unique(y_test)))


def plot_decision_boundary(X, y, model, title, class_colors=None):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    classes = np.unique(y)
    class_to_num = {cls: i for i, cls in enumerate(classes)}
    Z_num = np.array([class_to_num[z] for z in Z]).reshape(xx.shape)
    plt.contourf(xx, yy, Z_num, alpha=0.3, levels=len(classes)-1, cmap=plt.cm.RdYlBu)
    
    if class_colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors_cycle = prop_cycle.by_key()['color']
        class_colors = {cls: colors_cycle[i % len(colors_cycle)] for i, cls in enumerate(classes)}
    
    for cls in classes:
        idx = y == cls
        plt.scatter(X[idx, 0], X[idx, 1], c=class_colors[cls], label=cls, edgecolors='k')
    plt.xlabel('Сладость')
    plt.ylabel('Хруст')
    plt.title(title)
    plt.legend()
    plt.grid(True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_decision_boundary(X_train, y_train, sklearn_knn, 'sklearn k-NN (3 класса)', class_colors=colors)
plt.subplot(1, 2, 2)
plot_decision_boundary(X_train, y_train, custom_knn, 'Собственный k-NN (3 класса)', class_colors=colors)
plt.tight_layout()
plt.show()


new_class_data = {
    'продукт': ['Хлеб', 'Круассан', 'Рис', 'Овсянка'],
    'сладость': [2, 5, 1, 3],
    'хруст':    [4, 8, 2, 6],
    'класс':    ['Зерновые'] * 4
}
new_df = pd.DataFrame(new_class_data)
aug_new_df = augment_dataset(new_df, n_augment=30, noise_std=0.7)
extended_df = pd.concat([full_df, aug_new_df], ignore_index=True)
print(f"\nРазмер датасета после добавления класса 'Зерновые': {extended_df.shape[0]} примеров")

# Визуализация 4 классов
plt.figure(figsize=(8, 6))
classes_ext = extended_df['класс'].unique()
colors_ext = {'Фрукт': 'red', 'Овощ': 'green', 'Протеин': 'blue', 'Зерновые': 'orange'}
for cls in classes_ext:
    subset = extended_df[extended_df['класс'] == cls]
    plt.scatter(subset['сладость'], subset['хруст'], c=colors_ext[cls], label=cls, alpha=0.6, edgecolors='k')
plt.xlabel('Сладость')
plt.ylabel('Хруст')
plt.title('Расширенный датасет (4 класса)')
plt.legend()
plt.grid(True)
plt.show()

# Подготовка данных (4 класса)
X_ext = extended_df[['сладость', 'хруст']].values
y_ext = extended_df['класс'].values

X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
    X_ext, y_ext, test_size=0.25, random_state=42, stratify=y_ext)


custom_knn_ext = CustomKNN(k=5)

start_fit = time.time()
custom_knn_ext.fit(X_train_ext, y_train_ext)
time_fit_custom_ext = time.time() - start_fit

start_pred = time.time()
y_pred_custom_ext = custom_knn_ext.predict(X_test_ext)
time_pred_custom_ext = time.time() - start_pred

total_time_custom_ext = time_fit_custom_ext + time_pred_custom_ext
acc_custom_ext = accuracy_score(y_test_ext, y_pred_custom_ext)

print(f"\n=== Собственный k-NN с 4 классами (k=5) ===")
print(f"Точность: {acc_custom_ext:.4f}")
print(f"Время обучения: {time_fit_custom_ext:.6f} сек")
print(f"Время предсказания: {time_pred_custom_ext:.6f} сек")
print(f"Общее время: {total_time_custom_ext:.6f} сек")
print(f"Отношение точность/общее_время: {acc_custom_ext/total_time_custom_ext:.2f} (1/с)")
print("Отчёт классификации:")
print(classification_report(y_test_ext, y_pred_custom_ext))
print("Матрица TP/FP/FN/TN:")
print(multiclass_tp_fp_fn_tn(y_test_ext, y_pred_custom_ext, labels=np.unique(y_test_ext)))


sklearn_knn_ext = KNeighborsClassifier(n_neighbors=5)

start_fit = time.time()
sklearn_knn_ext.fit(X_train_ext, y_train_ext)
time_fit_sklearn_ext = time.time() - start_fit

start_pred = time.time()
y_pred_sklearn_ext = sklearn_knn_ext.predict(X_test_ext)
time_pred_sklearn_ext = time.time() - start_pred

total_time_sklearn_ext = time_fit_sklearn_ext + time_pred_sklearn_ext
acc_sklearn_ext = accuracy_score(y_test_ext, y_pred_sklearn_ext)

print(f"\n=== sklearn k-NN с 4 классами (k=5) ===")
print(f"Точность: {acc_sklearn_ext:.4f}")
print(f"Время обучения: {time_fit_sklearn_ext:.6f} сек")
print(f"Время предсказания: {time_pred_sklearn_ext:.6f} сек")
print(f"Общее время: {total_time_sklearn_ext:.6f} сек")
print(f"Отношение точность/общее_время: {acc_sklearn_ext/total_time_sklearn_ext:.2f} (1/с)")
print("Отчёт классификации:")
print(classification_report(y_test_ext, y_pred_sklearn_ext))
print("Матрица TP/FP/FN/TN:")
print(multiclass_tp_fp_fn_tn(y_test_ext, y_pred_sklearn_ext, labels=np.unique(y_test_ext)))


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_decision_boundary(X_train_ext, y_train_ext, sklearn_knn_ext, 'sklearn k-NN (4 класса)', class_colors=colors_ext)
plt.subplot(1, 2, 2)
plot_decision_boundary(X_train_ext, y_train_ext, custom_knn_ext, 'Собственный k-NN (4 класса)', class_colors=colors_ext)
plt.tight_layout()
plt.show()