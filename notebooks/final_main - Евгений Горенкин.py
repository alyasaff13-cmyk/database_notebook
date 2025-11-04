import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Вспомогательные функции

# Реализуем расчёт расстояния с помощью функции гаверсинуса
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Функция для определения нахождения точки в некотором радиусе. Понадобится для определения нахождения квартиры в Питере или Москве
def within_radius(center_lat, center_lon, lat_series, lon_series, radius_km=20.0):
    lat = np.array(lat_series, dtype=float)
    lon = np.array(lon_series, dtype=float)
    mask_valid = ~np.isnan(lat) & ~np.isnan(lon)
    res = np.zeros(len(lat), dtype=bool)
    if mask_valid.any():
        res[mask_valid] = haversine_distance(center_lat, center_lon, lat[mask_valid], lon[mask_valid]) <= radius_km
    return res


# Метрики
#  Подсчёт метрик
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, rmse, r2

# Вывод метрик
def print_metrics(model_name, mse, mae, rmse, r2):
    print(f"{model_name}:")
    print(f"  MSE:  {mse:>10.2f}")
    print(f"  MAE:  {mae:>10.2f}")
    print(f"  RMSE: {rmse:>10.2f}")
    print(f"  R²:   {r2:>10.4f}")
    print()

class KNNRegressor:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y, dtype=float)
        return self

    def predict(self, X, batch_size=200):
        # Батчевая векторная реализация: обрабатываем X по кускам, чтобы не упереться в память. Проблема в том, что компьютер не самый новый, поэтому обработать может не так много. Поэтому ограничим до ~ 100к.
        if self.X_train is None or self.y_train is None:
            raise ValueError("Модель не отфильтрована.")
        X = np.array(X, dtype=float)
        n = X.shape[0]
        preds = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = X[start:end] 
            # вычисление может быть тяжёлым, но при разумном batch_size подходит
            dists = np.sqrt(np.sum((Xb[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2))
            # argpartition быстро найдёт k ближайших индексов
            idx = np.argpartition(dists, self.n_neighbors, axis=1)[:, :self.n_neighbors] # По сути частичная сортировка
            preds_batch = np.mean(self.y_train[idx], axis=1)
            preds.append(preds_batch)
            print(f"Обработано {end}/{n} тестовых строк")
        return np.concatenate(preds, axis=0)

# Реализуем класс нашей линейной регрессии
class LinearRegression:
    def __init__(self, learning_rate=0.01, optimization='SGD', epsilon=1e-8, decay_rate=0.9, max_iter=1000):
        self.learning_rate = learning_rate
        self.optimization = optimization
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    # Основная функция, которая занимается обучением нашей модели. 
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        # Переменные для Momentum и AdaGrad
        v_w = np.zeros_like(self.weights)
        v_b = 0.0
        G_w = np.zeros_like(self.weights)
        G_b = 0.0

        # Итеративно обновляем веса, чтобы минимизировать ошибку
        for i in range(self.max_iter):
            # Предсказание
            y_pred = X.dot(self.weights) + self.bias

            # Градиенты
            error = y_pred - y
            dw = (2 / n_samples) * X.T.dot(error)
            db = (2 / n_samples) * np.sum(error)

            # Выбор оптимизатора
            # Обычный градиентный спуск
            if self.optimization == 'SGD':
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Градиентный спуск с инерцией
            elif self.optimization == 'Momentum':
                v_w = self.decay_rate * v_w + self.learning_rate * dw
                v_b = self.decay_rate * v_b + self.learning_rate * db
                self.weights -= v_w
                self.bias -= v_b

            # Шаг должен адаптироваться каждый раз, становясь всё меньше и меньше
            elif self.optimization == 'AdaGrad':
                G_w += dw ** 2
                G_b += db ** 2
                self.weights -= (self.learning_rate / (np.sqrt(G_w) + self.epsilon)) * dw
                self.bias -= (self.learning_rate / (np.sqrt(G_b) + self.epsilon)) * db

            # else:
            #     raise ValueError(f"Unknown optimization method: {self.optimization}")

            # Контроль сходимости 
            # if i % (self.max_iter // 10) == 0 or i == self.max_iter - 1:
            #     mse = np.mean(error ** 2)
            #     print(f"Iter {i:4d} | MSE: {mse:.6f}")

        return self

    def predict(self, X): # Предсказание
        X = np.array(X, dtype=float)
        return X.dot(self.weights) + self.bias


# Загрузка и предобработка
print("Читаем датасет...")
df = pd.read_csv('input_data.csv', delimiter=';')  

# 1) Добавляем признаки is_Moscow и is_Saint_Peterburg (20 км)
print("Добавляем is_Moscow и is_Saint_Peterburg ...")
if 'geo_lat' not in df.columns: 
    df['geo_lat'] = np.nan
if 'geo_lon' not in df.columns: 
    df['geo_lon'] = np.nan

# Проверяем на нахождение в радиусе 20 км от Питера и Москвы.
df['is_Moscow'] = within_radius(55.75583, 37.61778, df['geo_lat'], df['geo_lon'], radius_km=20.0) 
df['is_Saint_Peterburg'] = within_radius(59.93863, 30.31413, df['geo_lat'], df['geo_lon'], radius_km=20.0)

# 2) Относительный этаж (relative_location)
# Если levels == 0 или NaN -> relative = 0. Вообще, по логике этажи не могут быть нулевыми, но на всякий случай проверим.
if 'level' not in df.columns: 
    df['level'] = np.nan
if 'levels' not in df.columns: 
    df['levels'] = np.nan
df['relative_location'] = np.where((df['levels'] > 0) & (~df['levels'].isna()), df['level'] / df['levels'], 0.0)

# 3) Дни от первого наблюдения
# преобразуем дату в datetime, считаем дни от минимальной даты в датасете. Честно признаться, особого смысла в этом нет, но задание есть задание. Другая интерпретация - количество дней с первого наблюдения за каждым конкретным объектом, 
# но для этого во-первых понадобилось бы наличие таких наблюдений, во-вторыых признак, по которому каждую квартиру можно было бы уникально идентифицировать, но такого признака нет.
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    min_date = df['date'].min()
    df['days_from_first'] = (df['date'] - min_date).dt.days.fillna(0).astype(int)
else:
    df['days_from_first'] = 0

# Удалим date после добавления новых признаков
if 'date' in df.columns:
    df = df.drop(columns=['date'])

# 4) Удаляем ненужные колонки
to_drop = ['geo_lat', 'geo_lon', 'object_type', 'postal_code', 'street_id', 'id_region', 'house_id']
for col in to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# 5) Приведение rooms: в задании -1 означает апартаменты -> можно заменить на отдельный флаг. Делаем так, чтобы значения -1 не портили нашу модель
if 'rooms' in df.columns:
    df['is_apartment'] = (df['rooms'] == -1)
    # при необходимости: заменить -1 на 0 или NaN
    df['rooms'] = df['rooms'].replace(-1, 0)

# Предварительная фильтрация / подвыборка
# Для функционирования используем подвыборку для KNN и тестирования, но оставляем опцию обучать библиотечные на всём
MAX_SAMPLE_FOR_OUR_KNN = 100_000  
if len(df) > MAX_SAMPLE_FOR_OUR_KNN:
    print(f"\nДатасет большеват ({len(df):,}). Для моего KNN использую первые {MAX_SAMPLE_FOR_OUR_KNN:,} строк.")
    df_small = df.head(MAX_SAMPLE_FOR_OUR_KNN).copy()
else:
    df_small = df.copy()

# Готовим X, y

# Определяем числовные и категориальные признаки
candidate_numeric = ['level', 'levels', 'rooms', 'area', 'kitchen_area', 'relative_location', 'days_from_first']
candidate_categorical = ['building_type', 'is_Moscow', 'is_Saint_Peterburg', 'is_apartment']

numeric_features = ['level', 'levels', 'rooms', 'area', 'kitchen_area', 'relative_location', 'days_from_first']
categorical_features = ['building_type', 'is_Moscow', 'is_Saint_Peterburg', 'is_apartment']

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Удаляем строки с пропусками в критичных числовых признаках и таргете
required_for_drop = numeric_features + ['price']
df_small_clean = df_small.dropna(subset=required_for_drop)
print(f"\nAfter dropna: {df_small_clean.shape[0]:,} rows")

X_num = df_small_clean[numeric_features].copy()
X_cat = df_small_clean[categorical_features].copy()
y = df_small_clean['price'].copy()

# Диагностика целевой переменной
print("\nPrice description:")
print(y.describe())
# Рекомендуем: логарифмирование, если price сильно скошен
print("\nРекомендуется логарифмировать target, если сильно скошен (см. 'count','mean','std','min','max').")

# OneHotEncoding категориальных признаков
if len(categorical_features) > 0:
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore') # drop='first' чтобы избежать мультиколлинеарности (сильная зависимость объектов между собой, нам такой лишний груз не нужен)
    encoded = ohe.fit_transform(X_cat)

    encoded = encoded.toarray()

    try:
        ohe_names = ohe.get_feature_names_out(categorical_features)
    except Exception as e:
        print(e)

    X_encoded = pd.DataFrame(encoded, columns=ohe_names, index=X_num.index) # Создаём датафрейм с закодированными признаками
    X_pre = pd.concat([X_num, X_encoded], axis=1)
else:
    X_pre = X_num.copy() # Если нет категорий, то просто берём числовые признаки

# Здесь мы просматриваем таблицу на наличие строк, в которых у каких-либо признаков есть пропуски. Если они есть - удаляем, если нет - оставляем. Делается это 
#  с помощью создания булевой таблицы, состоящей из true и false, и последующим фильтром
mask_notna = X_pre.notna().all(axis=1)
X_pre = X_pre.loc[mask_notna]
y = y.loc[mask_notna]

# Нормализация числовых признаков. Это нужно для повышения качества модели.
# Таким образом мы приводим числовые признаки к одному масштабу
scaler = StandardScaler()
num_cols_present = [c for c in numeric_features if c in X_pre.columns]
if len(num_cols_present) > 0:
    X_pre[num_cols_present] = scaler.fit_transform(X_pre[num_cols_present])

# Логарифмируем целевую переменную, это нужно для того, чтобы в будущем уменьшить влияение выбросов - то есть слишком низких цен (в датасете есть цена в 0, лол) или слишком высоких.
LOG_TARGET = True
if LOG_TARGET:
    y_trans = np.log1p(y)
else:
    y_trans = y.values

# Разделим на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X_pre, y_trans, test_size=0.2, random_state=42)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")


# Запуск моделей

results = []
# Важно отметить, что значения метрик такие, какие они есть, потому что значения цены логарифмировалось. То есть значение mse и других метрик, которые мы видим, относятся не к рублям по факту, а к логарифму 1+p
# 1) Наш KNN на подвыборке (если X_train не слишком большой)
print("\n1) Мой KNN on на \"маленькой\" подвыборке")
try:
    our_knn = KNNRegressor(n_neighbors=5)
    our_knn.fit(X_train.values, y_train.values)
    our_knn_pred = our_knn.predict(X_test.values, batch_size=200)  
    our_knn_mse, our_knn_mae, our_knn_rmse, our_knn_r2 = calculate_metrics(y_test, our_knn_pred)
    print_metrics("Мой KNN", our_knn_mse, our_knn_mae, our_knn_rmse, our_knn_r2)
    results.append(('Мой KNN', our_knn_mse, our_knn_mae, our_knn_rmse, our_knn_r2))
except MemoryError as e:
    print("Ошибка помяти:", e)

# 2) Мой LinearRegression 
print("\n2) Мой LinearRegression с тремя вариантами оптимизаторов (SGD / Momentum / AdaGrad)")

our_lr_sgd_pred = None
our_lr_momentum_pred = None
our_lr_adagrad_pred = None
for opt in ['SGD', 'Momentum', 'AdaGrad']:
    print(f"\nTraining Our LinearRegression ({opt})")
    our_lr = LinearRegression(
        learning_rate=0.1,   
        optimization=opt,
        max_iter=1000
    )

    our_lr.fit(X_train, y_train)
    our_lr_pred = our_lr.predict(X_test)

    if opt == 'SGD':
        our_lr_sgd_pred = our_lr_pred
    elif opt == 'Momentum':
        our_lr_momentum_pred = our_lr_pred
    elif opt == 'AdaGrad':
        our_lr_adagrad_pred = our_lr_pred

    # вычисляем метрики
    our_lr_mse, our_lr_mae, our_lr_rmse, our_lr_r2 = calculate_metrics(y_test, our_lr_pred)

    # выводим и добавляем в результаты
    print_metrics(f"Мой LinearRegression ({opt})", our_lr_mse, our_lr_mae, our_lr_rmse, our_lr_r2)
    results.append((f"Мой LR ({opt})", our_lr_mse, our_lr_mae, our_lr_rmse, our_lr_r2))


# 3) Sklearn KNN
print("\n3) Sklearn KNeighborsRegressor")
sk_knn = KNeighborsRegressor(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)
# NOTE: если X_train очень большой, этот fit может занять время и память, но он гораздо оптимизированней
sk_knn.fit(X_train, y_train)
sk_knn_pred = sk_knn.predict(X_test)
sk_knn_mse, sk_knn_mae, sk_knn_rmse, sk_knn_r2 = calculate_metrics(y_test, sk_knn_pred)
print_metrics("Sklearn KNN", sk_knn_mse, sk_knn_mae, sk_knn_rmse, sk_knn_r2)
results.append(('Sklearn KNN', sk_knn_mse, sk_knn_mae, sk_knn_rmse, sk_knn_r2))

# 4) Sklearn LinearRegression
print("\n2) Sklearn LinearRegression")
sk_lr = SklearnLinearRegression()
sk_lr.fit(X_train, y_train)
sk_lr_pred = sk_lr.predict(X_test)
sk_lr_mse, sk_lr_mae, sk_lr_rmse, sk_lr_r2 = calculate_metrics(y_test, sk_lr_pred)
print_metrics("Sklearn LR", sk_lr_mse, sk_lr_mae, sk_lr_rmse, sk_lr_r2)
results.append(('Sklearn LR', sk_lr_mse, sk_lr_mae, sk_lr_rmse, sk_lr_r2))

# Сводная таблица результатов. Опять же, важно отметить, что результаты представлены в логарифмическом масштабе
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'MAE', 'RMSE', 'R2'])
print("\nСВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print(results_df.to_string(index=False, float_format='%.4f'))

if LOG_TARGET:
    print("\nПример интерпретации результатов")
    inv_true = np.expm1(y_test)

    models_to_check = [
        ('Мой KNN', our_knn_pred),
        ('Мой LR (SGD)', our_lr_sgd_pred),
        ('Мой LR (Momentum)', our_lr_momentum_pred),
        ('Мой LR (AdaGrad)', our_lr_adagrad_pred),
        ('Sklearn LR', sk_lr_pred),
        ('Sklearn KNN', sk_knn_pred),
    ]

    for name, preds in models_to_check:
        print(f"\n{name}:")
        inv_pred = np.expm1(preds)
        for t, p in zip(inv_true[:5].ravel(), inv_pred[:5].ravel()):  
            print(f"Настоящее значение={t:.2f}, Предположение={p:.2f}")

