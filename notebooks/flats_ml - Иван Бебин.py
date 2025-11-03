import numpy as np

class KNNRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the mean of these labels
        return np.mean(k_nearest_labels)

# Example usage
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, method='sgd', momentum=0.9, epsilon=1e-8,):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.momentum = momentum
        self.epsilon = epsilon
        self.weights = None
        self.bias = None
        self.velocity = None
        self.cache = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.velocity = np.zeros(n_features)
        self.cache = np.zeros(n_features)

        for _ in range(self.n_iterations):
            if self.method == 'sgd':
                # for _ in range(self.n_iterations):
                #     indices = np.random.permutation(n_samples)
                #     X_shuffled = X[indices]
                #     y_shuffled = y[indices]

                #     for i in range(0, n_samples, self.batch_size):
                #         X_i = X_shuffled[i:i + self.batch_size]
                #         y_i = y_shuffled[i:i + self.batch_size]
                self._update_sgd(X, y)
            elif self.method == 'momentum':
                self._update_momentum(X, y)
            elif self.method == 'adagrad':
                self._update_adagrad(X, y)
            else:
                raise ValueError("Invalid method. Choose 'sgd', 'momentum', or 'adagrad'.")

    def _update_sgd(self, X, y):
        y_predicted = self._predict(X)
        dw = (2 / X.shape[0]) * np.dot(X.T, (y_predicted - y))
        db = (2 / X.shape[0]) * np.sum(y_predicted - y)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def _update_momentum(self, X, y):
        y_predicted = self._predict(X)
        dw = (2 / X.shape[0]) * np.dot(X.T, (y_predicted - y))
        db = (2 / X.shape[0]) * np.sum(y_predicted - y)
        self.velocity = self.momentum * self.velocity + self.learning_rate * dw
        self.weights -= self.velocity
        self.bias -= self.learning_rate * db

    def _update_adagrad(self, X, y):
        y_predicted = self._predict(X)
        dw = (2 / X.shape[0]) * np.dot(X.T, (y_predicted - y))
        db = (2 / X.shape[0]) * np.sum(y_predicted - y)
        self.cache += dw ** 2
        self.weights -= self.learning_rate * dw / (np.sqrt(self.cache) + self.epsilon)
        self.bias -= self.learning_rate * db

    def _predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return self._predict(X)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    # Пример данных
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
    y = np.array([3, 5, 7, 9, 11, 13, 15, 17])

    x_test = np.array([[9,10],[5,7]])
    y_test = np.array([19, 12])

    # Создание и обучение модели с использованием SGD
    model_sgd = LinearRegression(method='sgd', batch_size=2, n_iterations=1000)
    model_sgd.fit(X, y)
    y_pred_sgd = model_sgd.predict(x_test)
    print("SGD Predictions:", y_pred_sgd)
    mse = mean_squared_error(y_test, y_pred_sgd)
    print("Mean Squared Error:", mse)
    # Создание и обучение модели с использованием Momentum
    model_momentum = LinearRegression(method='momentum')
    model_momentum.fit(X, y)
    y_pred_mom = model_momentum.predict(x_test)
    print("Momentum Predictions:", y_pred_mom)
    mse = mean_squared_error(y_test, y_pred_mom)
    print("Mean Squared Error:", mse)
    # Создание и обучение модели с использованием AdaGrad
    model_adagrad = LinearRegression(method='adagrad')
    model_adagrad.fit(X, y)
    y_pred_ada = model_adagrad.predict(x_test)
    print("AdaGrad Predictions:", y_pred_ada)
    mse = mean_squared_error(y_test, y_pred_ada)
    print("Mean Squared Error:", mse)