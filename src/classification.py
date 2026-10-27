import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.errors_history = []

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _step_function(self, z):
        return np.where(z >= 0.0, 1.0, 0.0)

    def fit(self, X, y):
        X_b = self._add_bias(X)
        n_samples, n_features_with_bias = X_b.shape

        self.w = np.zeros((n_features_with_bias, 1), dtype=np.float64)
        self.errors_history = []

        for _ in range(self.epochs):
            total_errors = 0

            for i in range(n_samples):
                xi = X_b[i : i + 1]  # shape: (1, n_features)
                target = y[i : i + 1]  # shape: (1, 1)

                z = np.dot(xi, self.w)
                y_pred = self._step_function(z)

                update = self.learning_rate * (target - y_pred)
                self.w += np.dot(xi.T, update)

                if update != 0.0:
                    total_errors += 1

            self.errors_history.append(total_errors)

            # Early stopping if completely converged
            if total_errors == 0:
                break

        return self

    def predict(self, X):
        X_b = self._add_bias(X)
        z = np.dot(X_b, self.w)
        return self._step_function(z)

    @property
    def bias(self):
        return self.w[0, 0] if self.w is not None else None

    @property
    def weights(self):
        return self.w[1:, :] if self.w is not None else None
