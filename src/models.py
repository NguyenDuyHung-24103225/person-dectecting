import numpy as np


def add_bias(X):
    X = np.asarray(X, dtype=np.float32)
    return np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])


class LinearRegressionFromScratch:
    """Linear Regression dung Gradient Descent.
    Voi bai toan phan lop: y = 1 neu co nguoi, 0 neu khong.
    Du doan: y_hat >= 0.5 -> co nguoi."""

    def __init__(self, learning_rate=0.01, n_iters=3000, reg=0.0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.reg = reg
        self.w = None
        self.b = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0
        for _ in range(self.n_iters):
            y_hat = X @ self.w + self.b
            err = y_hat - y
            dw = (X.T @ err) / n_samples + self.reg * self.w
            db = err.mean()
            self.w -= self.lr * dw
            self.b -= self.lr * db
            loss = np.mean(err ** 2) / 2.0
            self.loss_history_.append(float(loss))
        return self

    def predict_scores(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X @ self.w + self.b

    def predict(self, X, threshold=0.5):
        return (self.predict_scores(X) >= threshold).astype(int)


class LogisticRegressionFromScratch:
    """Logistic Regression dung Gradient Descent + sigmoid + L2."""

    def __init__(self, learning_rate=0.1, n_iters=3000, reg=0.0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.reg = reg
        self.w = None
        self.b = None
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0
        for _ in range(self.n_iters):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            err = p - y
            dw = (X.T @ err) / n_samples + self.reg * self.w
            db = err.mean()
            self.w -= self.lr * dw
            self.b -= self.lr * db
            eps = 1e-12
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            loss += 0.5 * self.reg * float(np.sum(self.w ** 2))
            self.loss_history_.append(float(loss))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
