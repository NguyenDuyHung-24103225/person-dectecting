import numpy as np


class LinearSVMFromScratch:
    """Soft-margin Linear SVM (hinge loss + L2) dung sub-gradient descent.
    Nhan y thuoc {-1, +1}."""

    def __init__(self, learning_rate=0.001, n_iters=2000, C=1.0, batch_size=256, random_state=42):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.C = C
        self.batch_size = batch_size
        self.random_state = random_state
        self.w = None
        self.b = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        y_svm = np.where(y > 0, 1.0, -1.0)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0
        for _ in range(self.n_iters):
            idx = rng.integers(0, n_samples, size=min(self.batch_size, n_samples))
            Xb = X[idx]
            yb = y_svm[idx]
            margins = yb * (Xb @ self.w + self.b)
            mask = (margins < 1).astype(np.float32)
            dw = self.w - self.C * (Xb * (yb * mask)[:, None]).sum(axis=0)
            db = -self.C * (yb * mask).sum()
            self.w -= self.lr * dw
            self.b -= self.lr * db
            loss = 0.5 * float(np.sum(self.w ** 2)) + self.C * float(np.mean(np.maximum(0, 1 - margins)))
            self.loss_history_.append(float(loss))
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X @ self.w + self.b

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)
