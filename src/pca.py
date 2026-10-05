import numpy as np


class StandardScalerFromScratch:
    """Chuan hoa z-score: (x - mean) / std."""

    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < self.eps] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PCAFromScratch:
    """PCA bang tri rieng cua ma tran hiep phuong sai (eigen decomposition)."""

    def __init__(self, n_components=100):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / (n_samples - 1)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        eigvals[eigvals < 0] = 0.0
        k = int(min(self.n_components, n_features))
        self.components_ = eigvecs[:, :k].T.astype(np.float32)
        self.explained_variance_ = eigvals[:k]
        total = eigvals.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / (total + 1e-12)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        return Z @ self.components_ + self.mean_
