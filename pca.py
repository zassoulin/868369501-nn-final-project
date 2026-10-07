"""Part B -- our own implementation of Principal Component Analysis (PCA).

Same interface as `sklearn.decomposition.PCA` (fit / transform / fit_transform /
inverse_transform, `components_`, `explained_variance_ratio_`), so it can be dropped into the
notebook in its place. The algorithm itself is implemented here with numpy only:
center the data -> covariance matrix -> eigen-decomposition -> sort eigenvectors by eigenvalue
-> project onto the top-k eigenvectors.
"""
import numpy as np


class PCA:
    """PCA via eigen-decomposition of the sample covariance matrix."""

    def __init__(self, n_components=None):
        # None keeps all min(n_samples, n_features) components
        self.n_components = n_components

    def fit(self, X):
        """Learn the principal axes of X (shape: n_samples x n_features). Returns self."""
        X = np.asarray(X, dtype=float)
        n, p = X.shape

        # 1. center the data: PCA describes variance around the mean
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # 2. sample covariance matrix (p x p, symmetric)
        C = Xc.T @ Xc / (n - 1)

        # 3. eigen-decomposition. eigh is meant for symmetric matrices: real eigenvalues
        # (ascending) and orthonormal eigenvectors (columns). Tiny negative eigenvalues are
        # just floating-point noise of a positive semi-definite matrix, so clip them to 0.
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 0.0, None)

        # 4. sort by eigenvalue, descending: the first axis is the direction of maximum variance
        order = np.argsort(eigvals)[::-1]

        # 5. number of components to keep
        k = min(n, p) if self.n_components is None else self.n_components
        if not 1 <= k <= min(n, p):
            raise ValueError(f"n_components={k} must be between 1 and min(n_samples, n_features)={min(n, p)}")

        # 6. principal axes as rows (unit length, mutually orthogonal)
        components = eigvecs[:, order[:k]].T

        # 7. an eigenvector is only defined up to its sign (v and -v are equally valid). For a
        # deterministic result we use the same convention as sklearn: flip each axis so that its
        # largest-magnitude entry is positive.
        max_idx = np.argmax(np.abs(components), axis=1)
        components *= np.sign(components[np.arange(k), max_idx])[:, None]

        self.components_ = components
        self.n_components_ = k
        # variance along each kept axis = its eigenvalue; the ratio is relative to the TOTAL
        # variance (all eigenvalues), so with k < p the ratios sum to less than 1
        self.explained_variance_ = eigvals[order[:k]]
        self.explained_variance_ratio_ = self.explained_variance_ / eigvals.sum()
        return self

    def transform(self, X):
        """Project X onto the principal axes -> shape (n_samples, n_components)."""
        self._check_fitted()
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """Map PCA coordinates back to the original feature space (the best rank-k
        reconstruction of the data in the mean-squared-error sense)."""
        self._check_fitted()
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_

    def _check_fitted(self):
        if not hasattr(self, "components_"):
            raise RuntimeError("This PCA instance is not fitted yet; call fit() first.")
