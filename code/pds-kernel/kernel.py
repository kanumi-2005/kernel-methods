import numpy as np
from scipy.spatial.distance import cdist


def linear_kernel(X1, X2):
    """
    Compute the linear kernel matrix between two datasets.

    The linear kernel is equivalent to the standard inner product between
    feature vectors.

    Parameters
    ----------
    X1 : ndarray of shape (n_samples_1, n_features)
        First input dataset.

    X2 : ndarray of shape (n_samples_2, n_features)
        Second input dataset.

    Returns
    -------
    K : ndarray of shape (n_samples_1, n_samples_2)
        Linear kernel matrix.

    Notes
    -----
    The linear kernel is defined as ``K(x, y) = x^T y``.

    Examples
    --------
    >>> X1 = np.array([[1, 2]])
    >>> X2 = np.array([[3, 4]])
    >>> linear_kernel(X1, X2)
    array([[11]])
    """
    return X1 @ X2.T


def poly_kernel(X1, X2, degree=3, coef0=1, gamma=1):
    """
    Compute the polynomial kernel matrix between two datasets.

    The polynomial kernel computes similarities by applying a polynomial
    transformation to the inner product between feature vectors.

    Parameters
    ----------
    X1 : ndarray of shape (n_samples_1, n_features)
        First input dataset.

    X2 : ndarray of shape (n_samples_2, n_features)
        Second input dataset.

    degree : int, default=3
        Degree of the polynomial kernel.

    coef0 : float, default=1
        Independent term added to the kernel function.

    gamma : float, default=1
        Scaling factor applied to the inner product.

    Returns
    -------
    K : ndarray of shape (n_samples_1, n_samples_2)
        Polynomial kernel matrix.

    Notes
    -----
    The polynomial kernel is defined as

    ``K(x, y) = (gamma * x^T y + coef0)^degree``.

    Examples
    --------
    >>> X1 = np.array([[1, 2]])
    >>> X2 = np.array([[3, 4]])
    >>> poly_kernel(X1, X2, degree=2)
    array([[144]])
    """
    return (gamma * X1 @ X2.T + coef0) ** degree


def rbf_kernel(X1, X2, gamma=1.0):
    """
    Compute the radial basis function (RBF) kernel matrix.

    The RBF kernel measures similarity based on the squared Euclidean
    distance between feature vectors.

    Parameters
    ----------
    X1 : ndarray of shape (n_samples_1, n_features)
        First input dataset.

    X2 : ndarray of shape (n_samples_2, n_features)
        Second input dataset.

    gamma : float, default=1.0
        Scaling parameter controlling the kernel width.

    Returns
    -------
    K : ndarray of shape (n_samples_1, n_samples_2)
        RBF kernel matrix.

    Notes
    -----
    The RBF kernel is defined as

    ``K(x, y) = exp(-gamma * ||x - y||^2)``.

    Examples
    --------
    >>> X1 = np.array([[1, 2]])
    >>> X2 = np.array([[1, 2]])
    >>> rbf_kernel(X1, X2)
    array([[1.]])
    """
    dists = cdist(X1, X2, metric='sqeuclidean')
    return np.exp(-gamma * dists)


def gaussian_kernel(X1, X2, sigma=1.0):
    """
    Compute the Gaussian kernel matrix between two datasets.

    The Gaussian kernel is a special case of the RBF kernel parameterized
    by the standard deviation ``sigma``.

    Parameters
    ----------
    X1 : ndarray of shape (n_samples_1, n_features)
        First input dataset.

    X2 : ndarray of shape (n_samples_2, n_features)
        Second input dataset.

    sigma : float, default=1.0
        Bandwidth parameter of the Gaussian kernel.

    Returns
    -------
    K : ndarray of shape (n_samples_1, n_samples_2)
        Gaussian kernel matrix.

    Notes
    -----
    The Gaussian kernel is defined as

    ``K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))``.

    Examples
    --------
    >>> X1 = np.array([[1, 2]])
    >>> X2 = np.array([[1, 2]])
    >>> gaussian_kernel(X1, X2)
    array([[1.]])
    """
    dists = cdist(X1, X2, metric='sqeuclidean')
    return np.exp(-dists / (2 * sigma ** 2))


def sigmoid_kernel(X1, X2, gamma=1.0, coef0=0):
    """
    Compute the sigmoid kernel matrix between two datasets.

    The sigmoid kernel applies the hyperbolic tangent function to the
    scaled inner product between feature vectors.

    Parameters
    ----------
    X1 : ndarray of shape (n_samples_1, n_features)
        First input dataset.

    X2 : ndarray of shape (n_samples_2, n_features)
        Second input dataset.

    gamma : float, default=1.0
        Scaling factor applied to the inner product.

    coef0 : float, default=0
        Independent term added before applying the hyperbolic tangent.

    Returns
    -------
    K : ndarray of shape (n_samples_1, n_samples_2)
        Sigmoid kernel matrix.

    Notes
    -----
    The sigmoid kernel is defined as

    ``K(x, y) = tanh(gamma * x^T y + coef0)``.

    Examples
    --------
    >>> X1 = np.array([[1, 2]])
    >>> X2 = np.array([[3, 4]])
    >>> sigmoid_kernel(X1, X2, gamma=0.1)
    array([[0.80049902]])
    """
    return np.tanh(gamma * X1 @ X2.T + coef0)


if __name__ == "__main__":
    np.random.seed(42)
    X1 = np.random.randn(4, 3)
    X2 = np.random.randn(5, 3)

    print("Linear kernel:\n",   linear_kernel(X1, X2))
    print("Poly kernel:\n",     poly_kernel(X1, X2, degree=3))
    print("RBF kernel:\n",      rbf_kernel(X1, X2, gamma=0.5))
    print("Gaussian kernel:\n", gaussian_kernel(X1, X2, sigma=1.0))
    print("Sigmoid kernel:\n",  sigmoid_kernel(X1, X2, gamma=0.1))
