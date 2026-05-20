import numpy as np
import pandas as pd
from scipy import stats


def gaussian_kernel(x, x_prime, sigma=1.0):
    """
    Compute the Gaussian kernel between two input vectors.

    The Gaussian kernel measures similarity between two vectors by applying
    an exponential decay to their squared Euclidean distance.

    Parameters
    ----------
    x : ndarray of shape (n_features,)
        First input vector.

    x_prime : ndarray of shape (n_features,)
        Second input vector.

    sigma : float, default=1.0
        Bandwidth parameter of the Gaussian kernel.

    Returns
    -------
    kernel_value : float
        Gaussian kernel value between ``x`` and ``x_prime``.

    Notes
    -----
    The kernel is defined as ``exp(-||x - x_prime||^2 / (2 sigma^2))``.

    Examples
    --------
    >>> x = np.array([1.0, 2.0])
    >>> xp = np.array([1.0, 3.0])
    >>> gaussian_kernel(x, xp)
    np.float64(0.6065306597126334)
    """
    return np.exp(-np.sum((x - x_prime) ** 2) / (2 * sigma ** 2))


def laplacian_kernel(x, x_prime):
    """
    Compute the Laplacian kernel between two input vectors.

    The Laplacian kernel measures similarity by applying an exponential decay
    to the Manhattan distance between two vectors.

    Parameters
    ----------
    x : ndarray of shape (n_features,)
        First input vector.

    x_prime : ndarray of shape (n_features,)
        Second input vector.

    Returns
    -------
    kernel_value : float
        Laplacian kernel value between ``x`` and ``x_prime``.

    Notes
    -----
    The kernel is defined as ``exp(-sum(abs(x - x_prime)))``.

    Examples
    --------
    >>> x = np.array([1.0, 2.0])
    >>> xp = np.array([1.0, 3.0])
    >>> laplacian_kernel(x, xp)
    np.float64(0.36787944117144233)
    """
    return np.exp(-np.sum(np.abs(x - x_prime)))


def cauchy_kernel(x, x_prime):
    """
    Compute the Cauchy kernel between two input vectors.

    The Cauchy kernel measures similarity by multiplying the inverse quadratic
    terms of the coordinate-wise differences.

    Parameters
    ----------
    x : ndarray of shape (n_features,)
        First input vector.

    x_prime : ndarray of shape (n_features,)
        Second input vector.

    Returns
    -------
    kernel_value : float
        Cauchy kernel value between ``x`` and ``x_prime``.

    Notes
    -----
    The kernel is defined as ``prod(1 / (1 + (x - x_prime)^2))``.

    Examples
    --------
    >>> x = np.array([1.0, 2.0])
    >>> xp = np.array([1.0, 3.0])
    >>> cauchy_kernel(x, xp)
    np.float64(0.5)
    """
    return np.prod(1 / (1 + (x - x_prime) ** 2))


def sample_gaussian_p(D_samples, N, sigma=1.0):
    """
    Draw random frequency samples for the Gaussian kernel.

    This function samples random vectors from the distribution associated with
    the Gaussian kernel in the random Fourier feature approximation.

    Parameters
    ----------
    D_samples : int
        Number of random frequency samples to generate.

    N : int
        Dimension of each random frequency sample.

    sigma : float, default=1.0
        Scale parameter used to control the sampling variance.

    Returns
    -------
    omegas : ndarray of shape (D_samples, N)
        Random frequency samples drawn from a normal distribution.

    Notes
    -----
    Samples are drawn from ``Normal(0, 1 / sigma)``.

    Examples
    --------
    >>> sample_gaussian_p(2, 3).shape
    (2, 3)
    """
    return np.random.normal(0, 1/sigma, size=(D_samples, N))


def sample_laplacian_p(D_samples, N):
    """
    Draw random frequency samples for the Laplacian kernel.

    This function samples random vectors from the distribution associated with
    the Laplacian kernel in the random Fourier feature approximation.

    Parameters
    ----------
    D_samples : int
        Number of random frequency samples to generate.

    N : int
        Dimension of each random frequency sample.

    Returns
    -------
    omegas : ndarray of shape (D_samples, N)
        Random frequency samples drawn from a standard Cauchy distribution.

    Notes
    -----
    Samples are drawn from the standard Cauchy distribution.

    Examples
    --------
    >>> sample_laplacian_p(2, 3).shape
    (2, 3)
    """
    return np.random.standard_cauchy(size=(D_samples, N))


def sample_cauchy_p(D_samples, N):
    """
    Draw random frequency samples for the Cauchy kernel.

    This function samples random vectors from the distribution associated with
    the Cauchy kernel in the random Fourier feature approximation.

    Parameters
    ----------
    D_samples : int
        Number of random frequency samples to generate.

    N : int
        Dimension of each random frequency sample.

    Returns
    -------
    omegas : ndarray of shape (D_samples, N)
        Random frequency samples drawn from a Laplace distribution.

    Notes
    -----
    Samples are drawn from ``Laplace(0, 1)``.

    Examples
    --------
    >>> sample_cauchy_p(2, 3).shape
    (2, 3)
    """
    return np.random.laplace(0, 1, size=(D_samples, N))


def get_samples(x, x_prime, sample_fn, D_samples):
    """
    Generate Monte Carlo samples for kernel approximation.

    This function draws random frequency vectors and computes cosine-based
    samples whose expectation approximates a shift-invariant kernel value.

    Parameters
    ----------
    x : ndarray of shape (n_features,)
        First input vector.

    x_prime : ndarray of shape (n_features,)
        Second input vector.

    sample_fn : callable
        Function used to generate random frequency samples. It must accept
        ``D_samples`` and ``N`` as arguments and return an array of shape
        ``(D_samples, N)``.

    D_samples : int
        Number of Monte Carlo samples to generate.

    Returns
    -------
    samples : ndarray of shape (D_samples,)
        Monte Carlo samples used to approximate the kernel value.

    Notes
    -----
    Each sample is computed as
    ``cos(omega @ x) cos(omega @ x_prime) +
    sin(omega @ x) sin(omega @ x_prime)``.

    By the cosine difference identity, this is equivalent to
    ``cos(omega @ (x - x_prime))``.

    Examples
    --------
    >>> x = np.array([1.0, 2.0])
    >>> xp = np.array([1.0, 3.0])
    >>> samples = get_samples(x, xp, sample_gaussian_p, 10)
    >>> samples.shape
    (10,)
    """
    omegas = sample_fn(D_samples, len(x))
    cos_x  = np.cos(omegas @ x)
    sin_x  = np.sin(omegas @ x)
    cos_xp = np.cos(omegas @ x_prime)
    sin_xp = np.sin(omegas @ x_prime)
    return cos_x * cos_xp + sin_x * sin_xp


if __name__ == "__main__":
    np.random.seed(42)

    N       = 5
    D       = 100000
    n_pairs = 5
    alpha   = 0.05

    pairs = [(np.random.randn(N), np.random.randn(N)) for _ in range(n_pairs)]

    kernels = [
        ("Gaussian",  gaussian_kernel,  sample_gaussian_p),
        ("Laplacian", laplacian_kernel, sample_laplacian_p),
        ("Cauchy",    cauchy_kernel,    sample_cauchy_p),
    ]

    rows = []
    for kernel_name, kernel_fn, sample_fn in kernels:
        for i, (x, xp) in enumerate(pairs):
            samples = get_samples(x, xp, sample_fn, D)
            exact   = kernel_fn(x, xp)

            result  = stats.ttest_1samp(samples, popmean=exact)

            rows.append({
                "Kernel" : kernel_name,
                "Pair"   : i + 1,
                "Exact"  : round(exact, 4),
                "Mean"   : round(np.mean(samples), 4),
                "t"      : round(result.statistic, 3),
                "p-value": round(result.pvalue, 4),
            })

    df = pd.DataFrame(rows)

    for kernel_name in df["Kernel"].unique():
        print(f"\n{'='*65}")
        print(f"Kernel: {kernel_name}  |  D={D}  |  α={alpha}")
        print(f"{'='*65}")
        sub = df[df["Kernel"] == kernel_name].drop(columns="Kernel")
        print(sub.to_string(index=False))
