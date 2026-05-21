import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)


def plot_training_data_with_decision_boundary(
        kernel, ax=None,
        long_title=True,
        support_vectors=True
    ):
    """
    Plot SVM training data with its decision boundary.

    This function trains a support vector classifier on the two-dimensional
    moon dataset and visualizes the predicted decision regions, decision
    boundary, margins, training samples, and optionally support vectors.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'} or callable
        Kernel type used by the support vector classifier.

    ax : matplotlib.axes.Axes, default=None
        Axes object on which to draw the plot. If ``None``, a new figure and
        axes are created.

    long_title : bool, default=True
        Whether to use a descriptive title. If ``False``, only the kernel
        name is used as the plot title.

    support_vectors : bool, default=True
        Whether to display the support vectors as large unfilled circles.

    Returns
    -------
    None
        The function modifies the given axes in place and does not return
        any value.

    Notes
    -----
    The classifier is fitted using :class:`sklearn.svm.SVC` with
    ``gamma=2``. The filled background shows predicted class regions, while
    the contour lines represent the decision function levels ``-1``, ``0``,
    and ``1``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> plot_training_data_with_decision_boundary("rbf", ax=ax)
    >>> plt.show()
    """

    clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    common_params = {"estimator": clf, "X": X, "ax": ax}

    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_title(f"Decision boundary of {kernel} kernel"
                 if long_title else kernel)


if __name__ == "__main__":
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, kernel in zip(axes.ravel(), kernels):
        plot_training_data_with_decision_boundary(kernel, ax=ax)

    plt.tight_layout()
    plt.show()
