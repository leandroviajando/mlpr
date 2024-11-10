import pathlib
from typing import NamedTuple, Optional, Union, overload

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


@overload
def mse(*, y_pred: np.ndarray, y: np.ndarray) -> float: ...


@overload
def mse(*, y_pred: np.floating, y: np.floating) -> float: ...


@overload
def mse(*, y_pred: float, y: float) -> float: ...


def mse(
    *,
    y_pred: Union[np.ndarray, np.floating, float],
    y: Union[np.ndarray, np.floating, float],
) -> float:
    """Mean Squared Error (MSE)"""

    return float(np.mean((y_pred - y) ** 2))


def plot_line_graph_and_histogram(amp_data: np.ndarray) -> None:

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(amp_data, label="Amplitude Data")
    plt.title("Line Graph of Amplitude Data")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")

    plt.subplot(1, 2, 2)
    plt.hist(amp_data, bins=30, alpha=0.7, color="blue")
    plt.title("Histogram of Amplitudes")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_residuals_and_amplitudes_histograms(
    *, amp_data: np.ndarray, residuals: np.ndarray
) -> None:

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    plt.title("Histogram of Residuals on Best Model Validation Data")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    plt.subplot(1, 2, 2)
    plt.hist(amp_data, bins=30, edgecolor="black", alpha=0.7)
    plt.title("Histogram of Amplitudes")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


class Plot(NamedTuple):
    points: Union[np.ndarray, np.floating]
    label: str
    color: str


def plot_amplitudes_against_time_points(
    X: np.ndarray, y: np.ndarray, *plots: Plot
) -> None:

    plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(0, 19 / 20, num=20), X, marker="o", label="Amplitude vs Time")
    plt.scatter([1], [y], color="red", label="Prediction target")

    for plot in plots:
        if isinstance(plot.points, np.floating):
            plt.scatter([1], [plot.points], color=plot.color, label=plot.label)
        else:
            plt.plot(
                np.linspace(0, 1, len(plot.points)),
                plot.points,
                color=plot.color,
                label=plot.label,
            )

    plt.title("Amplitude vs Time")
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


def fit_and_extrapolate(
    X: np.ndarray, deg: int, extrapolate_steps: Optional[int] = None
) -> np.ndarray:
    """Fits a polynomial of given degree, and optionally extrapolates a given number of steps."""

    coefficients = np.polyfit(np.arange(len(X)), X, deg=deg)

    return np.polyval(coefficients, np.arange(len(X) + (extrapolate_steps or 0)))


def Phi(C: int, K: int) -> np.ndarray:
    """Returns the design matrix of shape C x K (context length x number of basis functions)."""

    t = np.linspace(0, (C - 1) / 20, C)
    Phi_C_by_K = np.zeros((C, K))

    for c in range(C):
        Phi_C_by_K[c] = [t[c] ** k for k in range(K)]

    return Phi_C_by_K


def make_vv(C: int, K: int) -> np.ndarray:
    """Returns the C-dimensional vector v = Phi (Phi^T Phi)^{-1} phi(t=1) for a model
    with K features and a context of C previous amplitudes."""

    t1 = 1
    Phi_C_by_K = Phi(C, K)
    phi_t1 = np.array([t1**k for k in range(K)])  # vector phi(t=1) with K features

    try:
        v = Phi_C_by_K @ np.linalg.inv(Phi_C_by_K.T @ Phi_C_by_K) @ phi_t1
    except np.linalg.LinAlgError:
        v = Phi_C_by_K @ np.linalg.pinv(Phi_C_by_K.T @ Phi_C_by_K) @ phi_t1

    return v


if __name__ == "__main__":
    amp_data: np.ndarray = np.load(
        pathlib.Path(__file__).parent / "data" / "amp_data.npz"
    )["amp_data"]

    plot_line_graph_and_histogram(amp_data)

    amp_data = amp_data[: len(amp_data) - len(amp_data) % 21].reshape((-1, 21))
    np.random.shuffle(amp_data)

    """The shuffling means that our training, validation and testing datasets
    all come from the same distribution. Creating this ideal setting can be
    useful when first learning about some different methods. Although we should
    remember our models might not generalize well to new data with different
    distributions."""

    train_size = int(0.7 * len(amp_data))
    val_size = int(0.15 * len(amp_data))

    X_shuf_train = amp_data[:train_size, :-1]
    y_shuf_train = amp_data[:train_size, -1]

    X_shuf_val = amp_data[train_size : train_size + val_size, :-1]
    y_shuf_val = amp_data[train_size : train_size + val_size, -1]

    X_shuf_test = amp_data[train_size + val_size :, :-1]
    y_shuf_test = amp_data[train_size + val_size :, -1]

    """Given just one row of inputs, we could fit a curve of amplitude against
    time through the 20 points, and extrapolate it one step into the future."""

    linear_fit = fit_and_extrapolate(X_shuf_train[0], deg=1, extrapolate_steps=1)
    quartic_fit = fit_and_extrapolate(X_shuf_train[0], deg=4, extrapolate_steps=1)

    """The linear fit might be better when using only the most recent two points
    (at times t=18/20 and t=19/20) because it allows the model to focus on the
    most current trend in the data. This can help capture short-term fluctuations
    and provide a more accurate prediction for the immediate next time step, as
    it reduces the influence of older data that may not be as relevant to the
    current state of the system.

    On the other hand, the quartic fit might benefit from a longer context
    because it is designed to capture more complex relationships in the data.
    By using more points, the quartic model can account for variations and trends
    that may not be evident in just the last two points. This allows it to better
    understand the underlying patterns and make more informed predictions,
    especially when the data exhibits non-linear behaviour over a longer timeframe."""

    C = len(X_shuf_train[0])
    v_linear = make_vv(C=C, K=2)
    v_quartic = make_vv(C=C, K=5)
    linear_pred: np.floating = v_linear.T @ X_shuf_train[0]
    quartic_pred: np.floating = v_quartic.T @ X_shuf_train[0]

    """The advantage of identifying the prediction as a linear combination, v.T @ x,
    is that we can compute v once and rapidly make next-step predictions for N
    different sequences of C amplitudes. We don't need to fit N separate models!
    
    The predictions are the same as the extrapolated polynomials at time t = 1."""

    plot_amplitudes_against_time_points(
        X_shuf_train[0],
        y_shuf_train[0],
        Plot(linear_fit, label="Linear fit", color="blue"),
        Plot(quartic_fit, label="Quartic fit", color="orange"),
        Plot(linear_pred, label="Linear prediction", color="green"),
        Plot(quartic_pred, label="Quartic prediction", color="black"),
    )

    print(
        f"y_shuf_train[0] = {y_shuf_train[0]:.2f}, linear_pred = {linear_pred:.2f}, quartic_pred = {quartic_pred:.2f}"
    )

    """Which setting of C and K gives the smallest square error?"""

    best_C = -1
    best_K = -1
    smallest_error = float("inf")

    for C in range(2, 21):
        for K in range(2, 11):
            v = make_vv(C=C, K=K)
            predictions = X_shuf_train[:, -C:] @ v
            error = mse(y_pred=predictions, y=y_shuf_train)

            if error < smallest_error:
                smallest_error, best_C, best_K = error, C, K

    print(f"Best context length (C): {best_C}")
    print(f"Best number of basis functions (K): {best_K}")
    v = make_vv(C=best_C, K=best_K)
    print(
        f"Mean square error (MSE) on the train set: {mse(y_pred=X_shuf_train[:, -best_C:] @ v.T, y=y_shuf_train)}"
    )
    print(
        f"Mean square error (MSE) on the val set: {mse(y_pred=X_shuf_val[:, -best_C:] @ v.T, y=y_shuf_val)}"
    )
    print(
        f"Mean square error (MSE) on the test set: {mse(y_pred=X_shuf_test[:, -best_C:] @ v.T, y=y_shuf_test)}"
    )

    """It's possible we could do better by picking different basis functions.
    However, no matter which basis functions we pick, a linear model fitted by
    least squares will predict the next amplitude using a linear combination
    of the previous amplitudes.

    Given a large dataset, we can try to fit a good linear combination directly,
    without needing to specify basis functions. Using standard linear least squares
    fitting code, we can find the vector that minimises  sum_1^N (y^(n) - v^T x^(n))^2
    on the training set."""

    K = 1
    best_C_train = -1
    smallest_error_train = float("inf")

    for C in range(1, 21):
        v = make_vv(C=C, K=K)
        predictions_train = X_shuf_train[:, -C:] @ v
        error_train = mse(y_pred=predictions_train, y=y_shuf_train)

        if error_train < smallest_error_train:
            smallest_error_train, best_C_train = error_train, C

    print(
        f"Best context length (C) on training set: {best_C_train}, smallest square error: {smallest_error_train}"
    )

    best_C_val = -1
    smallest_error_val = float("inf")

    for C in range(1, 21):
        v = make_vv(C=C, K=K)
        predictions_val = X_shuf_val[:, -C:] @ v
        error_val = mse(y_pred=predictions_val, y=y_shuf_val)

        if error_val < smallest_error_val:
            smallest_error_val, best_C_val = error_val, C

    print(
        f"Best context length (C) on validation set: {best_C_val}, smallest square error: {smallest_error_val}"
    )

    v_best = make_vv(C=best_C_val, K=K)
    predictions_best = X_shuf_val[:, -best_C_val:] @ v_best

    residuals = predictions_best - y_shuf_val

    plot_residuals_and_amplitudes_histograms(amp_data=amp_data, residuals=residuals)
