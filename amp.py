from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


def plot_line_graph_and_histogram(amp_data: np.ndarray) -> None:
    # Plotting the line graph
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(amp_data, label="Amplitude Data")
    plt.title("Line Graph of Amplitude Data")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")

    # Plotting the histogram
    plt.subplot(1, 2, 2)
    plt.hist(amp_data, bins=30, alpha=0.7, color="blue")
    plt.title("Histogram of Amplitudes")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


class Plot(NamedTuple):
    fit: np.ndarray
    linspace: np.ndarray
    label: str
    color: str


time_points = np.linspace(0, 19 / 20, num=20)
"""time points = the context predicting from, i.e. 20 time steps"""


def plot_amplitudes_against_time_points(
    X: np.ndarray, y: np.ndarray, *plots: Plot
) -> None:

    plt.figure(figsize=(10, 5))
    plt.plot(time_points, X, marker="o", label="Amplitude vs Time")
    plt.scatter([1], [y], color="red", label="Prediction target")

    for plot in plots:
        plt.plot(plot.linspace, plot.fit, label=plot.label, color=plot.color)

    plt.title("Amplitude vs Time")
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


def fit_linear(X: np.ndarray) -> np.ndarray:
    coeffs_linear = np.polyfit(np.linspace(0, 1, num=len(X)), X, 1)
    linear_fit = np.polyval(coeffs_linear, time_points)

    return linear_fit


def fit_quartic(X: np.ndarray) -> np.ndarray:
    phi = np.vstack(
        [
            np.ones(len(X)),
            np.linspace(0, 1, num=len(X)),
            np.linspace(0, 1, num=len(X)) ** 2,
            np.linspace(0, 1, num=len(X)) ** 3,
            np.linspace(0, 1, num=len(X)) ** 4,
        ]
    ).T
    coeffs_quartic = np.linalg.lstsq(phi, X, rcond=None)[0]
    quartic_fit = np.polyval(coeffs_quartic[::-1], time_points)

    return quartic_fit


def Phi(C: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the design matrix of shape C x K (context length x number of basis functions),
    and the time points used to construct it."""

    t = np.linspace(19 / 20, 19 / 20 - (C - 1) / 20, C)
    Phi_matrix = np.zeros((C, K))

    for i in range(C):
        Phi_matrix[i] = [t[i] ** k for k in range(K)]

    return Phi_matrix, t


def make_vv(C: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the vector v = Phi (Phi^T Phi)^{-1} phi(t=1) for a model with K features
    and a context of C previous amplitudes, and the vector t of time points."""

    t1 = 1
    Phi_matrix, t = Phi(C, K)
    phi_t1 = np.array([t1**k for k in range(K)])  # phi(t=1) with K features

    try:
        v = Phi_matrix @ np.linalg.inv(Phi_matrix.T @ Phi_matrix) @ phi_t1
    except np.linalg.LinAlgError:
        v = Phi_matrix @ np.linalg.pinv(Phi_matrix.T @ Phi_matrix) @ phi_t1

    return v, t


def mse(*, y_pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y_pred - y) ** 2))


if __name__ == "__main__":
    amp_data: np.ndarray = np.load("data/amp_data.npz")["amp_data"]

    plot_line_graph_and_histogram(amp_data)

    amp_data = amp_data[: len(amp_data) - len(amp_data) % 21].reshape((-1, 21))
    np.random.shuffle(amp_data)

    """The shuffling means that our training, validation and testing datasets
    all come from the same distribution. Creating this ideal setting can be
    useful when first learning about some different methods. Although we should
    remember our models might not generalize well to new files with different
    distributions."""

    train_size = int(0.7 * len(amp_data))
    val_size = int(0.15 * len(amp_data))

    X_shuf_train = amp_data[:train_size, :-1]
    y_shuf_train = amp_data[:train_size, -1]

    X_shuf_val = amp_data[train_size : train_size + val_size, :-1]
    y_shuf_val = amp_data[train_size : train_size + val_size, -1]

    X_shuf_test = amp_data[train_size + val_size :, :-1]
    y_shuf_test = amp_data[train_size + val_size :, -1]
    y_shuf_test = amp_data[train_size + val_size :, -1]

    """Given just one row of inputs, we could fit a curve of amplitude against
    time through the 20 points, and extrapolate it one step into the future."""

    linear_fit = fit_linear(X_shuf_train[0])
    quartic_fit = fit_quartic(X_shuf_train[0])

    plot_amplitudes_against_time_points(
        X_shuf_train[0],
        y_shuf_train[0],
        Plot(fit=linear_fit, linspace=time_points, label="Linear fit", color="blue"),
        Plot(
            fit=quartic_fit, linspace=time_points, label="Quartic fit", color="orange"
        ),
    )

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
    especially when the data exhibits non-linear behaviour over a longer time frame."""

    v_linear, t_linear = make_vv(C=len(X_shuf_train[0]), K=2)
    v_quartic, t_quartic = make_vv(C=len(X_shuf_train[0]), K=5)

    plot_amplitudes_against_time_points(
        X_shuf_train[0],
        y_shuf_train[0],
        Plot(v_linear, t_linear, label="Linear fit", color="blue"),
        Plot(v_quartic, t_quartic, label="Quartic fit", color="orange"),
    )

    """The advantage of identifying the prediction as a linear combination, v.T @ x,
    is that we can compute v once and rapidly make next-step predictions for N
    different sequences of C amplitudes. We don't need to fit N separate models!"""

    print(
        f"{y_shuf_train[0] = } {X_shuf_train[0] @ v_linear = } {X_shuf_train[0] @ v_quartic = }"
    )

    best_C = None
    best_K = None
    smallest_error = float("inf")

    for C in range(2, 21):  # C from 2 to 20
        for K in range(2, 11):  # K from 2 to 10
            v, _ = make_vv(C=C, K=K)
            predictions = X_shuf_train[:, -C:] @ v
            error = mse(y_pred=predictions, y=y_shuf_train)

            if error < smallest_error:
                smallest_error = error
                best_C = C
                best_K = K

    print(
        f"Best context length (C): {best_C},",
        f"Best number of basis functions (K): {best_K},",
        f"Smallest square error: {smallest_error}",
    )

    """No matter which basis functions we pick, a linear model fitted by least squares
    will predict the next amplitude using a linear combination of the previous amplitudes.
    
    Given a large dataset, we can try to fit a good linear combination directly,
    without needing to specify basis functions. Using standard linear least squares fitting code,
    we can find the vector that minimizes sum_1^N (y^(n) - v^T x^(n))^2 on the training set"""

    best_C_train = None
    smallest_error_train = float("inf")
    
    for C in range(1, 21):  # C from 1 to 20
        v, _ = make_vv(C=C, K=2)
        predictions_train = X_shuf_train[:, -C:] @ v
        error_train = mse(y_pred=predictions_train, y=y_shuf_train)

        if error_train < smallest_error_train:
            smallest_error_train = error_train
            best_C_train = C

    print(f"Best context length (C) on training set: {best_C_train}, Smallest square error: {smallest_error_train}")

    best_C_val = -1
    smallest_error_val = float("inf")

    for C in range(1, 21):  # C from 1 to 20
        v, _ = make_vv(C=C, K=2)
        predictions_val = X_shuf_val[:, -C:] @ v
        error_val = mse(y_pred=predictions_val, y=y_shuf_val)

        if error_val < smallest_error_val:
            smallest_error_val = error_val
            best_C_val = C

    print(f"Best context length (C) on validation set: {best_C_val}, Smallest square error: {smallest_error_val}")
    import matplotlib.pyplot as plt

    v_best, _ = make_vv(C=best_C_val, K=2)
    predictions_best = X_shuf_val[:, -best_C_val:] @ v_best

    residuals = predictions_best - y_shuf_val

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Residuals on Best Model Validation Data')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    plt.subplot(1, 2, 2)
    plt.hist(amp_data, bins=30, edgecolor='black', alpha=0.7)
    plt.title("Histogram of Amplitudes from plot_line_graph_and_histogram")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
