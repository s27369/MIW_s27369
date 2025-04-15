import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def get_data_plot(X, y, file_path) -> (plt.Figure, plt.Axes):
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="data")
    ax.set_title(f"{file_path.split("/")[-1]}")
    return fig, ax


def add_reg_line(ax: plt.Axes, w0, w1, x_range=None):
    if x_range is None:
        x_range = ax.get_xlim()

    x_line = np.linspace(x_range[0], x_range[1], 100)
    y_line = w0 + w1 * x_line

    ax.plot(x_line, y_line, label = "Regression line", color = "red")
    ax.legend()
    return ax



def model_simple_linalg(X_train, X_test, y_train, y_test):
    # y = w0 * 1 + w1 * x


    bias_input_train = np.ones_like(X_train)
    X_train_extended = np.vstack([bias_input_train, X_train])

    # X_train_extended @ w = y_train
    w = np.linalg.lstsq(X_train_extended.T, y_train, rcond=None)  # w = [w0, w1]
    coefficients = w[0]

    bias_input_test = np.ones_like(X_test)
    X_test_extended = np.vstack([bias_input_test, X_test])
    y_pred = X_test_extended.T @ coefficients

    print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
    print("R2: {}".format(r2_score(y_test, y_pred)))

    return y_pred, coefficients[0], coefficients[1]
