import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


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

    eval_model(y_test, y_pred, "Simple LinAlg")
    return y_pred, coefficients[0], coefficients[1]

def model_poly(X_train, X_test, y_train, y_test):
    # y = w0 + w1 * x + w2 * x^2 + w3 * x^3

    bias_input_train = np.ones_like(X_train)
    X_train_extended = np.vstack([bias_input_train, X_train, X_train ** 2, X_train ** 3])
    w = np.linalg.lstsq(X_train_extended.T, y_train, rcond=None)
    coefficients = w[0]

    # y_pred2 = X_test_extended @ w
    bias_input_test = np.ones_like(X_test)
    X_test_extended = np.vstack([bias_input_test, X_test, X_test**2, X_test**3])
    y_pred = X_test_extended.T @ coefficients

    eval_model(y_test, y_pred, "Polynomial")

    return y_pred, coefficients[0], coefficients[1], coefficients[2], coefficients[3]
#--------------------------------------------------------------------------------------------------------
def eval_model(y_test, y_pred, model_name:str):
    print("MSE {}: {}".format(model_name, mean_squared_error(y_test, y_pred)))
    print("R2 {}: {}".format(model_name, r2_score(y_test, y_pred)))
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

    ax.plot(x_line, y_line, label="Regression line", color="red")
    ax.legend()
    return ax

def add_poly_line(ax, w, x_range=None):
    if x_range is None:
        x_range = ax.get_xlim()

    x_line = np.linspace(x_range[0], x_range[1], 200)
    y_line = sum(w[i] * x_line**i for i in range(len(w)))

    ax.plot(x_line, y_line, label="Polynomial model", color="green")
    ax.legend()

    return ax