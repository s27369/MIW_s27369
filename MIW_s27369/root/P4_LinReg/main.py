import os

import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split
from functions import *

matplotlib.use('Agg')
data_path = r"./data"
plot_path = r"./plots"
if __name__ == "__main__":
    data_files = os.listdir(data_path)
    data_files = [r"{}/{}".format(data_path, x) for x in data_files]


    for file_path in data_files:
        print(file_path)
        data = np.loadtxt(file_path)
        X = data[:, 0]
        y = data[:, 1]
        fig, ax = get_data_plot(X, y, file_path)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

        y_pred_simple, b_simple, a_simple = model_simple_linalg(X_train, X_test, y_train, y_test)
        ax = add_reg_line(ax, b_simple, a_simple)

        y_pred_poly, d_poly, c_poly, b_poly, a_poly = model_poly(X_train, X_test, y_train, y_test)
        w = [d_poly, c_poly, b_poly, a_poly]
        ax = add_poly_line(ax, w)

        plt.savefig(f"{plot_path}/plot_{file_path.split("/")[-1].replace(".txt", "")}.jpg")
        fig.show()
