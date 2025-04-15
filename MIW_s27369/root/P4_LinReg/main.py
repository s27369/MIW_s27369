import os
import numpy as np
from sklearn.model_selection import train_test_split
from functions import *
data_path = r"./data"
if __name__ == "__main__":
    data_files = os.listdir(data_path)
    data_files = [r"{}/{}".format(data_path, x) for x in data_files]


    for file_path in data_files[:1]:
        data = np.loadtxt(file_path)
        X = data[:, 0]
        y = data[:, 1]
        fig, ax = get_data_plot(X, y, file_path)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

        y_pred_simple, b_simple, a_simple = model_simple_linalg(X_train, X_test, y_train, y_test)
        ax = add_reg_line(ax, b_simple, a_simple)
        fig.show()

        # print(X_train_extended)
