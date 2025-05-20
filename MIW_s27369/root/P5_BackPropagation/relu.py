import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    import os
    os.environ["PYCHARM_MATPLOTLIB_BACKEND"] = "off"
    P = np.arange(-4, 4.1, 0.1)
    T = P**2 + (np.random.rand(len(P)) - 0.5)

    S1 = 2
    W1 = np.random.rand(S1, 1) - 0.5
    B1 = np.random.rand(S1, 1) - 0.5
    W2 = np.random.rand(1, S1) - 0.5
    B2 = np.random.rand(1, 1) - 0.5
    lr = 0.001

    for epoka in range(1, 201):
        X = W1 @ P.reshape(1, -1) + B1
        A1 = np.maximum(X, 0)  # ReLU
        A2 = W2 @ A1 + B2

        E2 = T - A2.flatten()
        E1 = W2.T @ E2.reshape(1, -1)

        dW2 = lr * (E2.reshape(1, -1) @ A1.T)
        dB2 = lr * E2.sum()
        dW1 = lr * ((X > 0).astype(float) * E1) @ P.reshape(1, -1).T
        dB1 = lr * ((X > 0).astype(float) * E1).sum(axis=1, keepdims=True)

        W2 += dW2
        B2 += dB2
        W1 += dW1
        B1 += dB1

        if epoka % 5 == 0:
            # plt.clf()
            plt.plot(P, T, 'r*', label='Target')
            plt.plot(P, A2.flatten(), label='Output')
            plt.legend()
            plt.pause(0.2)

    plt.show()
