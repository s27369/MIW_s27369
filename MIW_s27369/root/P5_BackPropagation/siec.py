import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["PYCHARM_MATPLOTLIB_BACKEND"] = "off"
if __name__=="__main__":
    P = np.arange(-2, 2.1, 0.1)
    T = P**2 + (np.random.rand(len(P)) - 0.5)

    S1 = 100
    W1 = np.random.rand(S1, 1) - 0.5
    B1 = np.random.rand(S1, 1) - 0.5
    W2 = np.random.rand(1, S1) - 0.5
    B2 = np.random.rand(1, 1) - 0.5
    lr = 0.01

    for epoka in range(1, 301):
        A1 = np.tanh(W1 @ P.reshape(1, -1) + B1 @ np.ones((1, len(P))))
        A2 = W2 @ A1 + B2

        E2 = T - A2.flatten()
        E1 = W2.T @ E2.reshape(1, -1)

        dW2 = lr * E2.reshape(1, -1) @ A1.T
        dB2 = lr * E2 @ np.ones((len(E2), 1))
        dW1 = lr * ((1 - A1**2) * E1) @ P.reshape(1, -1).T
        dB1 = lr * ((1 - A1**2) * E1) @ np.ones((len(P), 1))

        W2 += dW2
        B2 += dB2
        W1 += dW1
        B1 += dB1

        if epoka % 1 == 0:
            # plt.clf()
            plt.plot(P, T, 'r*', label='Target')
            plt.plot(P, A2.flatten(), label='Output')
            plt.legend()
            plt.pause(0.01)

    plt.show()
