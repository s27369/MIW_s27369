import numpy as np

class MyLogRegSoftmax:
    def __init__(self, eta=0.05, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.cost_ = []

    def initiate_weights(self, X, y):
        num_classes = len(np.unique(y))
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(X.shape[1] + 1, num_classes))

    def softmax(self, Z):
        max_value = np.max(Z, axis=1, keepdims=True)  #stabilizacja: zmienia np [1000, 1003, 999] na [-3, 0, -4], by uproscic obliczenia
        exp_z = np.exp(Z - max_value)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))  #zapewnienie dodatniości

    def fit(self, X, y):
        self.initiate_weights(X, y)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        y_one_hot = np.eye(len(np.unique(y)))[y]

        for _ in range(self.n_iter):
            net_input = X_bias.dot(self.w_)
            output = self.softmax(net_input)
            errors = y_one_hot - output

            self.w_ += self.eta * X_bias.T.dot(errors) / X.shape[0]
            cost = self.cross_entropy_loss(y_one_hot, output)
            self.cost_.append(cost)

        return self

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        net_input = X_bias.dot(self.w_)
        return np.argmax(self.softmax(net_input), axis=1)