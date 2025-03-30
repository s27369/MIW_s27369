import numpy as np
class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def initiate_weights_softmax(self, X, y):
        num_classes = len(set(y))
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(1 + X.shape[1], num_classes))


    def fit_softmax(self, X, y):
        self.initiate_weights_softmax(X, y)
        #self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            #cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            #self.cost_.append(cost)
        return self

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred) / y_true.shape[0]) #dzielenie by błąd nie rósł proporcjonalnie do liczby próbek

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def softmax(self, Z) -> float:
        max_value = max(Z, axis=1, keepdims=True)
        exp_z = np.exp(Z - max_value) #stabilizacja: zmienia np [8, 1003, 999] na [-992, 0, -4], by uproscic obliczenia
        return exp_z / np.sum(exp_z, axis=1, keepdims = True)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def fit_OvR(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        #self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            #cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            #self.cost_.append(cost)
        return self