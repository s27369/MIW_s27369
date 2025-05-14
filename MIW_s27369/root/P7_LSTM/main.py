import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

data_path = r"./rhm_de_d.csv"

def create_sliding_window(data, window_size):
    Xy = []
    for i in range(len(data) - window_size):
        window = data[i:i + window_size + 1]
        Xy.append(window)
    return np.array(Xy)

def prep_data():
    data = pd.read_csv(data_path)
    return data['Close'].dropna().values

def autoregressive_model(X, y):
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])  # kolumna jedynek
    # y = w1·x1 + w2·x2 + w3·x3 + bias
    w = np.linalg.pinv(X_bias) @ y  # wagi
    print(f"Wagi AR: {w}")
    y_pred = X_bias @ w
    AR_mse = mean_squared_error(y, y_pred)
    print(f"AR MSE: {AR_mse}")
    return w

if __name__=="__main__":
    raw_data = prep_data()
    plt.figure(figsize=(18, 4))
    window_sizes = [3, 8, 15]
    for window_size in window_sizes:
        print(("-"*20)+f"window_size={window_size}"+("-"*20))
        data = create_sliding_window(raw_data, window_size)

        X = data[:, :-1]
        y = data[:, -1]

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        #Linear autoregressive model (AR)
        w = autoregressive_model(X, y)

        #LSTM
        #skalowanie
        # scaler_X = MinMaxScaler(feature_range=(0, 1))
        # scaler_y = MinMaxScaler(feature_range=(0, 1))
        #
        # X_train_scaled = scaler_X.fit_transform(X_train)
        # y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        #
        # X_test_scaled = scaler_X.transform(X_test)
        # y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
        # X

        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        scaler_X.fit(X_train)
        X_train_scaled = scaler_X.transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # y
        scaler_y.fit(y_train.reshape(-1, 1))
        y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))  # ONLY transform

        #reshape (samples, timesteps, features)
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train_scaled.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=200,
            batch_size=10,
            validation_data=(X_test_scaled, y_test_scaled),
            verbose=1
        )

        #predykcje
        y_pred_scaled = model.predict(X_test_scaled)

        #AR predictions for test set
        X_test_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
        y_pred_ar = X_test_bias @ w

        #unscaled LSTM
        y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled)

        #MSE
        lstm_mse = mean_squared_error(y_test.reshape(-1, 1), y_pred_unscaled)
        ar_mse = mean_squared_error(y_test, y_pred_ar)
        print("XXXX")
        print(np.max(y_test_scaled))
        print("XxXxXx")
        print(f"AR MSE: {ar_mse}")
        print(f"LSTM MSE: {lstm_mse}")

        # #loss plot
        # plt.figure(figsize=(8, 4))
        # plt.plot(history.history['loss'], label='Training Loss')
        # plt.plot(history.history['val_loss'], label='Validation Loss')
        # plt.title('Loss During Training')
        # plt.xlabel('Epoch')
        # plt.ylabel('MSE Loss')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # plotting
        plt.subplot(1, 3, window_sizes.index(window_size) + 1)
        plt.plot(y_test, label='True', color='black')
        plt.plot(y_pred_ar, label='AR', color='green')
        plt.plot(y_pred_unscaled, label='LSTM', color='red')
        plt.title(f'window_size={window_size}')
        plt.xlabel('Time')
        plt.ylabel('Closing Price')
        plt.text(1, 1, f"AR MSE: {ar_mse:.4f}\nLSTM MSE: {lstm_mse:.4f}")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()