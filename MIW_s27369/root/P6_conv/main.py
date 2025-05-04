import matplotlib.pyplot as plt
# import tensorflow as tf
# print(tf.__version__)
import matplotlib
matplotlib.use('TkAgg')


from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np


(x_all, y_all), (x_all2, y_all2) = cifar10.load_data()
x = np.concatenate([x_all, x_all2])
y = np.concatenate([y_all, y_all2])

animals = [2, 3, 4, 5, 6, 7]
y = np.where(np.isin(y, animals), 0, 1)
# 0 -> zwierzęta
# 1 -> pojazdy

split = int(len(x) * 0.3)
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

def build_model(conv_layers):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    if conv_layers >= 2:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    if conv_layers == 3:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    return model

accuracies = []

for layers_num in [1, 2, 3]:
    model = build_model(layers_num)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'{layers_num} warstwy splotowe – test_acc = {test_acc:.4f}')
    accuracies.append(test_acc)

# Wykres porównawczy
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], accuracies, marker='o', linestyle='-', color='blue')
plt.title("Porównanie dokładności modeli")
plt.xlabel("Liczba warstw splotowych")
plt.ylabel("Dokładność na zbiorze testowym")
plt.xticks([1, 2, 3])
plt.ylim(0, 1)
plt.grid(True)
plt.show()
