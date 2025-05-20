from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, InputLayer, Reshape, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
def prep_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.expand_dims(x_train, -1)  # kształt (n,28,28,1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, x_test

#AUTOKODERY-----------------------------------------------------------------------------
#DENSE
def build_encoder(bottleneck_dim=32):
    model = Sequential(name='encoder')
    model.add(InputLayer(input_shape=(28,28,1))) #input w formacie wymiarow zdjecia mnist
    model.add(Flatten()) #zamienia 28x28x1 na 784x1
    model.add(Dense(bottleneck_dim*4, activation='relu')) #stopniowe zmniejszanie liczby neuronow
    model.add(Dense(bottleneck_dim*2, activation='relu'))
    model.add(Dense(bottleneck_dim, activation='relu', name='bottleneck')) #ostatnia warstwa encodera, bottleneck
    return model

def build_decoder(bottleneck_dim=32):
    model = Sequential(name='decoder')
    model.add(InputLayer(input_shape=(bottleneck_dim,)))
    model.add(Dense(bottleneck_dim*2, activation='relu'))
    model.add(Dense(bottleneck_dim*4, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    model.add(Reshape((28,28,1)))
    return model

def build_autoencoder(encoder, decoder):
    input_img = Input(shape=(28,28,1), name='autoencoder_input')
    code = encoder(input_img)
    output = decoder(code)
    autoencoder = Model(inputs=input_img, outputs=output, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

#CONV
def create_conv_model(input_shape):
    model = models.Sequential(name='conv_encoder')
    model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(4, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Conv2DTranspose(8, (3, 3), padding='same', activation='relu'))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Conv2DTranspose(16, (3, 3), padding='same', activation='relu'))
    model.compile(optimizer='Adam', loss='binary_crossentropy')
    return model

def extract_conv_encoder(full_model):
    encoder = Sequential()
    for i in range(5):  #pierwsze 5 warstw
        encoder.add(full_model.layers[i])
    return encoder


# def visualize_reconstructions(original, dense_recon, conv_recon, n=10):
#     plt.figure(figsize=(20, 4))
#     for i in range(n):
#         # Original
#         ax = plt.subplot(3, n, i + 1)
#         plt.imshow(original[i].reshape(28, 28), cmap='gray')
#         plt.title("Original")
#         plt.axis("off")
#
#         # Dense Autoencoder reconstruction
#         ax = plt.subplot(3, n, i + n + 1)
#         plt.imshow(dense_recon[i].reshape(28, 28), cmap='gray')
#         plt.title("Dense Recon")
#         plt.axis("off")
#
#         # Conv Autoencoder reconstruction
#         ax = plt.subplot(3, n, i + 2 * n + 1)
#         plt.imshow(conv_recon[i].reshape(28, 28), cmap='gray')
#         plt.title("Conv Recon")
#         plt.axis("off")
#
#     plt.tight_layout()
#     plt.show()

#KNN---------------------------------------------------------------------------------------
def find_optimal_k(train_codes, test_codes, y_train, y_test, name, max_k=20):
    accuracies = []
    k_values = range(1, max_k + 1)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_codes, y_train)
        y_pred = knn.predict(test_codes)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"{name} - K={k}: Accuracy = {acc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.title(f'{name} - KNN Accuracy vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

    optimal_k = k_values[np.argmax(accuracies)] #np.argmax zwraca indeks najw. wartości w tablicy
    print(f"Optimal K for {name}: {optimal_k} with accuracy {max(accuracies):.4f}")
    return optimal_k, max(accuracies)

def evaluate_KNN(train_codes, test_codes, y_train, y_test, name, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_codes, y_train)

    y_pred = knn.predict(test_codes)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"{name} Confusion Matrix:")
    print(cm)

    #heatmapa confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {name}')
    plt.colorbar()

    classes = range(10)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations in the cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), #d - decimal integer
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return acc