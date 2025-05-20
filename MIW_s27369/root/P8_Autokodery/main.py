from functions import *

if __name__=="__main__":
    #ładowanie danych
    x_train, x_test = prep_data()
    x_all = np.concatenate([x_train, x_test], axis=0)
    print(x_train.shape, x_test.shape, x_all.shape)
    #oryginalne labele
    (_, y_train), (_, y_test) = mnist.load_data()
    y_all = np.concatenate([y_train, y_test], axis=0)

    #dense autoencoder-----------------------------------------------------------------------------
    dense_encoder = build_encoder(bottleneck_dim=32)
    dense_decoder = build_decoder(bottleneck_dim=32)

    dense_autoencoder = build_autoencoder(dense_encoder, dense_decoder)
    dense_history = dense_autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, validation_data=(x_test, x_test), verbose=1)
    print(dense_autoencoder.summary())

    dense_codes = dense_encoder.predict(x_all)
    print(f"Dense_codes: {dense_codes.shape}")

    #conv autoencoder-----------------------------------------------------------------------------
    conv_autoencoder = create_conv_model(input_shape=(28, 28, 1))
    conv_history = conv_autoencoder.fit(
        x_train, x_train,
        epochs=20,
        batch_size=64,
        validation_data=(x_test, x_test),
        verbose=1
    )

    #wyciągnięcie samego kodera
    conv_encoder = extract_conv_encoder(conv_autoencoder)
    print(f"Total layers in conv_autoencoder: {len(conv_autoencoder.layers)}")

    temp_output = conv_encoder.predict(x_all)
    print("Conv encoder output shape:", temp_output.shape)

    #spłaszczenie conv feature maps by móc porównać z dense_codes
    conv_codes = temp_output.reshape(temp_output.shape[0], -1)
    print(f"Conv_codes: {conv_codes.shape}")

    #KNN-----------------------------------------------------------------------------
    train_size = len(x_train)

    dense_train_codes = dense_codes[:train_size]
    dense_test_codes = dense_codes[train_size:]

    conv_train_codes = conv_codes[:train_size]
    conv_test_codes = conv_codes[train_size:]

    optimal_k_dense, acc_KNN_dense = find_optimal_k(dense_train_codes, dense_test_codes, y_train, y_test, "Dense autocoder")
    optimal_k_conv, acc_KNN_conv= find_optimal_k(conv_train_codes, conv_test_codes, y_train, y_test, "Convolutional autocoder")

    evaluate_KNN(dense_train_codes, dense_test_codes, y_train, y_test, "Dense autocoder", n_neighbors=optimal_k_dense)
    evaluate_KNN(conv_train_codes, conv_test_codes, y_train, y_test, "Convolutional autocoder", n_neighbors=optimal_k_conv)

