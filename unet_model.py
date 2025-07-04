from tensorflow.keras import layers, models

def unet(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)

    # Decoder
    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(b)
    u2 = layers.concatenate([u2, c2])
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    u1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c3)
    u1 = layers.concatenate([u1, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    return models.Model(inputs, outputs)