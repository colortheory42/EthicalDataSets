import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('RGBA')
        img = img.resize((420, 420))
        img = np.array(img)
        if img is not None:
            images.append(img)
            labels.append(0 if 'boring' in folder else 1)
    return images, labels

def build_discriminator(img_shape):
    input_img = Input(shape=img_shape)
    x = Conv2D(32, kernel_size=3, strides=2, padding="same")(input_img)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_img, output)
    return model

def build_generator(noise_shape):
    input_noise = Input(shape=noise_shape)
    x = Dense(128 * 105 * 105, activation="relu")(input_noise)
    x = Reshape((105, 105, 128))(x)
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Conv2D(4, kernel_size=3, padding="same", activation='tanh')(x)
    model = Model(input_noise, output)
    return model

def main():
    # Load dataset
    boring_images, boring_labels = load_images_from_folder('dataset/boring')
    structured_images, structured_labels = load_images_from_folder('dataset/structured')

    # Combine and split dataset
    images = np.array(boring_images + structured_images)
    labels = np.array(boring_labels + structured_labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    img_shape = (420, 420, 4)
    noise_shape = (100,)

    optimizer = Adam(0.0002, 0.5)

    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    generator = build_generator(noise_shape)

    z = Input(shape=noise_shape)
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Pre-train the discriminator
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    discriminator.fit(X_combined, y_combined, epochs=10, batch_size=32)

    # Save the models
    generator.save('generator.h5')
    discriminator.save('discriminator.h5')
    combined.save('combined.h5')

if __name__ == "__main__":
    main()
