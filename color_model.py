# color_model.py
import numpy as np


def convert_to_specific_colors(pixel):
    """
    Convert a given pixel to one of the two specific colors.
    """
    blue = (255, 0, 0, 128)  # Specific blue color in (a, r, g, b)
    yellow = (255, 255, 255, 0)  # Specific yellow color in (a, r, g, b)

    # Example condition to determine the color
    if sum(pixel) > (127.5 * 4):
        return yellow
    else:
        return blue


def update_model(generator, discriminator, combined, feedback):
    """
    Update the model based on user feedback.
    """
    # Update logic for the models
    # Example: feedback is either 1 (fun) or 0 (boring)
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)

    X = generated_image / 127.5 - 1
    y = np.array([feedback])

    discriminator.train_on_batch(X, y)
    noise = np.random.normal(0, 1, (32, 100))
    valid_y = np.ones((32, 1))

    combined.train_on_batch(noise, valid_y)

    generator.save('generator.h5')
    discriminator.save('discriminator.h5')
    combined.save('combined.h5')
