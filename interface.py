import numpy as np
import pygame
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Import functions from color_model.py
from color_model import convert_to_specific_colors, update_model

def generate_image(generator):
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)[0]

    # Use the specific color conversion method from color_model.py
    new_image = np.zeros((generated_image.shape[0], generated_image.shape[1], 4), dtype=np.uint8)

    for y in range(generated_image.shape[0]):
        for x in range(generated_image.shape[1]):
            a, r, g, b = convert_to_specific_colors(generated_image[y, x])
            new_image[y, x] = [r, g, b, a]  # Convert to RGBA for Pygame

    print("Generated new image.")
    return new_image

def display_image(image_array, screen):
    try:
        # Convert the numpy array to a string buffer for Pygame
        image_array_str = image_array.tobytes()

        # Create a Pygame surface from the string buffer
        py_image = pygame.image.frombuffer(image_array_str, image_array.shape[1::-1], "RGBA")
        screen.fill((0, 0, 0))
        screen.blit(py_image, (0, 0))
        pygame.display.flip()

        # Debug: Print part of the image array
        print(f"Image array sample:\n{image_array[0, 0:5]}")

    except Exception as e:
        print(f"Error displaying image: {e}")

def user_feedback_loop():
    # Initialize Pygame
    pygame.init()

    # Set up display
    screen_width = 420
    screen_height = 420
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Image Feedback')

    # Load the models
    generator = load_model('generator.h5')
    discriminator = load_model('discriminator.h5')
    combined = load_model('combined.h5')

    # Compile the models
    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    running = True
    new_image_needed = True
    while running:
        if new_image_needed:
            image_array = generate_image(generator)
            display_image(image_array, screen)
            new_image_needed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    update_model(generator, discriminator, combined, 1)
                    new_image_needed = True
                elif event.key == pygame.K_n:
                    update_model(generator, discriminator, combined, 0)
                    new_image_needed = True
                elif event.key == pygame.K_q:
                    running = False

    pygame.quit()

if __name__ == "__main__":
    user_feedback_loop()
