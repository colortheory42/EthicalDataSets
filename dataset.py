import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os

# Create directories for the images
os.makedirs('dataset/boring', exist_ok=True)
os.makedirs('dataset/structured', exist_ok=True)

# Colors
yellow = (255, 255, 0, 255)
blue = (0, 0, 128, 255)


# Function to generate random noise images with only yellow and blue pixels
def create_random_noise_image(width, height, path):
    noise = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            noise[y, x] = random.choice([yellow, blue])
    image = Image.fromarray(noise, 'RGBA')
    image.save(path)


# Function to generate structured images with math equations
def create_structured_image(width, height, background_color, text_color, text, path):
    image = Image.new("RGBA", (width, height), background_color)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((width - text_width) // 2, (height - text_height) // 2)

    draw.text(position, text, fill=text_color, font=font)
    image.save(path)


# Generate random noise images
for i in range(50):  # Generate 50 random noise images
    path = f'dataset/boring/noise_{i}.png'
    create_random_noise_image(420, 420, path)

# Generate structured images with math equations
equations = ["{} + {}".format(random.randint(1, 100000), random.randint(0, 100)) for _ in range(50)]
for i, equation in enumerate(equations):
    # Yellow background with blue text
    path = f'dataset/structured/yellow_blue_{i}.png'
    create_structured_image(420, 420, yellow, blue, equation, path)

    # Blue background with yellow text
    path = f'dataset/structured/blue_yellow_{i}.png'
    create_structured_image(420, 420, blue, yellow, equation, path)
