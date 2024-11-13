import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from PIL import Image

# Directories
image_dir = r"C:\Users\sarth\OneDrive\Desktop\split\Test\Defective"
label_dir = r"C:\Users\sarth\OneDrive\Desktop\split\Test\Defective_labels"
save_to_dir = r"C:\Users\sarth\OneDrive\Desktop\pro\augmented_defective"

# Ensure save directory exists
os.makedirs(save_to_dir, exist_ok=True)

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load an image and corresponding label
image_path = r"C:\Users\sarth\OneDrive\Desktop\split\Test\Defective\kos01_Part5.jpg"
label_path = r"C:\Users\sarth\OneDrive\Desktop\split\Test\Defective_labels\kos01_Part5_label.bmp"

img = load_img(image_path)  # Load image
label = load_img(label_path, color_mode='grayscale')  # Load label image (assumed grayscale)

# Convert to numpy arrays
x = img_to_array(img)
y = img_to_array(label)

# Reshape for ImageDataGenerator
x = x.reshape((1,) + x.shape)  # Image
y = y.reshape((1,) + y.shape)  # Label

# Augment images and labels
i = 0
for batch_img, batch_label in zip(datagen.flow(x, batch_size=1, save_to_dir=save_to_dir, save_prefix='aug', save_format='jpeg'),
                                  datagen.flow(y, batch_size=1, save_to_dir=save_to_dir, save_prefix='aug_label', save_format='bmp')):
    i += 1
    if i > 20:  # Generate 20 augmented images and labels
        break
