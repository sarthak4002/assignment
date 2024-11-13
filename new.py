import os
from PIL import Image
import numpy as np

# Update these paths with your actual dataset folder and desired output folder
main_folder = r"C:\Users\sarth\Downloads\KolektorSDD"  # Path to your dataset folder
output_folder = r"C:\Users\sarth\OneDrive\Desktop\pro"  # Output folder for processed data

# Create output directories for categories: Defective and Non-Defective
categories = ['Defective', 'Non-Defective']  # Categories for output
os.makedirs(output_folder, exist_ok=True)
for category in categories:
    os.makedirs(os.path.join(output_folder, category), exist_ok=True)

# Process images and categorize them
for item_folder in os.listdir(main_folder):
    item_path = os.path.join(main_folder, item_folder)
    if os.path.isdir(item_path):  # Ensure it's a folder
        for file in os.listdir(item_path):
            if file.endswith(".jpg") and not file.endswith("_label.bmp"):  # Only process main images
                # File paths
                image_path = os.path.join(item_path, file)
                label_path = os.path.join(item_path, file.replace(".jpg", "_label.bmp"))

                # Check if label exists
                if not os.path.exists(label_path):
                    print(f"Label missing for {image_path}, skipping...")
                    continue
                else:
                    print(f"Processing image: {image_path} with label: {label_path}")

                # Load label and check for defects
                label_image = Image.open(label_path)
                label_array = np.array(label_image)

                # Check for white pixels in the label to classify as defective
                is_defective = np.any(label_array > 0)  # Check for white pixels in the label
                category = 'Defective' if is_defective else 'Non-Defective'

                # Make filenames unique by adding the folder name as a prefix
                unique_filename = f"{item_folder}_{file}"  # E.g., kos1_part0.jpg
                unique_label_filename = unique_filename.replace(".jpg", "_label.bmp")

                # Resize image and save in the appropriate folder
                resized_image = Image.open(image_path).resize((512, 1408))  # Resize to 512x1408 px
                output_image_path = os.path.join(output_folder, category, unique_filename)
                resized_image.save(output_image_path)

                # Resize label and save (if needed for further processing)
                resized_label = label_image.resize((512, 1408))  # Resize label to match image
                output_label_path = os.path.join(output_folder, category, unique_label_filename)
                resized_label.save(output_label_path)

print("Processing complete. All images have been categorized and resized.")
