import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
input_folder = r"C:\Users\sarth\OneDrive\Desktop\pro"
output_folder = r"C:\Users\sarth\OneDrive\Desktop\split"

# Categories
categories = ['Defective', 'Non-Defective']

# Create output directories
for split in ['Train', 'Validation', 'Test']:
    for category in categories:
        os.makedirs(os.path.join(output_folder, split, category), exist_ok=True)

# Split and copy files
for category in categories:
    category_path = os.path.join(input_folder, category)
    files = os.listdir(category_path)

    train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

    # Copy files
    for file_set, split in zip([train_files, val_files, test_files], ['Train', 'Validation', 'Test']):
        for file in file_set:
            shutil.copy(os.path.join(category_path, file), os.path.join(output_folder, split, category))

print("Dataset split complete.")
