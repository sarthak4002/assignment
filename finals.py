import os
import numpy as np
from skimage.feature import hog
from skimage import exposure
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paths to your dataset
train_dir = r"C:\Users\sarth\OneDrive\Desktop\split\Train"
val_dir = r"C:\Users\sarth\OneDrive\Desktop\split\Validation"
test_dir = r"C:\Users\sarth\OneDrive\Desktop\split\Test"

# Categories
categories = ['Defective', 'Non-Defective']

# Initialize lists for features and labels
features = []
labels = []

# Function to extract HOG features from an image
def extract_hog_features(image_path):
    image = imread(image_path, as_gray=True)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd

# Load images and labels from directories
for category in categories:
    category_path = os.path.join(train_dir, category)
    print(f"Processing category: {category}, Path: {category_path}")
    for file in os.listdir(category_path):
        if file.endswith(".jpg"):
            image_path = os.path.join(category_path, file)
            # Extract HOG features
            feature = extract_hog_features(image_path)
            features.append(feature)
            labels.append(0 if category == 'Non-Defective' else 1)  # 0 = Non-Defective, 1 = Defective
            print(f"Processed image: {file}, Label: {category}")

# Check if features and labels are being collected
print(f"Number of features: {len(features)}")
print(f"Shape of feature vector for one sample: {features[0].shape}")

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Training Complete!")

# Evaluate the model on the validation set
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Evaluate on the test set
test_features = []
test_labels = []

# Load test images and labels
for category in categories:
    category_path = os.path.join(test_dir, category)
    for file in os.listdir(category_path):
        if file.endswith(".jpg"):
            image_path = os.path.join(category_path, file)
            feature = extract_hog_features(image_path)
            test_features.append(feature)
            test_labels.append(0 if category == 'Non-Defective' else 1)

test_features = np.array(test_features)
test_labels = np.array(test_labels)

# Make predictions on the test set
test_pred = model.predict(test_features)
print("Test Accuracy:", accuracy_score(test_labels, test_pred))
print("Test Classification Report:\n", classification_report(test_labels, test_pred))

# Save the model
joblib.dump(model, r"C:\Users\sarth\OneDrive\Desktop\defect_classification_model_rf.joblib")
