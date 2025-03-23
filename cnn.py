import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from data_preprocessing import original_path, counterfeit_path
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Fix UnicodeEncodeError: 'charmap' codec can't encode character

# Ensure dataset folders exist
if not os.path.exists(original_path) or not os.path.exists(counterfeit_path):
    raise FileNotFoundError("❌ Dataset folders not found. Please check your paths.")

# Load dataset
X, y = [], []

# Load original QR code images
print("📥 Loading original QR code images...")
for filename in os.listdir(original_path):
    image_path = os.path.join(original_path, filename)
    try:
        img = load_img(image_path, color_mode="grayscale", target_size=(128, 128))
        X.append(img_to_array(img) / 255.0)  # Normalize pixel values
        y.append(0)  # Label for original
    except Exception as e:
        print(f"⚠️ Skipping {filename}: {e}")

# Load counterfeit QR code images
print("📥 Loading counterfeit QR code images...")
for filename in os.listdir(counterfeit_path):
    image_path = os.path.join(counterfeit_path, filename)
    try:
        img = load_img(image_path, color_mode="grayscale", target_size=(128, 128))
        X.append(img_to_array(img) / 255.0)  # Normalize pixel values
        y.append(1)  # Label for counterfeit
    except Exception as e:
        print(f"⚠️ Skipping {filename}: {e}")

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Ensure dataset is not empty
if len(X) == 0:
    raise ValueError("❌ No valid images were loaded. Check dataset folder structure.")

print(f"✅ Dataset loaded successfully with {len(X)} samples.")

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Define CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Apply Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator()  # No augmentation for validation

# Create data generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# Train Model
print("🔄 Training CNN model...")
model.fit(train_generator, 
          validation_data=val_generator, 
          epochs=5, 
          class_weight=class_weight_dict)

# Save Model in New Keras Format
model.save("cnn_model.keras")
print("✅ CNN model trained and saved successfully!")
