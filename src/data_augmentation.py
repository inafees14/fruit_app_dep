import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the image
image_path = "C:/Users/inafe/OneDrive/Desktop/Plants - Copy/watermelon/91.jpg"  # update if needed
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image = image.astype(np.float32) / 255.0  # Normalize pixel values

# Expand dimensions to match ImageDataGenerator input format
image = np.expand_dims(image, axis=0)


# Define individual augmentation generators
augmentations = {
    "Rotation (50°)": tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=50),
    "Width Shift (40%)": tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.4),
    "Height Shift (35%)": tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range=0.35),
    "Zoom (30%)": tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.3),
}


# Apply each augmentation separately
augmented_images = {
    label: gen.flow(image, batch_size=1).__next__()[0]
    for label, gen in augmentations.items()
}


# Plot original and augmented images
fig, axes = plt.subplots(1, 5, figsize=(20, 6))
axes[0].imshow(image[0])  # Original image
axes[0].set_title("Original Image")

# Plot augmented images with specific labels
for ax, (label, img) in zip(axes[1:], augmented_images.items()):
    ax.imshow(img)
    ax.set_title(label)

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
