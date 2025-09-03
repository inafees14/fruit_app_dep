import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf


def load_trained_model(model_path):
    """Load a trained Keras model from file."""
    return tf.keras.models.load_model(model_path)


def load_class_names(data_dir):
    """Load sorted class names from training directory."""
    return sorted(os.listdir(data_dir))


def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess an image for prediction with MobileNetV2."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array), img


def plot_training_curves(history_path, save_path="training_curves.png"):
    """Plot training and validation accuracy/loss curves."""
    try:
        with open(history_path, "rb") as file:
            history = pickle.load(file)

        plt.figure(figsize=(12, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history["accuracy"], label="Training Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history["loss"], label="Training Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    except FileNotFoundError:
        print(f"History file not found at {history_path}")
