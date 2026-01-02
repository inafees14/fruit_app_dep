import os
# import matplotlib.pyplot as plt # ❌ REMOVED - Not needed for the API
# from src.utils import load_trained_model, load_class_names, preprocess_image # ❌ REMOVED - We will define functions here
import numpy as np # ✅ ADDED
from PIL import Image # ✅ ADDED
#import tflite_runtime.interpreter as tflite
import tensorflow as tf

# --- Configuration (✅ CHANGED) ---
MODEL_PATH = os.path.join("checkpoints", "model.tflite") # ✅ Use the new .tflite model
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))
print("MODEL PATH:", os.path.abspath(MODEL_PATH))
DATA_DIR = "data" # This can be a relative path now
CLASS_NAMES_PATH = "class_names.txt" # ✅ We will use a simple text file

# --- Load Model and Class Names (✅ CHANGED) ---
try:
    #interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("✅ TFLite Model and class names loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    interpreter = None
    class_names = []

# --- Preprocessing Function (✅ ADDED) ---
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x, img

def predict_image(img_path, top_k=3):
    """Predict top-k classes for a single image using TFLite model."""
    if not interpreter:
        return [("Error", 0.0)], None

    # Use the new preprocessing function
    img_array, original_img = preprocess_image(img_path)

    # Run prediction with TFLite interpreter (✅ CHANGED)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    
    top_indices = preds.argsort()[-top_k:][::-1]
    return [(class_names[i], float(preds[i]) * 100) for i in top_indices], original_img

# Note: The 'predict_multiple_images' function below is for local testing ONLY.
# It requires 'matplotlib' which you should not include in your final requirements.txt for Heroku.
def predict_multiple_images(test_image_paths, save_path="predictions.png"):
    # This function remains the same but will now use the updated predict_image()
    pass # You can keep your original code for this function here for local testing.