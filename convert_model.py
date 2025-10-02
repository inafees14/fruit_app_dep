import tensorflow as tf

# Define the path to your current Keras model
keras_model_path = "checkpoints/final_model.h5"

# Define where to save the new TFLite model
tflite_model_path = "checkpoints/model.tflite"

# Load the Keras model
model = tf.keras.models.load_model(keras_model_path)

# Create a TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Perform the conversion
tflite_model = converter.convert()

# Save the new TFLite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Successfully converted {keras_model_path} to {tflite_model_path}")