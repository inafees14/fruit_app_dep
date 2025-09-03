import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils import load_trained_model, load_class_names, preprocess_image

# Paths
DATA_DIR = "E:/Desktop/Plants - Copy"
MODEL_PATH = os.path.join("checkpoints", "final_model.h5")

# Load model
model = load_trained_model(MODEL_PATH)

# Data generator for validation set
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

# True labels
y_true = validation_generator.classes

# Predictions
preds = model.predict(validation_generator)
y_pred = np.argmax(preds, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=validation_generator.class_indices.keys(),
    yticklabels=validation_generator.class_indices.keys(),
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("checkpoints/confusion_matrix.png")
plt.show()

# Classification Report
report = classification_report(
    y_true, y_pred, target_names=validation_generator.class_indices.keys()
)
print(report)
with open("checkpoints/classification_report.txt", "w") as f:
    f.write(report)
