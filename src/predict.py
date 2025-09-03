import os
import matplotlib.pyplot as plt
from src.utils import load_trained_model, load_class_names, preprocess_image

# Paths
MODEL_PATH = os.path.join("checkpoints", "final_model.h5")
DATA_DIR = "E:/Desktop/Plants - Copy"  # update with your dataset path

# Load model and class names
model = load_trained_model(MODEL_PATH)
class_names = load_class_names(DATA_DIR)


def predict_image(img_path, top_k=3):
    """Predict top-k classes for a single image."""
    img_array, img = preprocess_image(img_path)
    preds = model.predict(img_array)
    top_indices = preds[0].argsort()[-top_k:][::-1]
    return [(class_names[i], float(preds[0][i]) * 100) for i in top_indices], img


def predict_multiple_images(test_image_paths, save_path="predictions.png"):
    """Predict and plot results for multiple images."""
    plt.figure(figsize=(12, 12))

    for i, img_path in enumerate(test_image_paths):
        if i >= 9:  # show only first 9
            break
        predictions, img = predict_image(img_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        title = f"{predictions[0][0]}: {predictions[0][1]:.2f}%"
        plt.title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    # Print results
    for i, img_path in enumerate(test_image_paths):
        if i >= 9:
            break
        predictions, _ = predict_image(img_path)
        print(f"\nPredictions for {os.path.basename(img_path)}:")
        for cls, prob in predictions:
            print(f"  {cls}: {prob:.2f}%")


if __name__ == "__main__":
    test_images = [
        "D:/Download/Apple_p.jpg",
        "D:/Download/Banana_p.jpg",
        "D:/Download/Strawberry_p.jpg",
        "D:/Download/Mango_p.jpg",
        "E:/images/images/2610.jpg",
    ]
    predict_multiple_images(test_images)

predict_image(r'D:\fruit_app_dep\data\banana\131.jpg')