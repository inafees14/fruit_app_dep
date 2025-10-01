import os
import shutil
import random

# Source dataset
src_dir = "E:/Desktop/Plants - Copy"

# Destination dataset
dst_dir = "D:/fruit_app_dep/data"
os.makedirs(dst_dir, exist_ok=True)

# Number of images per class
n_images = 10

# Loop through each class folder
for class_name in os.listdir(src_dir):
    class_src = os.path.join(src_dir, class_name)
    class_dst = os.path.join(dst_dir, class_name)
    
    if os.path.isdir(class_src):
        os.makedirs(class_dst, exist_ok=True)
        
        # Get all images in the class
        images = [f for f in os.listdir(class_src) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Randomly select n_images (or fewer if not enough)
        selected_images = random.sample(images, min(n_images, len(images)))
        
        # Copy files
        for img in selected_images:
            src_path = os.path.join(class_src, img)
            dst_path = os.path.join(class_dst, img)
            shutil.copy(src_path, dst_path)
