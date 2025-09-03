# Print min, max, mean image sizes
print(f"Min Height: {image_heights.min()}, Max Height: {image_heights.max()}, Mean Height: {image_heights.mean():.2f}")
print(f"Min Width: {image_widths.min()}, Max Width: {image_widths.max()}, Mean Width: {image_widths.mean():.2f}")

# Show sample images from each class
import random
fig, axes = plt.subplots(2, 5, figsize=(16, 8))
for ax, cls in zip(axes.flatten(), random.sample(list(class_counts.keys()), min(10, len(class_counts)))):
    class_folder = os.path.join(dataset_path, cls)
    img_file = random.choice(os.listdir(class_folder))
    img_path = os.path.join(class_folder, img_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(cls, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()
