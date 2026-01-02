from src.predict import predict_image

igmg = r"D:\fruit_app_dep\data\apple\Apple (443).jpg"
# Use a real image path
result, _ = predict_image(igmg)
print(result)
