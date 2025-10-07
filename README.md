# 🍎 Fruit Classification AI

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-blue?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Heroku](https://img.shields.io/badge/Heroku-Deployed-purple?logo=heroku)](https://www.heroku.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a complete deep learning application that classifies images of various fruits. It features a lightweight **MobileNetV2** model deployed as a web service using **FastAPI** and hosted on **Heroku**.

## 🚀 Live Demo

The application is live and ready to use! Click the link below to try it out.

**[➡️ Launch Fruit Classifier App](https://fruit-classification-1-7c2a30615392.herokuapp.com/)**

## 📸 Application Screenshots

<table>
  <tr>
    <td align="center"><b>Homepage</b></td>
    <td align="center"><b>Image Upload</b></td>
    <td align="center"><b>Successful Prediction</b></td>
     <td align="center"><b>Unknown Image</b></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

## ✨ Features

-   **11 Fruit Classes**: Accurately classifies a wide variety of common fruits.
-   **High Accuracy**: Built on the MobileNetV2 architecture for efficient and precise predictions.
-   **Interactive Fun Facts**: Displays a random fun fact for each correctly identified fruit.
-   **Handles Unknown Images**: Implements a confidence threshold to gracefully handle images that are not fruits.
-   **Responsive UI**: Clean and simple user interface that works on desktop and mobile.
-   **RESTful API**: Backend powered by FastAPI, serving model predictions.

## 🛠️ Technology Stack

-   **Model**: TensorFlow / Keras (MobileNetV2)
-   **Backend**: FastAPI, Uvicorn
-   **Frontend**: HTML5, CSS3, Vanilla JavaScript
-   **Deployment**: Heroku, Git

## 🎯 Project Motivation

The primary goal is to develop an accurate and lightweight fruit classification system that can run efficiently. MobileNetV2 was specifically chosen as the backbone architecture because of its efficiency on resource-constrained devices, small size, and suitability for transfer learning. This has several practical applications in mobile apps, agricultural tech, and educational tools.

## 📂 Web App Project Structure
```bash
fruit_app_dep/
├── .git/
├── checkpoints/
├── data/
├── epochs/
├── notebooks/
│   └── EDA_and_Augmentation.ipynb
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data_analysis.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── templates/
│   ├── facts.json
│   ├── home.html
│   └── index.html
├── tests/
│   └── test_utils.py
├── uploads/
├── .gitignore
├── .python-version
├── .slugignore
├── Procfile
├── README.md
├── class_names.py
├── class_names.txt
├── convert_model.py
├── predictions.png
├── requirements.txt
```
## 📜 Version History (Heroku Releases)

This table highlights the key milestones of the deployment.

| Version | Date       | Key Changes (Commit Message)                         |
|---------|------------|------------------------------------------------------|
| `v13`   | 2025-10-07 | Fix: Case-sensitive bug for fun facts lookup         |
| `v12`   | 2025-10-07 | Feat: Add probability threshold for unknown images   |
| `v11`   | 2025-10-07 | Feat: Implement fun facts for each fruit class       |
| `v8`    | 2025-10-02 | Feat: Deploy two-page UI (Home + Predict)            |
| `v1`    | 2025-10-01 | Initial deployment of Fruit Classifier app           |

## 🚀 Local Development Setup

To run this web application on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd fruit-classification-app
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the FastAPI server:**
    ```bash
    uvicorn src.api:app --reload
    ```
4.  Open your browser and navigate to `http://127.0.0.1:8000`.

## 🧠 Model Training Details

The model was trained separately. The training repository contains scripts and data used to generate the final `model.h5` file.

-   **Dataset**: ~31,000 images across 11 fruit categories, sourced from Kaggle and free stock image websites.
-   **Training Strategy**: Transfer learning with a frozen MobileNetV2 base, a custom classification head, and data augmentation.

## 📈 Future Improvements

-   Implement model quantization for further size reduction.
-   Add real-time classification via webcam.
-   Expand the dataset to include more fruit varieties.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.




