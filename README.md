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

