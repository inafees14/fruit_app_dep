# src/api.py
from fastapi import FastAPI, UploadFile, File
import shutil
import os
from .predict import predict_image
from fastapi.templating import Jinja2Templates  # ✅ add this import
from fastapi.responses import HTMLResponse
from fastapi.requests import Request

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serves the new beautiful homepage."""
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    predictions, _ = predict_image(file_path)   # ignore img
    os.remove(file_path)  # Delete file after prediction

    return {
        "filename": file.filename,
        "predictions": [
            {"class": cls, "probability": prob} for cls, prob in predictions
        ]
    }

# -----------------------
# Serve frontend (index.html)
# -----------------------
templates = Jinja2Templates(directory="templates")

@app.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})