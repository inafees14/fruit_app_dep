# src/api.py
import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
import shutil
from .predict import predict_image
# The 'drive_utils' import has been removed

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/facts", response_class=FileResponse)
async def get_facts():
    return "templates/facts.json"

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 1. Save uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Run prediction
    predictions, _ = predict_image(file_path)

    # 3. Determine the final prediction
    CONFIDENCE_THRESHOLD = 60.0
    final_predictions = []
    
    if predictions:
        top_class, top_prob = predictions[0]
        if top_prob < CONFIDENCE_THRESHOLD:
            final_predictions.append({"class": "Not a Fruit!", "probability": top_prob})
        else:
            final_predictions.append({"class": top_class, "probability": top_prob})

    # 4. Clean up the local file
    os.remove(file_path)

    # 5. Return the JSON response to the user
    return {
        "filename": file.filename,
        "predictions": final_predictions
    }