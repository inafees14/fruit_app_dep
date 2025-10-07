# src/api.py
import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
import shutil
from .predict import predict_image
from . import drive_utils

# Configure logging
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

    # 3. Determine the final prediction and folder name
    CONFIDENCE_THRESHOLD = 70.0
    final_predictions = []
    predicted_class_for_upload = "Unknown"  # Default folder name

    if predictions:
        top_class, top_prob = predictions[0]
        if top_prob < CONFIDENCE_THRESHOLD:
            final_predictions.append({"class": "Not a Fruit!", "probability": top_prob})
            # The folder will remain "Unknown"
        else:
            final_predictions.append({"class": top_class, "probability": top_prob})
            predicted_class_for_upload = top_class # Set folder name to the predicted class

    # 4. Upload to Google Drive (BEFORE deleting the file and returning)
    try:
        service = drive_utils.authenticate()
        dest_folder_id = drive_utils.find_or_create_folder(service, predicted_class_for_upload)
        drive_utils.upload_image(service, file_path, file.filename, dest_folder_id)
        logging.info(f"Successfully uploaded {file.filename} to Google Drive folder '{predicted_class_for_upload}'.")
    except Exception as e:
        # This will now show the detailed error in your Heroku logs
        logging.error(f"Failed to upload to Google Drive. Reason: {e}", exc_info=True)

    # 5. Clean up (delete) the local file
    os.remove(file_path)

    # 6. Return the JSON response to the user (this is the LAST step)
    return {
        "filename": file.filename,
        "predictions": final_predictions
    }