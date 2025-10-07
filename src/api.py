# src/api.py
from fastapi import FastAPI, UploadFile, File
import shutil
import os
from .predict import predict_image
from fastapi.templating import Jinja2Templates 
from fastapi.responses import HTMLResponse, FileResponse
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
    predictions, _ = predict_image(file_path)
    os.remove(file_path)

    # ✅ --- NEW LOGIC STARTS HERE --- ✅
    CONFIDENCE_THRESHOLD = 50.0  # You can adjust this value (e.g., 75.0, 80.0)

    final_predictions = []
    if predictions:
        top_class, top_prob = predictions[0]

        # Check if the top prediction is below our confidence threshold
        if top_prob < CONFIDENCE_THRESHOLD:
            final_predictions.append({"class": "Not a Fruit!", "probability": top_prob})
        else:
            # If confidence is high enough, just return the top prediction
            final_predictions.append({"class": top_class, "probability": top_prob})
    # ✅ --- NEW LOGIC ENDS HERE --- ✅
    
    return {
        "filename": file.filename,
        "predictions": final_predictions # Return our carefully checked prediction
    }

@app.get("/facts", response_class=FileResponse)
async def get_facts():
    """Serves the facts.json file."""
    return "templates/facts.json"
    
templates = Jinja2Templates(directory="templates")

@app.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})