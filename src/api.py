# src/api.py
import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
import shutil
import datetime
from .predict import predict_image

# ✅ Add Cloudinary imports
import cloudinary
import cloudinary.uploader

# Add SQLAlchemy imports for the database
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# --- ✅ CLOUDINARY CONFIGURATION ---
cloudinary.config(
    cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key = os.environ.get('CLOUDINARY_API_KEY'),
    api_secret = os.environ.get('CLOUDINARY_API_SECRET'),
    secure = True
)
# --- END CLOUDINARY CONFIGURATION ---

# --- DATABASE SETUP (remains the same) ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    # ... (table definition is the same)
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    predicted_class = Column(String)
    confidence = Column(Float)
    is_fruit = Column(String)
Base.metadata.create_all(bind=engine)
# --- END DATABASE SETUP ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
templates = Jinja2Templates(directory="templates")

# --- Endpoints (remain the same) ---
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
# ... (other GET endpoints are the same)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    predictions, _ = predict_image(file_path)

    CONFIDENCE_THRESHOLD = 70.0
    final_predictions = []
    predicted_class_for_upload = "Unknown"

    if predictions:
        top_class, top_prob = predictions[0]
        if top_prob < CONFIDENCE_THRESHOLD:
            final_predictions.append({"class": "Not a Fruit!", "probability": top_prob})
        else:
            final_predictions.append({"class": top_class, "probability": top_prob})
            predicted_class_for_upload = top_class
    
    # ✅ --- UPLOAD IMAGE TO CLOUDINARY --- ✅
    try:
        # The 'folder' parameter automatically creates folders by class name!
        cloudinary.uploader.upload(file_path, folder=predicted_class_for_upload)
        logging.info(f"Successfully uploaded {file.filename} to Cloudinary folder '{predicted_class_for_upload}'.")
    except Exception as e:
        logging.error(f"Failed to upload to Cloudinary. Reason: {e}", exc_info=True)

    # Log to database (remains the same)
    # ...

    os.remove(file_path)

    return {"filename": file.filename, "predictions": final_predictions}