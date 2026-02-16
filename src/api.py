# src/api.py
import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
import shutil
import datetime
from .predict import predict_image

# Add Cloudinary imports
import cloudinary
import cloudinary.uploader

# Add SQLAlchemy imports for the database
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# --- CLOUDINARY CONFIGURATION ---
# Reads your secret keys from Heroku Config Vars
cloudinary.config(
    cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key = os.environ.get('CLOUDINARY_API_KEY'),
    api_secret = os.environ.get('CLOUDINARY_API_SECRET'),
    secure = True
)
# --- END CLOUDINARY CONFIGURATION ---

# --- DATABASE SETUP ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the database table with new columns for Cloudinary info
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    predicted_class = Column(String)
    confidence = Column(Float)
    is_fruit = Column(String)
    # Added new columns to link to the uploaded image
    cloudinary_id = Column(String, nullable=True)
    cloudinary_url = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)
# --- END DATABASE SETUP ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
templates = Jinja2Templates(directory="templates")


# --- Endpoints ---
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
    # 1. Save file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Run prediction
    predictions, _ = predict_image(file_path)

    # 3. Determine final prediction and class name for folder/logging
    CONFIDENCE_THRESHOLD = 70.0
    final_predictions = []
    log_entry_data = {"predicted_class": "Error", "confidence": 0.0, "is_fruit": "Error"}
    predicted_class_for_upload = "Unknown"

    if predictions:
        top_class, top_prob = predictions[0]
        if top_prob < CONFIDENCE_THRESHOLD:
            final_predictions.append({"class": "Not a Fruit!", "probability": top_prob})
            log_entry_data = {"predicted_class": top_class, "confidence": top_prob, "is_fruit": "Low Confidence"}
        else:
            final_predictions.append({"class": top_class, "probability": top_prob})
            log_entry_data = {"predicted_class": top_class, "confidence": top_prob, "is_fruit": "Yes"}
            predicted_class_for_upload = top_class
    
    # 4. Upload image to Cloudinary
    cloudinary_id = None
    cloudinary_url = None
    try:
        upload_result = cloudinary.uploader.upload(file_path, folder=predicted_class_for_upload)
        cloudinary_id = upload_result.get('public_id')
        cloudinary_url = upload_result.get('secure_url')
        logging.info(f"Successfully uploaded to Cloudinary. Public ID: {cloudinary_id}")
    except Exception as e:
        logging.error(f"Failed to upload to Cloudinary. Reason: {e}", exc_info=True)

    # 5. Log everything to the database
    db = SessionLocal()
    try:
        db_log_entry = PredictionLog(
            predicted_class=log_entry_data["predicted_class"],
            confidence=log_entry_data["confidence"],
            is_fruit=log_entry_data["is_fruit"],
            cloudinary_id=cloudinary_id,
            cloudinary_url=cloudinary_url
        )
        db.add(db_log_entry)
        db.commit()
        logging.info(f"Successfully logged prediction for {cloudinary_id} to database.")
    except Exception as e:
        logging.error(f"Failed to log to database. Reason: {e}")
        db.rollback()
    finally:
        db.close()

    # 6. Clean up the local file
    os.remove(file_path)

    # 7. Return the response to the user
    return {"filename": file.filename, "predictions": final_predictions}