# src/api.py
import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import os
import shutil
import datetime
from .predict import predict_image

# ✅ Add SQLAlchemy imports for the database
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# --- ✅ DATABASE SETUP ---
DATABASE_URL = os.environ.get('DATABASE_URL')
# A small fix for SQLAlchemy compatibility with Heroku's URL format
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the structure of our logging table in the database
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    predicted_class = Column(String)
    confidence = Column(Float)
    is_fruit = Column(String)

# This line creates the table in your database if it doesn't already exist
Base.metadata.create_all(bind=engine)
# --- END DATABASE SETUP ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
templates = Jinja2Templates(directory="templates")

# --- Your Endpoints ---
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
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    predictions, _ = predict_image(file_path)
    os.remove(file_path)

    CONFIDENCE_THRESHOLD = 70.0
    final_predictions = []
    
    # ✅ Prepare a database session and a log entry object
    db = SessionLocal()
    db_log_entry = PredictionLog()

    if predictions:
        top_class, top_prob = predictions[0]
        if top_prob < CONFIDENCE_THRESHOLD:
            final_predictions.append({"class": "Not a Fruit!", "probability": top_prob})
            # Log what the model guessed, even if confidence was low
            db_log_entry.predicted_class = top_class
            db_log_entry.confidence = top_prob
            db_log_entry.is_fruit = "Low Confidence"
        else:
            final_predictions.append({"class": top_class, "probability": top_prob})
            db_log_entry.predicted_class = top_class
            db_log_entry.confidence = top_prob
            db_log_entry.is_fruit = "Yes"
    
    # ✅ Save the new log entry to the database
    try:
        db.add(db_log_entry)
        db.commit()
    except Exception as e:
        logging.error(f"Failed to log prediction to database. Reason: {e}")
        db.rollback()
    finally:
        db.close()
    
    return {"filename": file.filename, "predictions": final_predictions}