import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from naive_bayes_logic.management import train_model_workflow, predict_workflow, get_model_status
from backend.models import PredictionRequest, PredictionResponse, TrainResponse, ModelStatusResponse
import shutil
import logging



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Naive Bayes Classifier API",
    description="API for training a Naive Bayes model and making predictions.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Directory to temporarily store uploaded CSV files
UPLOAD_DIR = "uploaded_datasets"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/train", response_model=TrainResponse)
async def train_model(file: UploadFile = File(...)):
    """
    Uploads a CSV dataset, trains the Naive Bayes model, and returns its accuracy.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File '{file.filename}' uploaded successfully to {file_path}")

        # Run the training workflow
        result = train_model_workflow(file_path)
        logger.info(f"Model training completed with accuracy: {result['accuracy']:.2f}%")

        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during model training: {e}")
    finally:
        # Clean up the uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up uploaded file: {file_path}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Makes a prediction using the trained Naive Bayes model based on provided features.
    """
    try:
        prediction_result = predict_workflow(request.features)
        logger.info(f"Prediction made: {prediction_result['prediction']}")
        return JSONResponse(content=prediction_result, status_code=200)
    except ValueError as e:
        logger.warning(f"Prediction input error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")


@app.get("/status", response_model=ModelStatusResponse)
async def get_status():
    """
    Returns the current status of the trained model, including accuracy and features.
    """
    try:
        status = get_model_status()
        logger.info(f"Model status requested: {status['status']}")
        return JSONResponse(content=status, status_code=200)
    except Exception as e:
        logger.error(f"Error getting model status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving model status: {e}")
