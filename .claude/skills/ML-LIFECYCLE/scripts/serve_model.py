"""
MLOps FASTAPI INFERENCE SERVER
Loads the champion model from MLflow and serves predictions via REST API.
"""

import os
import pandas as pd
from typing import List
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import mlflow.sklearn
import uvicorn


# Determine the absolute path to the mlruns directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_URI = "file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/")

# In production, this would be injected via environment variables
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Universal_ML_Pipeline")
MODEL_NAME = os.getenv("MODEL_NAME", "MultinomialNB") # Defaulting to a champion

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando MLOps Server")
    load_dotenv()
    
    mlflow.set_tracking_uri(MLRUNS_URI)
    
    model = None
    try:
        # Fetch the latest run for the experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is not None:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                                      order_by=["metrics.f1_score DESC"], max_results=1)
            
            if not runs.empty:
                best_run_id = runs.iloc[0].run_id
                model_uri = f"runs:/{best_run_id}/model_{MODEL_NAME}"
                
                print(f"Loading champion model from: {model_uri}")
                model = mlflow.sklearn.load_model(model_uri)
            else:
                print("No registered runs found.")
        else:
             print(f"Experiment '{EXPERIMENT_NAME}' not found.")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")

    app.state.model = model

    print("MLOps Server Iniciado")

    yield

    print("Apagando MLOps Server")
    app.state.model = None

app = FastAPI(title="ML-Full-Cycle Inference API", version="1.0", lifespan=lifespan)

class InferenceResponse(BaseModel):
    prediction: str
    model_used: str

@app.get("/health")
async def health_check(request: Request):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    return {"status": "healthy", "model_version": MODEL_NAME}

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: Request, payload: dict):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        # Convert JSON payload directly to DataFrame for the sklearn pipeline
        df = pd.DataFrame([payload])
        
        # The pipeline handles scaling and one-hot encoding automatically
        prediction = model.predict(df)
        
        return InferenceResponse(
            prediction=str(prediction[0]),
            model_used=MODEL_NAME
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Server running at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)