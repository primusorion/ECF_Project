"""
FastAPI Application for Predictive Maintenance
"""
from fastapi import FastAPI, HTTPException, status, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from datetime import datetime
import os
import sys
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.models import (
    SensorDataInput, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthStatus, MetricsResponse,
    SystemInfo, ErrorResponse, ThresholdUpdate, ModelInfo
)
from src.edge_inference.inference_engine import EdgeInferenceEngine
from src.utils import setup_logger

# Global variables
engine = None
start_time = None
logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global engine, start_time, logger
    
    # Startup
    logger = setup_logger('api', 'logs/api.log', 'INFO')
    logger.info("Starting Predictive Maintenance API...")
    
    start_time = time.time()
    
    # Initialize inference engine
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved_models')
    engine = EdgeInferenceEngine(models_dir)
    
    try:
        engine.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.warning("API started but models not loaded - predictions will fail")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Predictive Maintenance API...")


# Create FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="Edge AI-powered predictive maintenance for factory equipment",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to check if models are loaded
def get_engine():
    """Get inference engine dependency"""
    if engine is None or not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please check service health."
        )
    return engine


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns service health status
    """
    uptime = time.time() - start_time if start_time else 0
    
    return HealthStatus(
        status="healthy" if engine and engine.is_loaded else "degraded",
        models_loaded=engine.is_loaded if engine else False,
        uptime_seconds=uptime
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness probe for Kubernetes
    
    Returns 200 if service is ready to accept requests
    """
    if engine is None or not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready - models not loaded"
        )
    
    return {"status": "ready"}


@app.get("/info", response_model=SystemInfo, tags=["System"])
async def system_info():
    """
    Get system information
    
    Returns information about loaded models and system status
    """
    uptime = time.time() - start_time if start_time else 0
    
    models_info = {}
    if engine:
        models_info = {
            "anomaly_detector": ModelInfo(
                model_type="IsolationForest",
                is_loaded=engine.anomaly_detector is not None
            ),
            "failure_predictor": ModelInfo(
                model_type="XGBoost/Ensemble",
                is_loaded=engine.failure_predictor is not None
            ),
            "preprocessor": ModelInfo(
                model_type="StandardScaler",
                is_loaded=engine.preprocessor is not None
            )
        }
    
    return SystemInfo(
        api_version="1.0.0",
        models=models_info,
        uptime_seconds=uptime,
        total_predictions=engine.prediction_count if engine else 0
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics(inference_engine: EdgeInferenceEngine = Depends(get_engine)):
    """
    Get performance metrics
    
    Returns inference performance statistics
    """
    stats = inference_engine.get_performance_stats()
    
    if 'message' in stats:
        return MetricsResponse(
            total_predictions=0,
            avg_inference_time_ms=0.0,
            min_inference_time_ms=0.0,
            max_inference_time_ms=0.0,
            std_inference_time_ms=0.0
        )
    
    return MetricsResponse(**stats)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    sensor_data: SensorDataInput,
    inference_engine: EdgeInferenceEngine = Depends(get_engine)
):
    """
    Make prediction for single sensor reading
    
    - **equipment_id**: Unique equipment identifier
    - **equipment_type**: Type of equipment (Motor, Pump, etc.)
    - **temperature**: Temperature in °C
    - **vibration**: Vibration in mm/s
    - **pressure**: Pressure in Bar
    - **rpm**: Rotations per minute
    - **power_consumption**: Power consumption in kW
    - **acoustic_emission**: Acoustic emission in dB
    
    Returns prediction results with anomaly detection and failure probability
    """
    try:
        # Convert Pydantic model to dict
        sensor_dict = sensor_data.dict()
        
        # Make prediction
        result = inference_engine.predict_single(sensor_dict)
        
        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result['error']
            )
        
        # Convert to response model
        response = PredictionResponse(
            timestamp=datetime.fromisoformat(result['timestamp']),
            equipment_id=result['equipment_id'],
            predictions=result['predictions'],
            alert=result['alert'],
            inference_time_ms=result['inference_time_ms'],
            sensor_readings=sensor_data
        )
        
        logger.info(f"Prediction made for {sensor_data.equipment_id}: {result['alert']['level']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    inference_engine: EdgeInferenceEngine = Depends(get_engine)
):
    """
    Make predictions for multiple sensor readings
    
    Accepts up to 100 sensor readings per request
    """
    try:
        sensor_data_list = [data.dict() for data in request.sensor_data_list]
        
        # Make batch predictions
        results = inference_engine.predict_batch(sensor_data_list)
        
        # Convert to response models
        predictions = []
        total_time = 0
        
        for i, result in enumerate(results):
            if 'error' in result:
                logger.error(f"Prediction error for item {i}: {result['error']}")
                continue
            
            response = PredictionResponse(
                timestamp=datetime.fromisoformat(result['timestamp']),
                equipment_id=result['equipment_id'],
                predictions=result['predictions'],
                alert=result['alert'],
                inference_time_ms=result['inference_time_ms'],
                sensor_readings=request.sensor_data_list[i]
            )
            predictions.append(response)
            total_time += result['inference_time_ms']
        
        avg_time = total_time / len(predictions) if predictions else 0
        
        logger.info(f"Batch prediction completed: {len(predictions)} items")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            avg_inference_time_ms=avg_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.put("/thresholds", tags=["Configuration"])
async def update_thresholds(
    thresholds: ThresholdUpdate,
    inference_engine: EdgeInferenceEngine = Depends(get_engine)
):
    """
    Update alert thresholds
    
    Allows dynamic adjustment of anomaly and failure detection thresholds
    """
    try:
        # Filter out None values
        new_thresholds = {k: v for k, v in thresholds.dict().items() if v is not None}
        
        if not new_thresholds:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No thresholds provided"
            )
        
        inference_engine.update_thresholds(new_thresholds)
        logger.info(f"Thresholds updated: {new_thresholds}")
        
        return {
            "status": "success",
            "message": "Thresholds updated successfully",
            "updated_thresholds": new_thresholds
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Threshold update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/thresholds", tags=["Configuration"])
async def get_thresholds(inference_engine: EdgeInferenceEngine = Depends(get_engine)):
    """
    Get current alert thresholds
    """
    return {
        "thresholds": inference_engine.thresholds
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
