"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class EquipmentType(str, Enum):
    """Equipment type enumeration"""
    MOTOR = "Motor"
    PUMP = "Pump"
    COMPRESSOR = "Compressor"
    CONVEYOR = "Conveyor"


class AlertLevel(str, Enum):
    """Alert level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SensorDataInput(BaseModel):
    """Input model for sensor data"""
    equipment_id: str = Field(..., description="Unique equipment identifier")
    equipment_type: EquipmentType = Field(..., description="Type of equipment")
    temperature: float = Field(..., ge=0, le=200, description="Temperature in °C")
    vibration: float = Field(..., ge=0, le=20, description="Vibration in mm/s")
    pressure: float = Field(..., ge=0, le=20, description="Pressure in Bar")
    rpm: float = Field(..., ge=0, le=5000, description="RPM")
    power_consumption: float = Field(..., ge=0, le=200, description="Power in kW")
    acoustic_emission: float = Field(..., ge=0, le=150, description="Acoustic in dB")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "equipment_id": "Motor_001",
                "equipment_type": "Motor",
                "temperature": 75.5,
                "vibration": 4.2,
                "pressure": 7.5,
                "rpm": 1800.0,
                "power_consumption": 35.0,
                "acoustic_emission": 78.3
            }
        }


class PredictionResult(BaseModel):
    """Prediction result model"""
    anomaly_score: float = Field(..., description="Anomaly score (lower is more anomalous)")
    is_anomaly: bool = Field(..., description="Whether anomaly detected")
    failure_probability: float = Field(..., ge=0, le=1, description="Failure probability")
    failure_predicted: bool = Field(..., description="Whether failure predicted")


class AlertInfo(BaseModel):
    """Alert information model"""
    level: AlertLevel = Field(..., description="Alert severity level")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Alert timestamp")


class PredictionResponse(BaseModel):
    """Complete prediction response"""
    timestamp: datetime = Field(..., description="Prediction timestamp")
    equipment_id: str = Field(..., description="Equipment identifier")
    predictions: PredictionResult = Field(..., description="Prediction results")
    alert: AlertInfo = Field(..., description="Alert information")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    sensor_readings: SensorDataInput = Field(..., description="Input sensor data")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    sensor_data_list: List[SensorDataInput] = Field(..., min_items=1, max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "sensor_data_list": [
                    {
                        "equipment_id": "Motor_001",
                        "equipment_type": "Motor",
                        "temperature": 75.5,
                        "vibration": 4.2,
                        "pressure": 7.5,
                        "rpm": 1800.0,
                        "power_consumption": 35.0,
                        "acoustic_emission": 78.3
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total predictions made")
    avg_inference_time_ms: float = Field(..., description="Average inference time")


class HealthStatus(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    models_loaded: bool = Field(..., description="Whether models are loaded")
    version: str = Field(default="1.0.0", description="API version")
    uptime_seconds: float = Field(..., description="Service uptime")


class MetricsResponse(BaseModel):
    """Performance metrics response"""
    total_predictions: int = Field(..., description="Total predictions made")
    avg_inference_time_ms: float = Field(..., description="Average inference time")
    min_inference_time_ms: float = Field(..., description="Minimum inference time")
    max_inference_time_ms: float = Field(..., description="Maximum inference time")
    std_inference_time_ms: float = Field(..., description="Standard deviation of inference time")


class ModelInfo(BaseModel):
    """Model information"""
    model_type: str = Field(..., description="Type of model")
    is_loaded: bool = Field(..., description="Whether model is loaded")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")


class SystemInfo(BaseModel):
    """System information response"""
    api_version: str = Field(..., description="API version")
    models: Dict[str, ModelInfo] = Field(..., description="Loaded models information")
    uptime_seconds: float = Field(..., description="System uptime")
    total_predictions: int = Field(..., description="Total predictions made")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)


class ThresholdUpdate(BaseModel):
    """Threshold update request"""
    anomaly_score: Optional[float] = Field(None, description="Anomaly score threshold")
    failure_probability: Optional[float] = Field(None, ge=0, le=1, description="Failure probability threshold")
    critical_failure_probability: Optional[float] = Field(None, ge=0, le=1, description="Critical failure threshold")
    rul_critical: Optional[int] = Field(None, ge=0, description="Critical RUL threshold in hours")
    
    @validator('failure_probability', 'critical_failure_probability')
    def validate_probability(cls, v):
        if v is not None and not 0 <= v <= 1:
            raise ValueError('Probability must be between 0 and 1')
        return v


class RetrainingRequest(BaseModel):
    """Model retraining request"""
    data_source: str = Field(..., description="Path to training data")
    model_types: List[str] = Field(default=["anomaly_detector", "failure_predictor"], description="Models to retrain")
    validation_required: bool = Field(default=True, description="Whether to validate before deploying")


class RetrainingResponse(BaseModel):
    """Model retraining response"""
    status: str = Field(..., description="Retraining status")
    models_retrained: List[str] = Field(..., description="Models that were retrained")
    metrics: Dict[str, Any] = Field(..., description="Training metrics")
    timestamp: datetime = Field(default_factory=datetime.now)
