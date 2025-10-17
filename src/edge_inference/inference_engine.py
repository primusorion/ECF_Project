"""
Edge Inference Engine for Predictive Maintenance
Optimized for deployment on edge devices with limited resources
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pickle
import os
import time
import json
from datetime import datetime


class EdgeInferenceEngine:
    """
    Lightweight inference engine for edge deployment
    Loads models and performs real-time predictions with minimal latency
    """
    
    def __init__(self, models_dir: str):
        """
        Initialize edge inference engine
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.preprocessor = None
        self.anomaly_detector = None
        self.failure_predictor = None
        self.rul_estimator = None
        self.is_loaded = False
        
        # Performance tracking
        self.inference_times = []
        self.prediction_count = 0
        
        # Alert thresholds
        self.thresholds = {
            'anomaly_score': -0.5,
            'failure_probability': 0.7,
            'critical_failure_probability': 0.9,
            'rul_critical': 50  # hours
        }
    
    def load_models(self):
        """Load all required models"""
        print("Loading models for edge inference...")
        
        try:
            # Load preprocessor
            preprocessor_path = os.path.join(
                os.path.dirname(self.models_dir), 
                'preprocessor.pkl'
            )
            with open(preprocessor_path, 'rb') as f:
                preprocessor_data = pickle.load(f)
                self.preprocessor = preprocessor_data
            print("[OK] Preprocessor loaded")
            
            # Load anomaly detector
            anomaly_path = os.path.join(
                self.models_dir, 
                'anomaly_detector.pkl'
            )
            with open(anomaly_path, 'rb') as f:
                self.anomaly_detector = pickle.load(f)
            print("[OK] Anomaly Detector loaded")
            
            # Load failure predictor
            failure_path = os.path.join(
                self.models_dir,
                'failure_predictor.pkl'
            )
            with open(failure_path, 'rb') as f:
                failure_data = pickle.load(f)
                self.failure_predictor = failure_data
            print("[OK] Failure Predictor loaded")
            
            self.is_loaded = True
            print("[OK] All models loaded successfully\n")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def preprocess_sensor_data(self, sensor_data: Dict) -> np.ndarray:
        """
        Preprocess raw sensor data for inference
        
        Args:
            sensor_data: Dictionary with sensor readings
            
        Returns:
            Preprocessed feature array
        """
        # Convert to DataFrame
        df = pd.DataFrame([sensor_data])
        
        # Basic feature engineering (simplified for edge)
        df['temp_vibration_interaction'] = df['temperature'] * df['vibration']
        df['power_efficiency'] = df['power_consumption'] / (df['rpm'] + 1)
        
        # One-hot encode equipment type if needed
        if 'equipment_type' in df.columns:
            equipment_types = ['Motor', 'Pump', 'Compressor', 'Conveyor']
            for eq_type in equipment_types:
                df[f'equipment_type_{eq_type}'] = (df['equipment_type'] == eq_type).astype(int)
            df.drop('equipment_type', axis=1, inplace=True)
        
        # Remove non-feature columns
        exclude_cols = ['equipment_id', 'timestamp', 'status']
        for col in exclude_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        # Ensure all required features are present
        feature_columns = self.preprocessor['feature_columns']
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select and order features
        df = df[feature_columns]
        
        # Scale features
        X_scaled = self.preprocessor['scaler'].transform(df)
        
        return X_scaled
    
    def predict_single(self, sensor_data: Dict) -> Dict:
        """
        Make predictions for a single sensor reading
        
        Args:
            sensor_data: Dictionary with sensor readings
            
        Returns:
            Dictionary with predictions and alerts
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess data
            X = self.preprocess_sensor_data(sensor_data)
            
            # Anomaly detection
            anomaly_score = self.anomaly_detector.score_samples(X)[0]
            is_anomaly = anomaly_score < self.thresholds['anomaly_score']
            
            # Failure prediction
            if self.failure_predictor['model_type'] == 'ensemble':
                failure_probs = []
                for model in self.failure_predictor['models'].values():
                    prob = model.predict_proba(X)[0, 1]
                    failure_probs.append(prob)
                failure_probability = np.mean(failure_probs)
            else:
                failure_probability = self.failure_predictor['model'].predict_proba(X)[0, 1]
            
            # Determine alert level
            alert_level = self._determine_alert_level(
                anomaly_score, failure_probability
            )
            
            # Compile results
            result = {
                'timestamp': datetime.now().isoformat(),
                'equipment_id': sensor_data.get('equipment_id', 'unknown'),
                'predictions': {
                    'anomaly_score': float(anomaly_score),
                    'is_anomaly': bool(is_anomaly),
                    'failure_probability': float(failure_probability),
                    'failure_predicted': bool(failure_probability > self.thresholds['failure_probability'])
                },
                'alert': {
                    'level': alert_level,
                    'message': self._generate_alert_message(alert_level, failure_probability)
                },
                'sensor_readings': sensor_data
            }
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.prediction_count += 1
            
            result['inference_time_ms'] = inference_time * 1000
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, sensor_data_list: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple sensor readings
        
        Args:
            sensor_data_list: List of sensor reading dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for sensor_data in sensor_data_list:
            result = self.predict_single(sensor_data)
            results.append(result)
        
        return results
    
    def _determine_alert_level(
        self, 
        anomaly_score: float, 
        failure_probability: float
    ) -> str:
        """Determine alert level based on predictions"""
        if failure_probability >= self.thresholds['critical_failure_probability']:
            return 'CRITICAL'
        elif failure_probability >= self.thresholds['failure_probability']:
            return 'HIGH'
        elif anomaly_score < self.thresholds['anomaly_score']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_alert_message(
        self, 
        alert_level: str, 
        failure_probability: float
    ) -> str:
        """Generate human-readable alert message"""
        messages = {
            'CRITICAL': f'CRITICAL: Imminent equipment failure detected! '
                       f'Failure probability: {failure_probability:.1%}. '
                       f'Immediate maintenance required.',
            'HIGH': f'HIGH: High risk of equipment failure. '
                   f'Failure probability: {failure_probability:.1%}. '
                   f'Schedule maintenance within 24 hours.',
            'MEDIUM': 'MEDIUM: Unusual equipment behavior detected. '
                     'Monitor closely and prepare for maintenance.',
            'LOW': 'LOW: Equipment operating normally.'
        }
        return messages.get(alert_level, 'Unknown alert level')
    
    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {'message': 'No predictions made yet'}
        
        return {
            'total_predictions': self.prediction_count,
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000,
            'min_inference_time_ms': np.min(self.inference_times) * 1000,
            'max_inference_time_ms': np.max(self.inference_times) * 1000,
            'std_inference_time_ms': np.std(self.inference_times) * 1000
        }
    
    def update_thresholds(self, new_thresholds: Dict):
        """Update alert thresholds"""
        self.thresholds.update(new_thresholds)
        print(f"Thresholds updated: {self.thresholds}")


class StreamProcessor:
    """Processes streaming sensor data in real-time"""
    
    def __init__(self, inference_engine: EdgeInferenceEngine, buffer_size: int = 10):
        """
        Initialize stream processor
        
        Args:
            inference_engine: EdgeInferenceEngine instance
            buffer_size: Number of recent predictions to keep
        """
        self.engine = inference_engine
        self.buffer_size = buffer_size
        self.recent_predictions = []
        self.alert_history = []
    
    def process_stream(self, sensor_data: Dict) -> Dict:
        """
        Process a single streaming data point
        
        Args:
            sensor_data: Sensor reading dictionary
            
        Returns:
            Prediction result
        """
        result = self.engine.predict_single(sensor_data)
        
        # Add to buffer
        self.recent_predictions.append(result)
        if len(self.recent_predictions) > self.buffer_size:
            self.recent_predictions.pop(0)
        
        # Track alerts
        if result.get('alert', {}).get('level') in ['HIGH', 'CRITICAL']:
            self.alert_history.append(result)
        
        return result
    
    def get_recent_trend(self) -> Dict:
        """Analyze recent prediction trend"""
        if not self.recent_predictions:
            return {'message': 'No recent predictions'}
        
        failure_probs = [
            p['predictions']['failure_probability'] 
            for p in self.recent_predictions 
            if 'predictions' in p
        ]
        
        anomaly_scores = [
            p['predictions']['anomaly_score']
            for p in self.recent_predictions
            if 'predictions' in p
        ]
        
        return {
            'num_predictions': len(self.recent_predictions),
            'avg_failure_probability': np.mean(failure_probs) if failure_probs else 0,
            'trend': 'increasing' if len(failure_probs) > 1 and 
                    failure_probs[-1] > failure_probs[0] else 'stable',
            'recent_alerts': len([p for p in self.recent_predictions 
                                if p.get('alert', {}).get('level') in ['HIGH', 'CRITICAL']])
        }
    
    def export_alert_history(self, filepath: str):
        """Export alert history to JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.alert_history, f, indent=2)
        print(f"Alert history exported to: {filepath}")


def main():
    """Demo edge inference"""
    models_dir = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'models', 'saved_models'
    )
    
    # Initialize engine
    engine = EdgeInferenceEngine(models_dir)
    
    try:
        engine.load_models()
    except Exception as e:
        print(f"Could not load models: {e}")
        print("Please train models first by running model_trainer.py")
        return
    
    # Create sample sensor data
    sample_data = {
        'equipment_id': 'Motor_001',
        'equipment_type': 'Motor',
        'temperature': 85.5,
        'vibration': 6.2,
        'pressure': 7.8,
        'rpm': 1750,
        'power_consumption': 42.5,
        'acoustic_emission': 82.3
    }
    
    print("=== Edge Inference Demo ===\n")
    print("Sample Sensor Data:")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
    
    print("\nRunning inference...")
    result = engine.predict_single(sample_data)
    
    print("\n=== Prediction Results ===")
    print(json.dumps(result, indent=2))
    
    print("\n=== Performance Stats ===")
    stats = engine.get_performance_stats()
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
