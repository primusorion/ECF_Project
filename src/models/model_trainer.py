"""
Machine Learning Models for Predictive Maintenance
Includes Anomaly Detection, Failure Prediction, and RUL Estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import pickle
import os
import json
from datetime import datetime

# Classical ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_recall_fscore_support
)
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class AnomalyDetector:
    """Detects anomalies in sensor data using Isolation Forest"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random seed
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.is_fitted = False
        
    def train(self, X: np.ndarray):
        """Train the anomaly detection model"""
        print("Training Anomaly Detector...")
        self.model.fit(X)
        self.is_fitted = True
        print("[OK] Anomaly Detector trained")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (lower is more anomalous)
        
        Returns:
            Array of anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before scoring")
        return self.model.score_samples(X)
    
    def save(self, filepath: str):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Anomaly Detector saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        print(f"Anomaly Detector loaded from: {filepath}")


class FailurePredictor:
    """Predicts equipment failure using ensemble methods"""
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        Initialize failure predictor
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm', 'ensemble')
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.models = {}
        self.is_fitted = False
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'ensemble':
            # Train multiple models
            self.models = {
                'rf': RandomForestClassifier(
                    n_estimators=150, max_depth=15, random_state=self.random_state, n_jobs=-1
                ),
                'xgb': xgb.XGBClassifier(
                    n_estimators=150, max_depth=8, learning_rate=0.1, 
                    random_state=self.random_state, n_jobs=-1
                ),
                'lgb': lgb.LGBMClassifier(
                    n_estimators=150, max_depth=8, learning_rate=0.1,
                    random_state=self.random_state, n_jobs=-1
                )
            }
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the failure prediction model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print(f"Training Failure Predictor ({self.model_type})...")
        
        if self.model_type == 'ensemble':
            for name, model in self.models.items():
                print(f"  Training {name}...")
                model.fit(X_train, y_train)
        else:
            if X_val is not None and y_val is not None and self.model_type in ['xgboost', 'lightgbm']:
                eval_set = [(X_val, y_val)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        print("[OK] Failure Predictor trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict failure probability"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if self.model_type == 'ensemble':
            predictions = []
            for model in self.models.values():
                pred = model.predict_proba(X)[:, 1]
                predictions.append(pred)
            return np.mean(predictions, axis=0)
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict failure class"""
        probs = self.predict(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred_proba = self.predict(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary'
        )
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance"""
        if self.model_type == 'ensemble':
            # Average importance across models
            importances = []
            for model in self.models.values():
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            avg_importance = np.mean(importances, axis=0)
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            return None
    
    def save(self, filepath: str):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model_type == 'ensemble':
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model_type': self.model_type,
                    'models': self.models
                }, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model_type': self.model_type,
                    'model': self.model
                }, f)
        
        print(f"Failure Predictor saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model_type = data['model_type']
            
            if self.model_type == 'ensemble':
                self.models = data['models']
            else:
                self.model = data['model']
        
        self.is_fitted = True
        print(f"Failure Predictor loaded from: {filepath}")


class RULEstimator:
    """Estimates Remaining Useful Life using regression"""
    
    def __init__(self, random_state: int = 42):
        """Initialize RUL estimator"""
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train RUL estimation model"""
        print("Training RUL Estimator...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print("[OK] RUL Estimator trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict remaining useful life"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mape = np.mean(np.abs((y - y_pred) / (y + 1))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def save(self, filepath: str):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"RUL Estimator saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        print(f"RUL Estimator loaded from: {filepath}")


def train_all_models(
    data: Dict,
    output_dir: str = 'models/saved_models',
    model_type: str = 'xgboost'
) -> Dict:
    """
    Train all models and save them
    
    Args:
        data: Dictionary with preprocessed data
        output_dir: Directory to save models
        model_type: Type of failure prediction model
        
    Returns:
        Dictionary with trained models and metrics
    """
    results = {}
    
    # 1. Train Anomaly Detector
    anomaly_detector = AnomalyDetector()
    anomaly_detector.train(data['X_train'])
    anomaly_detector.save(os.path.join(output_dir, 'anomaly_detector.pkl'))
    results['anomaly_detector'] = anomaly_detector
    
    # 2. Train Failure Predictor
    failure_predictor = FailurePredictor(model_type=model_type)
    failure_predictor.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Evaluate
    train_metrics = failure_predictor.evaluate(data['X_train'], data['y_train'])
    val_metrics = failure_predictor.evaluate(data['X_val'], data['y_val'])
    test_metrics = failure_predictor.evaluate(data['X_test'], data['y_test'])
    
    print("\n=== Failure Predictor Performance ===")
    print(f"Train - Accuracy: {train_metrics['accuracy']:.4f}, AUC-ROC: {train_metrics['roc_auc']:.4f}")
    print(f"Val   - Accuracy: {val_metrics['accuracy']:.4f}, AUC-ROC: {val_metrics['roc_auc']:.4f}")
    print(f"Test  - Accuracy: {test_metrics['accuracy']:.4f}, AUC-ROC: {test_metrics['roc_auc']:.4f}")
    print(f"Test  - Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
    
    failure_predictor.save(os.path.join(output_dir, 'failure_predictor.pkl'))
    results['failure_predictor'] = failure_predictor
    results['metrics'] = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    
    print(f"\n[OK] All models trained and saved to: {output_dir}")
    print(f"[OK] Metrics saved to: {metrics_path}")
    
    return results


def main():
    """Train models on preprocessed data"""
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.preprocessing.preprocessor import DataPreprocessor
    
    # Load data
    data_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'data', 'synthetic', 'sensor_data.csv'
    )
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please run data generation first.")
        return
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(df)
    
    # Save preprocessor
    preprocessor_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'models', 'preprocessor.pkl'
    )
    preprocessor.save(preprocessor_path)
    
    # Train models
    output_dir = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'models', 'saved_models'
    )
    
    results = train_all_models(data, output_dir, model_type='xgboost')
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == '__main__':
    main()
