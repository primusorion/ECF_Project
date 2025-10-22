"""
Unit tests for model_trainer module
"""
import pytest
import numpy as np
from src.models.model_trainer import (
    AnomalyDetector,
    FailurePredictor,
    RULEstimator
)


class TestAnomalyDetector:
    """Test cases for AnomalyDetector"""
    
    def test_initialization(self):
        """Test anomaly detector initialization"""
        detector = AnomalyDetector(contamination=0.1, random_state=42)
        assert detector.contamination == 0.1
        assert not detector.is_fitted
    
    def test_training(self):
        """Test anomaly detector training"""
        detector = AnomalyDetector(random_state=42)
        X_train = np.random.randn(100, 10)
        
        detector.train(X_train)
        assert detector.is_fitted
    
    def test_prediction(self):
        """Test anomaly prediction"""
        detector = AnomalyDetector(random_state=42)
        X_train = np.random.randn(100, 10)
        detector.train(X_train)
        
        X_test = np.random.randn(10, 10)
        predictions = detector.predict(X_test)
        
        assert len(predictions) == 10
        assert set(predictions).issubset({-1, 1})  # -1 for anomaly, 1 for normal
    
    def test_anomaly_scores(self):
        """Test anomaly score calculation"""
        detector = AnomalyDetector(random_state=42)
        X_train = np.random.randn(100, 10)
        detector.train(X_train)
        
        X_test = np.random.randn(10, 10)
        scores = detector.get_anomaly_scores(X_test)
        
        assert len(scores) == 10
        assert all(isinstance(score, (float, np.floating)) for score in scores)
    
    def test_save_and_load(self, temp_test_dir):
        """Test saving and loading model"""
        detector = AnomalyDetector(random_state=42)
        X_train = np.random.randn(100, 10)
        detector.train(X_train)
        
        # Save
        save_path = temp_test_dir / "anomaly_detector.pkl"
        detector.save(str(save_path))
        assert save_path.exists()
        
        # Load
        new_detector = AnomalyDetector()
        new_detector.load(str(save_path))
        assert new_detector.is_fitted


class TestFailurePredictor:
    """Test cases for FailurePredictor"""
    
    @pytest.mark.parametrize("model_type", ['random_forest', 'xgboost', 'lightgbm'])
    def test_initialization(self, model_type):
        """Test failure predictor initialization with different models"""
        predictor = FailurePredictor(model_type=model_type, random_state=42)
        assert predictor.model_type == model_type
        assert not predictor.is_fitted
    
    def test_training(self):
        """Test failure predictor training"""
        predictor = FailurePredictor(model_type='xgboost', random_state=42)
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        predictor.train(X_train, y_train)
        assert predictor.is_fitted
    
    def test_prediction(self):
        """Test failure prediction"""
        predictor = FailurePredictor(model_type='xgboost', random_state=42)
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        predictor.train(X_train, y_train)
        
        X_test = np.random.randn(10, 10)
        predictions = predictor.predict(X_test)
        
        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions)  # Probabilities
    
    def test_predict_class(self):
        """Test class prediction with threshold"""
        predictor = FailurePredictor(model_type='xgboost', random_state=42)
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        predictor.train(X_train, y_train)
        
        X_test = np.random.randn(10, 10)
        class_predictions = predictor.predict_class(X_test, threshold=0.5)
        
        assert len(class_predictions) == 10
        assert set(class_predictions).issubset({0, 1})
    
    def test_evaluation(self):
        """Test model evaluation"""
        predictor = FailurePredictor(model_type='xgboost', random_state=42)
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        predictor.train(X_train, y_train)
        
        X_test = np.random.randn(50, 10)
        y_test = np.random.randint(0, 2, 50)
        
        metrics = predictor.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_ensemble_mode(self):
        """Test ensemble model mode"""
        predictor = FailurePredictor(model_type='ensemble', random_state=42)
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        predictor.train(X_train, y_train)
        
        assert len(predictor.models) == 3  # RF, XGB, LGB
        assert predictor.is_fitted


class TestRULEstimator:
    """Test cases for RULEstimator"""
    
    def test_initialization(self):
        """Test RUL estimator initialization"""
        estimator = RULEstimator(random_state=42)
        assert not estimator.is_fitted
    
    def test_training(self):
        """Test RUL estimator training"""
        estimator = RULEstimator(random_state=42)
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.uniform(0, 500, 100)  # RUL in hours
        
        estimator.train(X_train, y_train)
        assert estimator.is_fitted
    
    def test_prediction(self):
        """Test RUL prediction"""
        estimator = RULEstimator(random_state=42)
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.uniform(0, 500, 100)
        estimator.train(X_train, y_train)
        
        X_test = np.random.randn(10, 10)
        predictions = estimator.predict(X_test)
        
        assert len(predictions) == 10
        assert all(isinstance(p, (float, np.floating)) for p in predictions)
    
    def test_evaluation(self):
        """Test RUL evaluation metrics"""
        estimator = RULEstimator(random_state=42)
        
        X_train = np.random.randn(100, 10)
        y_train = np.random.uniform(100, 500, 100)
        estimator.train(X_train, y_train)
        
        X_test = np.random.randn(50, 10)
        y_test = np.random.uniform(100, 500, 50)
        
        metrics = estimator.evaluate(X_test, y_test)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert all(v >= 0 for v in metrics.values())
