"""
Unit tests for inference_engine module
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.edge_inference.inference_engine import EdgeInferenceEngine, StreamProcessor


class TestEdgeInferenceEngine:
    """Test cases for EdgeInferenceEngine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = EdgeInferenceEngine(models_dir="models/saved_models")
        assert engine.models_dir == "models/saved_models"
        assert not engine.is_loaded
        assert engine.prediction_count == 0
    
    def test_threshold_defaults(self):
        """Test default alert thresholds"""
        engine = EdgeInferenceEngine(models_dir="models/saved_models")
        
        assert 'anomaly_score' in engine.thresholds
        assert 'failure_probability' in engine.thresholds
        assert engine.thresholds['failure_probability'] == 0.7
    
    def test_update_thresholds(self):
        """Test updating alert thresholds"""
        engine = EdgeInferenceEngine(models_dir="models/saved_models")
        
        new_thresholds = {'failure_probability': 0.8}
        engine.update_thresholds(new_thresholds)
        
        assert engine.thresholds['failure_probability'] == 0.8
    
    def test_determine_alert_level(self):
        """Test alert level determination"""
        engine = EdgeInferenceEngine(models_dir="models/saved_models")
        
        # Critical
        level = engine._determine_alert_level(-0.6, 0.95)
        assert level == 'CRITICAL'
        
        # High
        level = engine._determine_alert_level(-0.3, 0.75)
        assert level == 'HIGH'
        
        # Medium
        level = engine._determine_alert_level(-0.6, 0.5)
        assert level == 'MEDIUM'
        
        # Low
        level = engine._determine_alert_level(-0.3, 0.3)
        assert level == 'LOW'
    
    def test_generate_alert_message(self):
        """Test alert message generation"""
        engine = EdgeInferenceEngine(models_dir="models/saved_models")
        
        msg = engine._generate_alert_message('CRITICAL', 0.95)
        assert 'CRITICAL' in msg
        assert '95' in msg
        
        msg = engine._generate_alert_message('LOW', 0.2)
        assert 'normally' in msg.lower()
    
    def test_preprocess_sensor_data(self, sample_sensor_data):
        """Test sensor data preprocessing"""
        engine = EdgeInferenceEngine(models_dir="models/saved_models")
        
        # Mock preprocessor
        engine.preprocessor = {
            'feature_columns': ['temperature', 'vibration', 'pressure', 'rpm',
                              'power_consumption', 'acoustic_emission',
                              'temp_vibration_interaction', 'power_efficiency',
                              'equipment_type_Motor', 'equipment_type_Pump',
                              'equipment_type_Compressor', 'equipment_type_Conveyor'],
            'scaler': Mock()
        }
        engine.preprocessor['scaler'].transform = Mock(return_value=np.random.randn(1, 12))
        
        X = engine.preprocess_sensor_data(sample_sensor_data)
        
        assert X.shape[0] == 1  # Single sample
        assert engine.preprocessor['scaler'].transform.called
    
    def test_performance_stats_empty(self):
        """Test performance stats with no predictions"""
        engine = EdgeInferenceEngine(models_dir="models/saved_models")
        stats = engine.get_performance_stats()
        
        assert 'message' in stats
    
    def test_performance_stats_with_data(self):
        """Test performance stats with predictions"""
        engine = EdgeInferenceEngine(models_dir="models/saved_models")
        engine.inference_times = [0.01, 0.02, 0.015, 0.012]
        engine.prediction_count = 4
        
        stats = engine.get_performance_stats()
        
        assert 'total_predictions' in stats
        assert 'avg_inference_time_ms' in stats
        assert stats['total_predictions'] == 4
        assert stats['avg_inference_time_ms'] > 0


class TestStreamProcessor:
    """Test cases for StreamProcessor"""
    
    def test_initialization(self):
        """Test stream processor initialization"""
        engine = Mock()
        processor = StreamProcessor(engine, buffer_size=10)
        
        assert processor.buffer_size == 10
        assert len(processor.recent_predictions) == 0
        assert len(processor.alert_history) == 0
    
    def test_process_stream(self, sample_sensor_data):
        """Test processing stream data"""
        engine = Mock()
        engine.predict_single = Mock(return_value={
            'predictions': {'failure_probability': 0.5},
            'alert': {'level': 'LOW'}
        })
        
        processor = StreamProcessor(engine, buffer_size=5)
        result = processor.process_stream(sample_sensor_data)
        
        assert engine.predict_single.called
        assert len(processor.recent_predictions) == 1
    
    def test_buffer_limit(self, sample_sensor_data):
        """Test that buffer respects size limit"""
        engine = Mock()
        engine.predict_single = Mock(return_value={
            'predictions': {'failure_probability': 0.5},
            'alert': {'level': 'LOW'}
        })
        
        processor = StreamProcessor(engine, buffer_size=3)
        
        # Add more than buffer size
        for _ in range(5):
            processor.process_stream(sample_sensor_data)
        
        assert len(processor.recent_predictions) == 3
    
    def test_alert_tracking(self, sample_sensor_data):
        """Test that high/critical alerts are tracked"""
        engine = Mock()
        
        # Mix of alert levels
        results = [
            {'predictions': {}, 'alert': {'level': 'LOW'}},
            {'predictions': {}, 'alert': {'level': 'HIGH'}},
            {'predictions': {}, 'alert': {'level': 'MEDIUM'}},
            {'predictions': {}, 'alert': {'level': 'CRITICAL'}},
        ]
        
        engine.predict_single = Mock(side_effect=results)
        processor = StreamProcessor(engine, buffer_size=10)
        
        for _ in range(4):
            processor.process_stream(sample_sensor_data)
        
        # Should track 2 alerts (HIGH and CRITICAL)
        assert len(processor.alert_history) == 2
    
    def test_get_recent_trend(self, sample_sensor_data):
        """Test trend analysis"""
        engine = Mock()
        engine.predict_single = Mock(return_value={
            'predictions': {
                'failure_probability': 0.5,
                'anomaly_score': -0.3
            },
            'alert': {'level': 'LOW'}
        })
        
        processor = StreamProcessor(engine, buffer_size=5)
        
        # Process some data
        for _ in range(3):
            processor.process_stream(sample_sensor_data)
        
        trend = processor.get_recent_trend()
        
        assert 'num_predictions' in trend
        assert 'avg_failure_probability' in trend
        assert trend['num_predictions'] == 3
