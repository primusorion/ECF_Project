"""
Integration tests for end-to-end workflows
"""
import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path


class TestEndToEndWorkflow:
    """Test complete workflow from data generation to prediction"""
    
    @pytest.mark.integration
    def test_data_generation_to_training(self, temp_test_dir):
        """Test workflow: generate data -> preprocess -> train"""
        from src.data_generation.data_generator import EquipmentDataGenerator
        from src.preprocessing.preprocessor import DataPreprocessor
        from src.models.model_trainer import AnomalyDetector, FailurePredictor
        
        # Step 1: Generate data
        generator = EquipmentDataGenerator(seed=42)
        df = generator.generate_mixed_dataset(
            total_samples=500,
            normal_ratio=0.7,
            degradation_ratio=0.25,
            failure_ratio=0.05
        )
        
        assert len(df) == 500
        
        # Step 2: Preprocess
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_pipeline(df, test_size=0.2, val_size=0.1)
        
        assert 'X_train' in data
        assert len(data['X_train']) > 0
        
        # Step 3: Train models
        anomaly_detector = AnomalyDetector(random_state=42)
        anomaly_detector.train(data['X_train'])
        assert anomaly_detector.is_fitted
        
        failure_predictor = FailurePredictor(model_type='xgboost', random_state=42)
        failure_predictor.train(data['X_train'], data['y_train'])
        assert failure_predictor.is_fitted
        
        # Step 4: Evaluate
        metrics = failure_predictor.evaluate(data['X_test'], data['y_test'])
        assert 'accuracy' in metrics
        assert metrics['accuracy'] > 0.5  # Should beat random guessing
    
    @pytest.mark.integration
    def test_training_to_inference(self, temp_test_dir, sample_dataframe):
        """Test workflow: train models -> save -> load -> predict"""
        from src.preprocessing.preprocessor import DataPreprocessor
        from src.models.model_trainer import AnomalyDetector, FailurePredictor
        
        # Train and save models
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_pipeline(sample_dataframe)
        
        # Save preprocessor
        preprocessor_path = temp_test_dir / "preprocessor.pkl"
        preprocessor.save(str(preprocessor_path))
        
        # Train and save anomaly detector
        anomaly_detector = AnomalyDetector(random_state=42)
        anomaly_detector.train(data['X_train'])
        anomaly_path = temp_test_dir / "anomaly_detector.pkl"
        anomaly_detector.save(str(anomaly_path))
        
        # Train and save failure predictor
        failure_predictor = FailurePredictor(model_type='xgboost', random_state=42)
        failure_predictor.train(data['X_train'], data['y_train'])
        failure_path = temp_test_dir / "failure_predictor.pkl"
        failure_predictor.save(str(failure_path))
        
        # Verify files exist
        assert preprocessor_path.exists()
        assert anomaly_path.exists()
        assert failure_path.exists()
        
        # Load and predict
        new_detector = AnomalyDetector()
        new_detector.load(str(anomaly_path))
        predictions = new_detector.predict(data['X_test'][:5])
        assert len(predictions) == 5
    
    @pytest.mark.integration
    def test_config_driven_workflow(self, test_config, temp_test_dir):
        """Test that config values are properly used throughout workflow"""
        from src.data_generation.data_generator import EquipmentDataGenerator
        
        config = test_config
        
        # Use config for data generation
        generator = EquipmentDataGenerator(seed=config['data_generation']['seed'])
        df = generator.generate_mixed_dataset(
            total_samples=config['data_generation']['total_samples'],
            normal_ratio=config['data_generation']['normal_ratio'],
            degradation_ratio=config['data_generation']['degradation_ratio'],
            failure_ratio=config['data_generation']['failure_ratio']
        )
        
        assert len(df) == config['data_generation']['total_samples']
        
        # Verify ratios
        status_counts = df['status'].value_counts()
        expected_normal = int(config['data_generation']['total_samples'] * 
                            config['data_generation']['normal_ratio'])
        assert status_counts['Normal'] == expected_normal
