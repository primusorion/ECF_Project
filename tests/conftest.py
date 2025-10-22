"""
Pytest configuration and fixtures
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing"""
    return {
        'equipment_id': 'Motor_001',
        'equipment_type': 'Motor',
        'temperature': 75.5,
        'vibration': 4.2,
        'pressure': 7.5,
        'rpm': 1800,
        'power_consumption': 35.0,
        'acoustic_emission': 78.3
    }


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'equipment_id': [f'Motor_{i%5:03d}' for i in range(n_samples)],
        'equipment_type': np.random.choice(['Motor', 'Pump', 'Compressor'], n_samples),
        'temperature': np.random.normal(70, 10, n_samples),
        'vibration': np.random.normal(4, 1, n_samples),
        'pressure': np.random.normal(7, 1, n_samples),
        'rpm': np.random.normal(1800, 200, n_samples),
        'power_consumption': np.random.normal(40, 10, n_samples),
        'acoustic_emission': np.random.normal(80, 5, n_samples),
        'status': np.random.choice(['Normal', 'Degradation', 'Failure'], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'data_generation': {
            'total_samples': 100,
            'seed': 42,
            'normal_ratio': 0.7,
            'degradation_ratio': 0.25,
            'failure_ratio': 0.05
        },
        'model_training': {
            'test_size': 0.2,
            'validation_size': 0.1,
            'failure_predictor': {'type': 'xgboost'}
        },
        'paths': {
            'data': {
                'synthetic': 'data/synthetic',
                'raw': 'data/raw',
                'processed': 'data/processed'
            },
            'models': {
                'saved': 'models/saved_models',
                'optimized': 'models/edge_optimized'
            },
            'logs': 'logs'
        }
    }


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary test directory"""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir
