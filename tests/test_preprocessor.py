"""
Unit tests for preprocessor module
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor"""
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is not None
        assert not preprocessor.is_fitted
    
    def test_feature_engineering(self, sample_dataframe):
        """Test feature engineering"""
        preprocessor = DataPreprocessor()
        df_features = preprocessor.create_features(sample_dataframe.copy())
        
        # Check new features created
        assert 'temp_vibration_interaction' in df_features.columns
        assert 'power_efficiency' in df_features.columns
        
        # Check equipment type encoding
        assert 'equipment_type_Motor' in df_features.columns
        assert 'equipment_type_Pump' in df_features.columns
    
    def test_status_encoding(self, sample_dataframe):
        """Test status label encoding"""
        preprocessor = DataPreprocessor()
        y = preprocessor.encode_status(sample_dataframe['status'])
        
        # Check binary encoding (Normal=0, Degradation/Failure=1)
        assert y.dtype in [np.int64, np.int32, int]
        assert set(y.unique()).issubset({0, 1})
    
    def test_preprocessing_pipeline(self, sample_dataframe):
        """Test full preprocessing pipeline"""
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_pipeline(
            sample_dataframe,
            test_size=0.2,
            val_size=0.1
        )
        
        # Check all required keys present
        assert 'X_train' in data
        assert 'X_val' in data
        assert 'X_test' in data
        assert 'y_train' in data
        assert 'y_val' in data
        assert 'y_test' in data
        
        # Check shapes
        total_samples = len(sample_dataframe)
        assert len(data['X_train']) + len(data['X_val']) + len(data['X_test']) == total_samples
        
        # Check scaling was applied
        assert data['X_train'].std() != 1.0  # Should be close to 1 but not exactly
        assert preprocessor.is_fitted
    
    def test_save_and_load(self, temp_test_dir, sample_dataframe):
        """Test saving and loading preprocessor"""
        preprocessor = DataPreprocessor()
        preprocessor.preprocess_pipeline(sample_dataframe)
        
        # Save
        save_path = temp_test_dir / "test_preprocessor.pkl"
        preprocessor.save(str(save_path))
        assert save_path.exists()
        
        # Load
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load(str(save_path))
        assert new_preprocessor.is_fitted
        assert new_preprocessor.scaler is not None
    
    def test_transform_consistency(self, sample_dataframe):
        """Test that transform produces consistent results"""
        preprocessor = DataPreprocessor()
        
        # Fit on data
        data = preprocessor.preprocess_pipeline(sample_dataframe)
        
        # Transform new data
        new_sample = sample_dataframe.iloc[:10].copy()
        X_new = preprocessor.transform(new_sample)
        
        assert X_new.shape[1] == data['X_train'].shape[1]
    
    def test_handle_missing_values(self):
        """Test handling of missing values"""
        df_with_nan = pd.DataFrame({
            'equipment_type': ['Motor'] * 10,
            'temperature': [70, np.nan, 75, 80, np.nan, 72, 78, 82, 76, 74],
            'vibration': [4, 5, np.nan, 6, 5, 4, 5, 6, np.nan, 4],
            'pressure': [7] * 10,
            'rpm': [1800] * 10,
            'power_consumption': [40] * 10,
            'acoustic_emission': [80] * 10,
            'status': ['Normal'] * 10
        })
        
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_pipeline(df_with_nan)
        
        # Check no NaN values in output
        assert not np.isnan(data['X_train']).any()
