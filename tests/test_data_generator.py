"""
Unit tests for data_generator module
"""
import pytest
import pandas as pd
import numpy as np
from src.data_generation.data_generator import EquipmentDataGenerator


class TestEquipmentDataGenerator:
    """Test cases for EquipmentDataGenerator"""
    
    def test_initialization(self):
        """Test generator initialization"""
        generator = EquipmentDataGenerator(seed=42)
        assert generator.seed == 42
        assert generator.equipment_types == ['Motor', 'Pump', 'Compressor', 'Conveyor']
    
    def test_generate_normal_data(self):
        """Test normal equipment data generation"""
        generator = EquipmentDataGenerator(seed=42)
        df = generator.generate_normal_data(n_samples=100)
        
        assert len(df) == 100
        assert 'equipment_id' in df.columns
        assert 'equipment_type' in df.columns
        assert 'temperature' in df.columns
        assert 'vibration' in df.columns
        assert 'status' in df.columns
        assert all(df['status'] == 'Normal')
        
        # Check reasonable value ranges
        assert df['temperature'].mean() < 80
        assert df['vibration'].mean() < 5
    
    def test_generate_degradation_data(self):
        """Test degradation data generation"""
        generator = EquipmentDataGenerator(seed=42)
        df = generator.generate_degradation_data(n_samples=50)
        
        assert len(df) == 50
        assert all(df['status'] == 'Degradation')
        
        # Degradation should have higher values
        assert df['temperature'].mean() > 70
        assert df['vibration'].mean() > 4
    
    def test_generate_failure_data(self):
        """Test failure data generation"""
        generator = EquipmentDataGenerator(seed=42)
        df = generator.generate_failure_data(n_samples=10)
        
        assert len(df) == 10
        assert all(df['status'] == 'Failure')
        
        # Failure should have critical values
        assert df['temperature'].max() > 90
        assert df['vibration'].max() > 7
    
    def test_generate_mixed_dataset(self):
        """Test mixed dataset generation"""
        generator = EquipmentDataGenerator(seed=42)
        df = generator.generate_mixed_dataset(
            total_samples=1000,
            normal_ratio=0.7,
            degradation_ratio=0.25,
            failure_ratio=0.05
        )
        
        assert len(df) == 1000
        
        # Check status distribution
        status_counts = df['status'].value_counts()
        assert status_counts['Normal'] == 700
        assert status_counts['Degradation'] == 250
        assert status_counts['Failure'] == 50
    
    def test_output_file_creation(self, temp_test_dir):
        """Test data file output"""
        generator = EquipmentDataGenerator(seed=42)
        output_path = temp_test_dir / "test_sensor_data.csv"
        
        df = generator.generate_mixed_dataset(
            total_samples=100,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 100
    
    def test_reproducibility(self):
        """Test that same seed produces same results"""
        gen1 = EquipmentDataGenerator(seed=42)
        gen2 = EquipmentDataGenerator(seed=42)
        
        df1 = gen1.generate_normal_data(n_samples=50)
        df2 = gen2.generate_normal_data(n_samples=50)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_column_types(self):
        """Test that generated data has correct column types"""
        generator = EquipmentDataGenerator(seed=42)
        df = generator.generate_normal_data(n_samples=10)
        
        assert df['equipment_id'].dtype == object
        assert df['equipment_type'].dtype == object
        assert df['temperature'].dtype in [np.float64, float]
        assert df['vibration'].dtype in [np.float64, float]
        assert df['status'].dtype == object
