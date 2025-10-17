"""
Data Generation Module for Predictive Maintenance
Generates synthetic sensor data for factory equipment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional
import os


class EquipmentDataGenerator:
    """
    Generates synthetic sensor data for factory equipment
    Simulates normal operation, degradation, and failure scenarios
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
        
        # Equipment types and their characteristics
        self.equipment_types = {
            'Motor': {
                'temperature_base': 60,
                'vibration_base': 2.5,
                'pressure_base': 7.5,
                'rpm_base': 1800,
                'power_base': 30,
                'acoustic_base': 75
            },
            'Pump': {
                'temperature_base': 55,
                'vibration_base': 3.0,
                'pressure_base': 8.0,
                'rpm_base': 1500,
                'power_base': 25,
                'acoustic_base': 70
            },
            'Compressor': {
                'temperature_base': 70,
                'vibration_base': 2.0,
                'pressure_base': 9.0,
                'rpm_base': 2500,
                'power_base': 45,
                'acoustic_base': 85
            },
            'Conveyor': {
                'temperature_base': 45,
                'vibration_base': 1.5,
                'pressure_base': 6.0,
                'rpm_base': 1200,
                'power_base': 20,
                'acoustic_base': 65
            }
        }
        
    def generate_normal_operation(
        self, 
        equipment_type: str,
        n_samples: int,
        start_time: Optional[datetime] = None,
        sampling_interval: int = 60  # seconds
    ) -> pd.DataFrame:
        """
        Generate data for normal equipment operation
        
        Args:
            equipment_type: Type of equipment
            n_samples: Number of samples to generate
            start_time: Starting timestamp
            sampling_interval: Time between samples in seconds
            
        Returns:
            DataFrame with sensor readings
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
            
        params = self.equipment_types[equipment_type]
        
        # Generate timestamps
        timestamps = [start_time + timedelta(seconds=i*sampling_interval) 
                     for i in range(n_samples)]
        
        # Generate sensor readings with normal variation
        data = {
            'timestamp': timestamps,
            'equipment_id': [f'{equipment_type}_{i%5 + 1:03d}' for i in range(n_samples)],
            'equipment_type': [equipment_type] * n_samples,
            'temperature': np.random.normal(
                params['temperature_base'], 
                params['temperature_base'] * 0.1, 
                n_samples
            ),
            'vibration': np.abs(np.random.normal(
                params['vibration_base'], 
                params['vibration_base'] * 0.15, 
                n_samples
            )),
            'pressure': np.random.normal(
                params['pressure_base'], 
                params['pressure_base'] * 0.08, 
                n_samples
            ),
            'rpm': np.random.normal(
                params['rpm_base'], 
                params['rpm_base'] * 0.05, 
                n_samples
            ),
            'power_consumption': np.random.normal(
                params['power_base'], 
                params['power_base'] * 0.12, 
                n_samples
            ),
            'acoustic_emission': np.random.normal(
                params['acoustic_base'], 
                params['acoustic_base'] * 0.08, 
                n_samples
            ),
            'status': ['normal'] * n_samples,
            'failure_within_24h': [0] * n_samples,
            'remaining_useful_life': np.random.uniform(500, 1000, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def generate_degradation_pattern(
        self,
        equipment_type: str,
        n_samples: int,
        degradation_rate: float = 0.01,
        start_time: Optional[datetime] = None,
        sampling_interval: int = 60
    ) -> pd.DataFrame:
        """
        Generate data showing equipment degradation over time
        
        Args:
            equipment_type: Type of equipment
            n_samples: Number of samples
            degradation_rate: Rate of degradation per sample
            start_time: Starting timestamp
            sampling_interval: Time between samples in seconds
            
        Returns:
            DataFrame with degrading sensor readings
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=15)
            
        params = self.equipment_types[equipment_type]
        timestamps = [start_time + timedelta(seconds=i*sampling_interval) 
                     for i in range(n_samples)]
        
        # Create degradation factors
        degradation_factor = np.linspace(1.0, 1.5, n_samples)
        
        data = {
            'timestamp': timestamps,
            'equipment_id': [f'{equipment_type}_{i%5 + 1:03d}' for i in range(n_samples)],
            'equipment_type': [equipment_type] * n_samples,
            'temperature': params['temperature_base'] * degradation_factor + 
                          np.random.normal(0, 3, n_samples),
            'vibration': params['vibration_base'] * degradation_factor + 
                        np.random.normal(0, 0.5, n_samples),
            'pressure': params['pressure_base'] * (1 + (degradation_factor - 1) * 0.5) + 
                       np.random.normal(0, 0.5, n_samples),
            'rpm': params['rpm_base'] * (2 - degradation_factor * 0.3) + 
                  np.random.normal(0, 100, n_samples),
            'power_consumption': params['power_base'] * degradation_factor + 
                               np.random.normal(0, 3, n_samples),
            'acoustic_emission': params['acoustic_base'] * degradation_factor + 
                               np.random.normal(0, 5, n_samples),
            'status': ['degrading'] * n_samples,
            'failure_within_24h': [0] * (n_samples - 48) + [1] * 48,  # Last 48 hours
            'remaining_useful_life': np.linspace(500, 10, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def generate_failure_scenario(
        self,
        equipment_type: str,
        n_samples: int = 100,
        start_time: Optional[datetime] = None,
        sampling_interval: int = 60
    ) -> pd.DataFrame:
        """
        Generate data for equipment failure scenario
        
        Args:
            equipment_type: Type of equipment
            n_samples: Number of samples (typically smaller)
            start_time: Starting timestamp
            sampling_interval: Time between samples in seconds
            
        Returns:
            DataFrame with failure condition readings
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=2)
            
        params = self.equipment_types[equipment_type]
        timestamps = [start_time + timedelta(seconds=i*sampling_interval) 
                     for i in range(n_samples)]
        
        # Extreme values indicating failure
        data = {
            'timestamp': timestamps,
            'equipment_id': [f'{equipment_type}_{i%5 + 1:03d}' for i in range(n_samples)],
            'equipment_type': [equipment_type] * n_samples,
            'temperature': np.random.uniform(90, 110, n_samples),
            'vibration': np.random.uniform(8, 15, n_samples),
            'pressure': np.random.uniform(2, 4, n_samples),  # Low pressure
            'rpm': np.random.uniform(300, 600, n_samples),  # Very low RPM
            'power_consumption': np.random.uniform(60, 80, n_samples),
            'acoustic_emission': np.random.uniform(95, 110, n_samples),
            'status': ['failure'] * n_samples,
            'failure_within_24h': [1] * n_samples,
            'remaining_useful_life': [0] * n_samples
        }
        
        return pd.DataFrame(data)
    
    def generate_mixed_dataset(
        self,
        total_samples: int = 10000,
        normal_ratio: float = 0.7,
        degradation_ratio: float = 0.25,
        failure_ratio: float = 0.05,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate a mixed dataset with normal, degrading, and failure scenarios
        
        Args:
            total_samples: Total number of samples to generate
            normal_ratio: Ratio of normal operation samples
            degradation_ratio: Ratio of degradation samples
            failure_ratio: Ratio of failure samples
            output_path: Path to save the dataset (optional)
            
        Returns:
            Combined DataFrame
        """
        n_normal = int(total_samples * normal_ratio)
        n_degradation = int(total_samples * degradation_ratio)
        n_failure = int(total_samples * failure_ratio)
        
        all_data = []
        
        # Generate for each equipment type
        for equipment_type in self.equipment_types.keys():
            # Normal operation
            normal_data = self.generate_normal_operation(
                equipment_type,
                n_normal // len(self.equipment_types)
            )
            all_data.append(normal_data)
            
            # Degradation
            degradation_data = self.generate_degradation_pattern(
                equipment_type,
                n_degradation // len(self.equipment_types)
            )
            all_data.append(degradation_data)
            
            # Failure
            failure_data = self.generate_failure_scenario(
                equipment_type,
                n_failure // len(self.equipment_types)
            )
            all_data.append(failure_data)
        
        # Combine and shuffle
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Add additional features
        combined_df['hour_of_day'] = combined_df['timestamp'].dt.hour
        combined_df['day_of_week'] = combined_df['timestamp'].dt.dayofweek
        combined_df['is_weekend'] = combined_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            combined_df.to_csv(output_path, index=False)
            print(f"Dataset saved to: {output_path}")
            print(f"Total samples: {len(combined_df)}")
            print(f"\nStatus distribution:\n{combined_df['status'].value_counts()}")
        
        return combined_df


def main():
    """Generate sample dataset"""
    generator = EquipmentDataGenerator(seed=42)
    
    # Generate mixed dataset
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'synthetic')
    output_path = os.path.join(output_dir, 'sensor_data.csv')
    
    print("Generating synthetic sensor data...")
    df = generator.generate_mixed_dataset(
        total_samples=10000,
        output_path=output_path
    )
    
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nStatistical Summary:")
    print(df.describe())


if __name__ == '__main__':
    main()
