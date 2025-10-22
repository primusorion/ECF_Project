"""
Advanced Feature Engineering Module
Implements rolling statistics, FFT, wavelets, and time-based features
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import pywavelets as pywt
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for predictive maintenance
    Implements signal processing and time-series features
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names = []
    
    def add_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Add rolling window statistics
        
        Args:
            df: Input dataframe
            columns: Columns to compute statistics for
            windows: Window sizes for rolling calculations
            
        Returns:
            DataFrame with additional rolling statistic columns
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                
                # Rolling min
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                
                # Rolling max
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                
                # Rolling range
                df[f'{col}_rolling_range_{window}'] = df[f'{col}_rolling_max_{window}'] - df[f'{col}_rolling_min_{window}']
        
        return df
    
    def add_fft_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_features: int = 5
    ) -> pd.DataFrame:
        """
        Add FFT (Fast Fourier Transform) features for frequency analysis
        Useful for vibration and acoustic emission signals
        
        Args:
            df: Input dataframe
            columns: Columns to compute FFT for
            n_features: Number of top frequency features to extract
            
        Returns:
            DataFrame with FFT features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Compute FFT
            signal_data = df[col].values
            fft_vals = np.abs(fft(signal_data))
            freqs = fftfreq(len(signal_data))
            
            # Get top N frequency components
            top_indices = np.argsort(fft_vals)[-n_features:]
            
            for i, idx in enumerate(top_indices):
                df[f'{col}_fft_mag_{i+1}'] = fft_vals[idx]
                df[f'{col}_fft_freq_{i+1}'] = abs(freqs[idx])
            
            # Spectral energy
            df[f'{col}_spectral_energy'] = np.sum(fft_vals**2)
            
            # Spectral centroid
            df[f'{col}_spectral_centroid'] = np.sum(freqs * fft_vals) / np.sum(fft_vals)
        
        return df
    
    def add_wavelet_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        wavelet: str = 'db4',
        level: int = 3
    ) -> pd.DataFrame:
        """
        Add wavelet transform features
        Useful for detecting transient events and patterns
        
        Args:
            df: Input dataframe
            columns: Columns to compute wavelets for
            wavelet: Wavelet type (e.g., 'db4', 'sym4', 'coif3')
            level: Decomposition level
            
        Returns:
            DataFrame with wavelet features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            signal_data = df[col].values
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)
            
            # Extract features from each level
            for i, coeff in enumerate(coeffs):
                df[f'{col}_wavelet_energy_l{i}'] = np.sum(coeff**2)
                df[f'{col}_wavelet_mean_l{i}'] = np.mean(coeff)
                df[f'{col}_wavelet_std_l{i}'] = np.std(coeff)
                df[f'{col}_wavelet_max_l{i}'] = np.max(np.abs(coeff))
        
        return df
    
    def add_time_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Add time-based features
        
        Args:
            df: Input dataframe
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with time features
        """
        df = df.copy()
        
        if timestamp_col not in df.columns:
            return df
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract time components
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Shift-based features (assuming 3 shifts: morning, afternoon, night)
        df['shift'] = pd.cut(df['hour'], bins=[0, 8, 16, 24], labels=['night', 'morning', 'afternoon'], include_lowest=True)
        df['shift'] = df['shift'].astype(str)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Add lagged features (previous values)
        
        Args:
            df: Input dataframe
            columns: Columns to create lags for
            lags: Lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_rate_of_change(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = [1, 5, 10]
    ) -> pd.DataFrame:
        """
        Add rate of change features
        
        Args:
            df: Input dataframe
            columns: Columns to compute rate of change
            periods: Periods for rate calculation
            
        Returns:
            DataFrame with rate of change features
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for period in periods:
                df[f'{col}_roc_{period}'] = df[col].pct_change(periods=period)
                df[f'{col}_diff_{period}'] = df[col].diff(periods=period)
        
        return df
    
    def add_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[tuple]
    ) -> pd.DataFrame:
        """
        Add interaction features (products and ratios)
        
        Args:
            df: Input dataframe
            feature_pairs: List of (feature1, feature2) tuples
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all advanced features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with all advanced features
        """
        sensor_columns = ['temperature', 'vibration', 'pressure', 'rpm',
                         'power_consumption', 'acoustic_emission']
        
        # Remove columns that don't exist
        sensor_columns = [col for col in sensor_columns if col in df.columns]
        
        print("Adding rolling statistics...")
        df = self.add_rolling_statistics(df, sensor_columns, windows=[5, 10])
        
        print("Adding time features...")
        if 'timestamp' in df.columns:
            df = self.add_time_features(df)
        
        print("Adding lag features...")
        df = self.add_lag_features(df, sensor_columns, lags=[1, 2, 3])
        
        print("Adding rate of change...")
        df = self.add_rate_of_change(df, sensor_columns, periods=[1, 5])
        
        print("Adding interaction features...")
        interactions = [
            ('temperature', 'vibration'),
            ('power_consumption', 'rpm'),
            ('pressure', 'temperature')
        ]
        df = self.add_interaction_features(df, interactions)
        
        # Fill NaN values created by rolling/lag operations
        df = df.fillna(method='bfill').fillna(0)
        
        print(f"Total features created: {len(df.columns)}")
        
        return df


# Example usage
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'equipment_id': 'Motor_001',
        'temperature': 70 + 10 * np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.randn(n_samples) * 2,
        'vibration': 4 + 2 * np.sin(np.linspace(0, 20*np.pi, n_samples)) + np.random.randn(n_samples),
        'pressure': 7 + np.random.randn(n_samples) * 0.5,
        'rpm': 1800 + np.random.randn(n_samples) * 50,
        'power_consumption': 40 + np.random.randn(n_samples) * 5,
        'acoustic_emission': 80 + np.random.randn(n_samples) * 3
    })
    
    # Create advanced features
    engineer = AdvancedFeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print(f"\nOriginal columns: {len(['temperature', 'vibration', 'pressure', 'rpm', 'power_consumption', 'acoustic_emission'])}")
    print(f"Total columns after feature engineering: {len(df_features.columns)}")
    print(f"\nSample columns:")
    print(df_features.columns.tolist()[:20])
