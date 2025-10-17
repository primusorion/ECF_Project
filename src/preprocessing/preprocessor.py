"""
Data Preprocessing Module for Predictive Maintenance
Handles data cleaning, feature engineering, and preparation for ML models
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os


class DataPreprocessor:
    """
    Preprocesses sensor data for machine learning models
    Includes feature engineering, scaling, and data splitting
    """
    
    def __init__(self):
        """Initialize preprocessor with scalers and encoders"""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_fitted = False
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(
            df_clean[numeric_columns].median()
        )
        
        # Handle categorical missing values
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Remove extreme outliers (beyond 4 standard deviations)
        for col in numeric_columns:
            if col not in ['failure_within_24h', 'is_weekend', 'hour_of_day', 'day_of_week']:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                df_clean = df_clean[
                    (df_clean[col] >= mean - 4*std) & 
                    (df_clean[col] <= mean + 4*std)
                ]
        
        return df_clean.reset_index(drop=True)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw sensor data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df_eng = df.copy()
        
        # Temperature-related features
        df_eng['temp_deviation'] = np.abs(
            df_eng['temperature'] - df_eng.groupby('equipment_type')['temperature'].transform('mean')
        )
        df_eng['temp_rate_of_change'] = df_eng.groupby('equipment_id')['temperature'].diff().fillna(0)
        
        # Vibration-related features
        df_eng['vibration_squared'] = df_eng['vibration'] ** 2
        df_eng['vibration_rate_of_change'] = df_eng.groupby('equipment_id')['vibration'].diff().fillna(0)
        
        # Pressure-related features
        df_eng['pressure_deviation'] = np.abs(
            df_eng['pressure'] - df_eng.groupby('equipment_type')['pressure'].transform('mean')
        )
        
        # Power efficiency
        df_eng['power_efficiency'] = df_eng['power_consumption'] / (df_eng['rpm'] + 1)
        
        # Temperature-Vibration interaction
        df_eng['temp_vibration_interaction'] = df_eng['temperature'] * df_eng['vibration']
        
        # Rolling statistics (if timestamp is available)
        if 'timestamp' in df_eng.columns:
            df_eng = df_eng.sort_values(['equipment_id', 'timestamp'])
            
            # Rolling mean (last 5 readings)
            for col in ['temperature', 'vibration', 'pressure', 'rpm']:
                df_eng[f'{col}_rolling_mean'] = df_eng.groupby('equipment_id')[col].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                )
                
                df_eng[f'{col}_rolling_std'] = df_eng.groupby('equipment_id')[col].transform(
                    lambda x: x.rolling(window=5, min_periods=1).std()
                ).fillna(0)
        
        # Operating condition score
        df_eng['operating_condition_score'] = (
            (df_eng['temperature'] / 100) * 0.25 +
            (df_eng['vibration'] / 10) * 0.25 +
            (df_eng['acoustic_emission'] / 100) * 0.25 +
            (df_eng['power_consumption'] / 100) * 0.25
        )
        
        # Equipment age indicator (based on remaining useful life)
        if 'remaining_useful_life' in df_eng.columns:
            max_rul = df_eng['remaining_useful_life'].max()
            df_eng['equipment_age_factor'] = 1 - (df_eng['remaining_useful_life'] / max_rul)
        
        return df_eng
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        categorical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                # Use one-hot encoding for equipment_type
                if col == 'equipment_type':
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded.drop(col, axis=1, inplace=True)
                # Use label encoding for status
                elif col == 'status':
                    if not self.is_fitted:
                        df_encoded[f'{col}_encoded'] = self.label_encoder.fit_transform(df_encoded[col])
                    else:
                        df_encoded[f'{col}_encoded'] = self.label_encoder.transform(df_encoded[col])
        
        return df_encoded
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_column: str = 'failure_within_24h',
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            exclude_columns: Columns to exclude from features
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if exclude_columns is None:
            exclude_columns = [
                'timestamp', 'equipment_id', 'status', 
                'failure_within_24h', 'remaining_useful_life'
            ]
        
        # Separate features and target
        feature_cols = [col for col in df.columns 
                       if col not in exclude_columns and col != target_column]
        
        X = df[feature_cols].copy()
        y = df[target_column].copy() if target_column in df.columns else None
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        return X, y
    
    def scale_features(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> np.ndarray:
        """
        Scale features using StandardScaler
        
        Args:
            X: Features DataFrame
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Scaled features as numpy array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transform. Set fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
               pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            val_size: Proportion of validation set from training data
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate validation set from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str = 'failure_within_24h',
        test_size: float = 0.2,
        val_size: float = 0.1,
        fit: bool = True
    ) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            test_size: Test set proportion
            val_size: Validation set proportion
            fit: Whether to fit scalers
            
        Returns:
            Dictionary containing processed datasets
        """
        print("Step 1: Cleaning data...")
        df_clean = self.clean_data(df)
        
        print("Step 2: Engineering features...")
        df_eng = self.engineer_features(df_clean)
        
        print("Step 3: Encoding categorical variables...")
        categorical_cols = ['equipment_type', 'status']
        df_encoded = self.encode_categorical(df_eng, categorical_cols)
        
        print("Step 4: Preparing features and target...")
        X, y = self.prepare_features(df_encoded, target_column)
        
        if fit:
            print("Step 5: Splitting data...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
                X, y, test_size, val_size
            )
            
            print("Step 6: Scaling features...")
            X_train_scaled = self.scale_features(X_train, fit=True)
            X_val_scaled = self.scale_features(X_val, fit=False)
            X_test_scaled = self.scale_features(X_test, fit=False)
            
            return {
                'X_train': X_train_scaled,
                'X_val': X_val_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'feature_names': self.feature_columns
            }
        else:
            print("Step 5: Scaling features...")
            X_scaled = self.scale_features(X, fit=False)
            
            return {
                'X': X_scaled,
                'y': y,
                'feature_names': self.feature_columns
            }
    
    def save(self, filepath: str):
        """Save preprocessor state"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Preprocessor saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.feature_columns = data['feature_columns']
            self.is_fitted = data['is_fitted']
        print(f"Preprocessor loaded from: {filepath}")


def main():
    """Test preprocessing pipeline"""
    # Load sample data
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', '..', 'data', 'synthetic', 'sensor_data.csv'
    )
    
    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Run preprocessing pipeline
        result = preprocessor.preprocess_pipeline(df)
        
        print("\n=== Preprocessing Results ===")
        print(f"Training samples: {result['X_train'].shape[0]}")
        print(f"Validation samples: {result['X_val'].shape[0]}")
        print(f"Test samples: {result['X_test'].shape[0]}")
        print(f"Number of features: {result['X_train'].shape[1]}")
        print(f"\nFeature names: {result['feature_names'][:10]}...")
        
        # Save preprocessor
        save_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'models', 'preprocessor.pkl'
        )
        preprocessor.save(save_path)
    else:
        print(f"Data file not found: {data_path}")
        print("Please run data generation first.")


if __name__ == '__main__':
    main()
