"""
Demo Script for Predictive Maintenance System
Demonstrates key features without requiring a notebook
"""

import os
import sys
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation.data_generator import EquipmentDataGenerator
from src.preprocessing.preprocessor import DataPreprocessor
from src.utils import print_banner


def demo_data_generation():
    """Demonstrate data generation"""
    print_banner("DEMO: DATA GENERATION")
    
    print("Creating Equipment Data Generator...")
    generator = EquipmentDataGenerator(seed=42)
    
    print("\n1. Generating Normal Operation Data...")
    normal_data = generator.generate_normal_operation('Motor', n_samples=100)
    print(f"   Generated {len(normal_data)} samples")
    print("\n   Sample data:")
    print(normal_data[['temperature', 'vibration', 'pressure', 'rpm', 'status']].head())
    
    print("\n2. Generating Degradation Pattern...")
    degradation_data = generator.generate_degradation_pattern('Pump', n_samples=100)
    print(f"   Generated {len(degradation_data)} samples")
    print("\n   Notice increasing sensor values:")
    print(degradation_data[['temperature', 'vibration', 'failure_within_24h']].tail())
    
    print("\n3. Generating Failure Scenario...")
    failure_data = generator.generate_failure_scenario('Compressor', n_samples=50)
    print(f"   Generated {len(failure_data)} samples")
    print("\n   Extreme values indicating failure:")
    print(failure_data[['temperature', 'vibration', 'pressure', 'status']].head())
    
    print("\n✓ Data generation demo complete!")
    return normal_data, degradation_data, failure_data


def demo_preprocessing():
    """Demonstrate preprocessing"""
    print_banner("DEMO: DATA PREPROCESSING")
    
    # Generate sample data
    generator = EquipmentDataGenerator(seed=42)
    df = generator.generate_mixed_dataset(total_samples=1000)
    
    print(f"Original data: {df.shape}")
    print("\nOriginal columns:")
    print(df.columns.tolist())
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    print("\n1. Cleaning data...")
    df_clean = preprocessor.clean_data(df)
    print(f"   Cleaned data: {df_clean.shape}")
    
    print("\n2. Engineering features...")
    df_eng = preprocessor.engineer_features(df_clean)
    print(f"   Data with engineered features: {df_eng.shape}")
    print("\n   New features created:")
    new_features = [col for col in df_eng.columns if col not in df.columns]
    print(f"   {new_features[:10]}...")
    
    print("\n3. Encoding categorical variables...")
    df_encoded = preprocessor.encode_categorical(df_eng, ['equipment_type', 'status'])
    print(f"   Encoded data: {df_encoded.shape}")
    
    print("\n✓ Preprocessing demo complete!")
    return df_encoded


def demo_sensor_analysis():
    """Analyze sensor patterns"""
    print_banner("DEMO: SENSOR ANALYSIS")
    
    generator = EquipmentDataGenerator(seed=42)
    df = generator.generate_mixed_dataset(total_samples=1000)
    
    print("Analyzing sensor patterns by equipment status...\n")
    
    # Group by status
    status_groups = df.groupby('status')
    
    for status, group in status_groups:
        print(f"\n{status.upper()} STATUS:")
        print("-" * 50)
        
        stats = group[['temperature', 'vibration', 'pressure', 'rpm']].describe()
        print(stats.loc[['mean', 'std', 'min', 'max']])
    
    # Alert conditions
    print("\n\nALERT CONDITIONS:")
    print("-" * 50)
    critical_temp = df[df['temperature'] > 95]
    critical_vib = df[df['vibration'] > 8]
    
    print(f"High Temperature (>95°C): {len(critical_temp)} samples ({len(critical_temp)/len(df)*100:.1f}%)")
    print(f"High Vibration (>8 mm/s): {len(critical_vib)} samples ({len(critical_vib)/len(df)*100:.1f}%)")
    
    print("\n✓ Sensor analysis demo complete!")


def demo_edge_inference_simulation():
    """Simulate edge inference without trained models"""
    print_banner("DEMO: EDGE INFERENCE SIMULATION")
    
    print("Simulating edge device inference...\n")
    
    # Sample sensor readings
    scenarios = [
        {
            'name': 'Normal Operation',
            'data': {
                'equipment_id': 'Motor_001',
                'temperature': 62.5,
                'vibration': 2.3,
                'pressure': 7.5,
                'rpm': 1800,
                'power_consumption': 30.2,
            },
            'expected_failure_prob': 0.15
        },
        {
            'name': 'Degrading Equipment',
            'data': {
                'equipment_id': 'Pump_002',
                'temperature': 85.2,
                'vibration': 6.8,
                'pressure': 7.1,
                'rpm': 1550,
                'power_consumption': 48.5,
            },
            'expected_failure_prob': 0.75
        },
        {
            'name': 'Critical Failure',
            'data': {
                'equipment_id': 'Compressor_003',
                'temperature': 98.5,
                'vibration': 9.2,
                'pressure': 3.2,
                'rpm': 450,
                'power_consumption': 72.3,
            },
            'expected_failure_prob': 0.95
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")
        print(f"Equipment ID: {scenario['data']['equipment_id']}")
        print(f"\nSensor Readings:")
        for key, value in scenario['data'].items():
            if key != 'equipment_id':
                print(f"  {key:20s}: {value}")
        
        # Determine alert level
        failure_prob = scenario['expected_failure_prob']
        if failure_prob >= 0.9:
            alert_level = "🔴 CRITICAL"
            message = "Immediate maintenance required!"
        elif failure_prob >= 0.7:
            alert_level = "🟠 HIGH"
            message = "Schedule maintenance within 24 hours"
        elif failure_prob >= 0.4:
            alert_level = "🟡 MEDIUM"
            message = "Monitor closely"
        else:
            alert_level = "🟢 LOW"
            message = "Equipment operating normally"
        
        print(f"\nPrediction:")
        print(f"  Failure Probability: {failure_prob:.1%}")
        print(f"  Alert Level: {alert_level}")
        print(f"  Recommendation: {message}")
    
    print("\n\n✓ Edge inference simulation complete!")


def demo_summary():
    """Print summary and next steps"""
    print_banner("DEMO SUMMARY")
    
    print("""
This demo showcased:

1. ✓ Synthetic Data Generation
   - Normal operation patterns
   - Equipment degradation
   - Failure scenarios

2. ✓ Data Preprocessing
   - Data cleaning
   - Feature engineering
   - Categorical encoding

3. ✓ Sensor Analysis
   - Statistical patterns
   - Alert conditions
   - Status distributions

4. ✓ Edge Inference Simulation
   - Real-time prediction scenarios
   - Alert generation
   - Risk assessment

NEXT STEPS:
""")
    
    print("To run the full system:")
    print("  python main.py --mode full")
    
    print("\nTo train models on real data:")
    print("  python main.py --mode train")
    
    print("\nTo launch monitoring dashboard:")
    print("  python main.py --mode monitor")
    
    print("\nFor more information:")
    print("  - README.md - Full documentation")
    print("  - GETTING_STARTED.md - Quick start guide")
    print("  - config/config.yaml - Configuration options")
    
    print("\n" + "="*60)
    print("Thank you for exploring the Predictive Maintenance System!")
    print("="*60 + "\n")


def main():
    """Run all demos"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║        PREDICTIVE MAINTENANCE SYSTEM - DEMO                      ║
    ║        Interactive Demo of Key Features                          ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Run demos
        demo_data_generation()
        input("\nPress Enter to continue to preprocessing demo...")
        
        demo_preprocessing()
        input("\nPress Enter to continue to sensor analysis...")
        
        demo_sensor_analysis()
        input("\nPress Enter to continue to inference simulation...")
        
        demo_edge_inference_simulation()
        input("\nPress Enter to see summary...")
        
        demo_summary()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
