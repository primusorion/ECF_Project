"""
Main Application Entry Point for Predictive Maintenance System
Orchestrates all modules: data generation, training, inference, and monitoring
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import (
    load_config, 
    setup_logger, 
    ensure_directories,
    print_banner,
    PerformanceTimer
)


def run_data_generation(args, config, logger):
    """Generate synthetic sensor data"""
    from src.data_generation.data_generator import EquipmentDataGenerator
    
    print_banner("DATA GENERATION")
    logger.info("Starting data generation...")
    
    generator = EquipmentDataGenerator(seed=config['data_generation']['seed'])
    
    output_path = os.path.join(
        config['paths']['data']['synthetic'],
        'sensor_data.csv'
    )
    
    n_samples = args.samples or config['data_generation']['total_samples']
    
    with PerformanceTimer("Data Generation", logger):
        df = generator.generate_mixed_dataset(
            total_samples=n_samples,
            normal_ratio=config['data_generation']['normal_ratio'],
            degradation_ratio=config['data_generation']['degradation_ratio'],
            failure_ratio=config['data_generation']['failure_ratio'],
            output_path=output_path
        )
    
    logger.info(f"[OK] Generated {len(df)} samples")
    logger.info(f"[OK] Data saved to: {output_path}")


def run_training(args, config, logger):
    """Train machine learning models"""
    from src.preprocessing.preprocessor import DataPreprocessor
    from src.models.model_trainer import train_all_models
    import pandas as pd
    
    print_banner("MODEL TRAINING")
    logger.info("Starting model training...")
    
    # Load data
    data_path = args.data or os.path.join(
        config['paths']['data']['synthetic'],
        'sensor_data.csv'
    )
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run data generation first: python main.py --mode generate")
        return
    
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    with PerformanceTimer("Data Preprocessing", logger):
        data = preprocessor.preprocess_pipeline(
            df,
            test_size=config['model_training']['test_size'],
            val_size=config['model_training']['validation_size']
        )
    
    # Save preprocessor
    preprocessor_path = os.path.join(
        os.path.dirname(config['paths']['models']['saved']),
        'preprocessor.pkl'
    )
    preprocessor.save(preprocessor_path)
    
    # Train models
    logger.info("Training models...")
    models_dir = config['paths']['models']['saved']
    model_type = config['model_training']['failure_predictor']['type']
    
    with PerformanceTimer("Model Training", logger):
        results = train_all_models(data, models_dir, model_type)
    
    logger.info("[OK] Training complete!")


def run_inference(args, config, logger):
    """Run edge inference on sensor data"""
    from src.edge_inference.inference_engine import EdgeInferenceEngine
    import json
    
    print_banner("EDGE INFERENCE")
    logger.info("Starting edge inference...")
    
    # Initialize engine
    models_dir = config['paths']['models']['saved']
    engine = EdgeInferenceEngine(models_dir)
    
    try:
        engine.load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.info("Please train models first: python main.py --mode train")
        return
    
    # Sample sensor data
    sample_data = {
        'equipment_id': 'Motor_001',
        'equipment_type': 'Motor',
        'temperature': 85.5,
        'vibration': 6.2,
        'pressure': 7.8,
        'rpm': 1750,
        'power_consumption': 42.5,
        'acoustic_emission': 82.3
    }
    
    logger.info("Running inference on sample data...")
    logger.info(f"Input: {json.dumps(sample_data, indent=2)}")
    
    result = engine.predict_single(sample_data)
    
    logger.info("\n" + "="*60)
    logger.info("PREDICTION RESULTS:")
    logger.info("="*60)
    logger.info(json.dumps(result, indent=2))
    logger.info("="*60)
    
    # Performance stats
    stats = engine.get_performance_stats()
    logger.info(f"\nInference Time: {stats['avg_inference_time_ms']:.2f} ms")


def run_monitoring(args, config, logger):
    """Launch monitoring dashboard"""
    from src.monitoring.dashboard import MonitoringDashboard
    
    print_banner("MONITORING DASHBOARD")
    logger.info("Launching monitoring dashboard...")
    
    port = args.port or config['monitoring']['port']
    
    dashboard = MonitoringDashboard(port=port)
    dashboard.run(debug=config['monitoring']['debug'])


def run_full_pipeline(args, config, logger):
    """Run the complete pipeline"""
    print_banner("FULL PIPELINE EXECUTION")
    logger.info("Running complete predictive maintenance pipeline...")
    
    # Step 1: Generate data
    logger.info("\n>>> Step 1/3: Data Generation")
    run_data_generation(args, config, logger)
    
    # Step 2: Train models
    logger.info("\n>>> Step 2/3: Model Training")
    run_training(args, config, logger)
    
    # Step 3: Run inference demo
    logger.info("\n>>> Step 3/3: Inference Demo")
    run_inference(args, config, logger)
    
    logger.info("\n" + "="*60)
    logger.info("[OK] Full pipeline completed successfully!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("  - Launch dashboard: python main.py --mode monitor")
    logger.info("  - View data: check data/synthetic/sensor_data.csv")
    logger.info("  - View models: check models/saved_models/")


def main():
    """Main application entry point"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Predictive Maintenance of Factory Equipment Using Edge AI'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['generate', 'train', 'infer', 'monitor', 'full'],
        default='full',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        help='Number of samples to generate (for generate mode)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to data file (for train mode)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (for infer mode)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Port for monitoring dashboard'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration...")
        config = {
            'data_generation': {'total_samples': 10000, 'seed': 42, 
                              'normal_ratio': 0.7, 'degradation_ratio': 0.25, 
                              'failure_ratio': 0.05},
            'model_training': {'test_size': 0.2, 'validation_size': 0.1,
                             'failure_predictor': {'type': 'xgboost'}},
            'monitoring': {'port': 8050, 'debug': False},
            'paths': {
                'data': {'synthetic': 'data/synthetic', 'raw': 'data/raw', 
                        'processed': 'data/processed'},
                'models': {'saved': 'models/saved_models', 
                          'optimized': 'models/edge_optimized'},
                'logs': 'logs'
            },
            'logging': {'level': 'INFO', 'file': 'logs/predictive_maintenance.log'}
        }
    
    # Setup directories
    ensure_directories(config)
    
    # Setup logger
    logger = setup_logger(
        name='predictive_maintenance',
        log_file=config['logging']['file'],
        level=config['logging']['level']
    )
    
    # ASCII Art Banner
    print("\n" + "="*70)
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║        PREDICTIVE MAINTENANCE SYSTEM - EDGE AI                   ║
    ║        Factory Equipment Monitoring & Failure Prediction         ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    print("="*70 + "\n")
    
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Configuration loaded from: {args.config}")
    
    # Route to appropriate function
    try:
        if args.mode == 'generate':
            run_data_generation(args, config, logger)
        elif args.mode == 'train':
            run_training(args, config, logger)
        elif args.mode == 'infer':
            run_inference(args, config, logger)
        elif args.mode == 'monitor':
            run_monitoring(args, config, logger)
        elif args.mode == 'full':
            run_full_pipeline(args, config, logger)
        
    except KeyboardInterrupt:
        logger.info("\n\nOperation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
