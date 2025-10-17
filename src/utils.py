"""
Utility Functions for Predictive Maintenance System
"""

import os
import yaml
import json
import logging
from datetime import datetime
from typing import Dict, Any
import colorlog


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'config', 'config.yaml'
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logger(
    name: str = 'predictive_maintenance',
    log_file: str = None,
    level: str = 'INFO'
) -> logging.Logger:
    """
    Setup logger with colored console output and file logging
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    console_format = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directories(config: Dict[str, Any]):
    """
    Create necessary directories if they don't exist
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    
    directories = [
        paths.get('data', {}).get('raw', 'data/raw'),
        paths.get('data', {}).get('synthetic', 'data/synthetic'),
        paths.get('data', {}).get('processed', 'data/processed'),
        paths.get('models', {}).get('saved', 'models/saved_models'),
        paths.get('models', {}).get('optimized', 'models/edge_optimized'),
        paths.get('logs', 'logs'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_json(data: Dict, filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Dict:
    """
    Load JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def format_timestamp(timestamp: datetime = None, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format timestamp to string
    
    Args:
        timestamp: Datetime object (default: now)
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime(format_str)


def get_model_size(filepath: str) -> float:
    """
    Get model file size in MB
    
    Args:
        filepath: Path to model file
        
    Returns:
        File size in MB
    """
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    return 0.0


def print_banner(text: str, char: str = '='):
    """
    Print a formatted banner
    
    Args:
        text: Text to display
        char: Character to use for banner
    """
    width = max(60, len(text) + 10)
    print('\n' + char * width)
    print(f'{text:^{width}}')
    print(char * width + '\n')


def format_alert_message(
    equipment_id: str,
    alert_level: str,
    failure_probability: float,
    sensor_data: Dict
) -> str:
    """
    Format alert message for notification
    
    Args:
        equipment_id: Equipment identifier
        alert_level: Alert level (LOW, MEDIUM, HIGH, CRITICAL)
        failure_probability: Predicted failure probability
        sensor_data: Current sensor readings
        
    Returns:
        Formatted alert message
    """
    timestamp = format_timestamp()
    
    message = f"""
╔════════════════════════════════════════════════════════════╗
║           PREDICTIVE MAINTENANCE ALERT                     ║
╠════════════════════════════════════════════════════════════╣
║ Equipment ID:     {equipment_id:<40} ║
║ Alert Level:      {alert_level:<40} ║
║ Failure Prob:     {failure_probability:.1%:<40} ║
║ Timestamp:        {timestamp:<40} ║
╠════════════════════════════════════════════════════════════╣
║ Current Sensor Readings:                                   ║
║   Temperature:    {sensor_data.get('temperature', 'N/A'):<8} °C                              ║
║   Vibration:      {sensor_data.get('vibration', 'N/A'):<8} mm/s                            ║
║   Pressure:       {sensor_data.get('pressure', 'N/A'):<8} Bar                              ║
║   RPM:            {sensor_data.get('rpm', 'N/A'):<8}                                  ║
║   Power:          {sensor_data.get('power_consumption', 'N/A'):<8} kW                               ║
╚════════════════════════════════════════════════════════════╝
"""
    
    return message


def calculate_metrics_summary(metrics: Dict) -> str:
    """
    Create a formatted summary of model metrics
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted metrics summary
    """
    summary = f"""
Model Performance Summary:
{'='*50}
Accuracy:     {metrics.get('accuracy', 0):.4f}
Precision:    {metrics.get('precision', 0):.4f}
Recall:       {metrics.get('recall', 0):.4f}
F1-Score:     {metrics.get('f1_score', 0):.4f}
ROC-AUC:      {metrics.get('roc_auc', 0):.4f}
{'='*50}
"""
    return summary


class PerformanceTimer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str = 'Operation', logger: logging.Logger = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        if self.logger:
            self.logger.info(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if self.logger:
            self.logger.info(f"{self.name} completed in {duration:.2f} seconds")
        else:
            print(f"{self.name} completed in {duration:.2f} seconds")
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


def main():
    """Test utilities"""
    # Load config
    config = load_config()
    print("Configuration loaded:")
    print(json.dumps(config, indent=2))
    
    # Setup logger
    logger = setup_logger()
    logger.info("Logger initialized")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Create directories
    ensure_directories(config)
    logger.info("Directories created")
    
    # Test timer
    with PerformanceTimer("Test Operation", logger):
        import time
        time.sleep(1)


if __name__ == '__main__':
    main()
