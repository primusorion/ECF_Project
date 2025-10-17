# Getting Started with Predictive Maintenance System

## Quick Start Guide

### Step 1: Installation

```bash
# Navigate to project directory
cd c:\ECF_project

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Complete System

The easiest way to get started is to run the full pipeline:

```bash
python main.py --mode full
```

This will:
1. Generate 10,000 synthetic sensor data samples
2. Train machine learning models (Anomaly Detection & Failure Prediction)
3. Run inference demo

### Step 3: Launch Monitoring Dashboard

```bash
python main.py --mode monitor
```

Then open your browser to: `http://localhost:8050`

## Individual Components

### Data Generation Only

```bash
python main.py --mode generate --samples 5000
```

### Training Only

```bash
python main.py --mode train --data data/synthetic/sensor_data.csv
```

### Inference Demo

```bash
python main.py --mode infer
```

## Project Structure Explained

```
ECF_project/
│
├── src/                          # Source code
│   ├── data_generation/          # Synthetic data generator
│   ├── preprocessing/            # Data cleaning & feature engineering
│   ├── models/                   # ML model training
│   ├── edge_inference/           # Edge deployment & inference
│   ├── monitoring/               # Dashboard & alerts
│   └── utils.py                  # Utility functions
│
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
│
├── data/                        # Data storage
│   ├── synthetic/               # Generated data
│   ├── raw/                    # Raw data (if any)
│   └── processed/              # Processed features
│
├── models/                      # Trained models
│   └── saved_models/           # Model files
│
├── logs/                        # Application logs
│
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

## What Does Each Module Do?

### 1. Data Generation (`src/data_generation/`)
- Generates synthetic sensor data for 4 equipment types:
  - Motors
  - Pumps
  - Compressors
  - Conveyors
- Simulates normal operation, degradation, and failure scenarios
- Creates realistic sensor patterns (temperature, vibration, pressure, etc.)

### 2. Preprocessing (`src/preprocessing/`)
- Cleans raw sensor data
- Engineers features from raw sensor readings
- Creates rolling statistics and interaction terms
- Scales features for ML models

### 3. Model Training (`src/models/`)
- **Anomaly Detector**: Identifies unusual equipment behavior
- **Failure Predictor**: Predicts equipment failure probability
- **RUL Estimator**: Estimates remaining useful life
- Uses ensemble methods (XGBoost, Random Forest, LightGBM)

### 4. Edge Inference (`src/edge_inference/`)
- Optimized for deployment on edge devices
- Low-latency predictions (<100ms)
- Real-time sensor data processing
- Alert generation based on thresholds

### 5. Monitoring Dashboard (`src/monitoring/`)
- Real-time visualization of equipment health
- Interactive charts and graphs
- Alert management
- Equipment status tracking

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Alert thresholds
edge_inference:
  thresholds:
    failure_probability: 0.70      # High risk threshold
    critical_failure_probability: 0.90  # Critical threshold

# Model selection
model_training:
  failure_predictor:
    type: xgboost  # Options: random_forest, xgboost, lightgbm, ensemble
```

## Understanding the Output

### Alert Levels

- **LOW**: Equipment operating normally (green)
- **MEDIUM**: Anomalies detected, monitor closely (yellow)
- **HIGH**: High failure probability, schedule maintenance (orange)
- **CRITICAL**: Imminent failure, immediate action required (red)

### Key Metrics

- **Failure Probability**: 0-100% chance of failure within 24-72 hours
- **Anomaly Score**: Lower values indicate more unusual behavior
- **Remaining Useful Life (RUL)**: Estimated hours until maintenance needed

## Troubleshooting

### Import Errors

If you get import errors, make sure you've installed all dependencies:

```bash
pip install -r requirements.txt
```

### No Models Found

If inference fails with "models not found", train models first:

```bash
python main.py --mode train
```

### Dashboard Not Loading

- Check if port 8050 is available
- Try a different port: `python main.py --mode monitor --port 8080`
- Check firewall settings

## Next Steps

1. **Customize Equipment Types**: Edit `data_generator.py` to add your equipment
2. **Integrate Real Sensors**: Replace synthetic data with real IoT sensor feeds
3. **Deploy to Edge**: Use ONNX conversion for edge device deployment
4. **Add Notifications**: Implement email/SMS alerts in monitoring module
5. **Cloud Integration**: Connect to cloud platforms for centralized monitoring

## Performance Expectations

- **Data Generation**: ~10-30 seconds for 10,000 samples
- **Model Training**: ~2-5 minutes on standard laptop
- **Inference Time**: <100ms per prediction
- **Model Size**: ~5-15 MB per model

## Support

For questions or issues:
1. Check logs in `logs/predictive_maintenance.log`
2. Review error messages in console
3. Ensure all dependencies are installed
4. Verify Python version (3.8+)

---

**Happy Monitoring! 🏭📊**
