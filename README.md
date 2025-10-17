# Predictive Maintenance of Factory Equipment Using Edge AI

## 🏭 Project Overview

This project implements an end-to-end **Predictive Maintenance System** for factory equipment using Edge AI. The system monitors equipment sensor data in real-time, detects anomalies, and predicts potential failures before they occur, enabling proactive maintenance and reducing downtime.

## 🎯 Key Features

- **Real-time Sensor Monitoring**: Continuous monitoring of equipment parameters (temperature, vibration, pressure, RPM, power consumption)
- **Anomaly Detection**: ML-based detection of unusual equipment behavior
- **Failure Prediction**: Predictive models to forecast equipment failures 24-72 hours in advance
- **Edge Deployment**: Optimized models for deployment on edge devices with limited resources
- **Alert System**: Automated notifications for detected anomalies and predicted failures
- **Interactive Dashboard**: Real-time visualization of equipment health and predictions
- **Data Generation**: Synthetic sensor data generator for testing and development

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Factory Equipment                         │
│         (Motors, Pumps, Compressors, Conveyors)             │
└────────────────────┬────────────────────────────────────────┘
                     │ Sensor Data
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Edge Device (IoT Gateway)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Data        │→ │ Preprocessing │→ │ ML Inference     │  │
│  │ Collection  │  │ & Feature Eng │  │ Engine           │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │ Predictions & Alerts
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Monitoring Dashboard & Alerts                   │
│         (Maintenance Team / Operations Center)              │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ECF_project/
│
├── data/                       # Data storage
│   ├── raw/                   # Raw sensor data
│   ├── processed/             # Processed features
│   └── synthetic/             # Generated synthetic data
│
├── models/                    # Trained ML models
│   ├── saved_models/         # Serialized models
│   └── edge_optimized/       # Quantized/optimized models
│
├── src/                       # Source code
│   ├── data_generation/      # Synthetic data generation
│   ├── preprocessing/        # Data preprocessing
│   ├── models/              # ML model implementations
│   ├── edge_inference/      # Edge deployment code
│   └── monitoring/          # Dashboard and alerting
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_demo.ipynb
│
├── config/                   # Configuration files
├── logs/                    # Application logs
├── tests/                   # Unit tests
│
├── main.py                  # Main application entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment

### Installation

1. Clone or navigate to the project directory:
```bash
cd c:\ECF_project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Generate Synthetic Data
```bash
python main.py --mode generate --samples 10000
```

#### 2. Train Models
```bash
python main.py --mode train --data data/synthetic/sensor_data.csv
```

#### 3. Run Edge Inference
```bash
python main.py --mode infer --model models/saved_models/predictive_model.pkl
```

#### 4. Start Monitoring Dashboard
```bash
python main.py --mode monitor --port 8050
```

#### 5. Run Full Pipeline
```bash
python main.py --mode full
```

## 🤖 Machine Learning Models

### 1. Anomaly Detection
- **Algorithm**: Isolation Forest / Autoencoder
- **Purpose**: Detect unusual sensor patterns indicating potential issues
- **Output**: Anomaly score (0-1)

### 2. Failure Prediction
- **Algorithm**: Random Forest / Gradient Boosting / LSTM
- **Purpose**: Predict equipment failure within next 24-72 hours
- **Output**: Failure probability and estimated time to failure

### 3. Remaining Useful Life (RUL)
- **Algorithm**: Regression models (Random Forest, XGBoost)
- **Purpose**: Estimate remaining operational hours before maintenance required
- **Output**: Hours until maintenance needed

## 📊 Sensor Data Parameters

The system monitors the following parameters:

| Parameter | Unit | Normal Range | Critical Threshold |
|-----------|------|--------------|-------------------|
| Temperature | °C | 40-80 | >95 |
| Vibration | mm/s | 0-5 | >8 |
| Pressure | Bar | 5-10 | >12 or <3 |
| RPM | Revolutions/min | 1000-3000 | <500 or >3500 |
| Power Consumption | kW | 10-50 | >70 |
| Acoustic Emission | dB | 60-85 | >95 |

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- Data collection frequency
- Model parameters
- Alert thresholds
- Edge device specifications
- Dashboard settings

## 📈 Performance Metrics

- **Anomaly Detection**: F1-Score, Precision, Recall
- **Failure Prediction**: Accuracy, AUC-ROC, Lead Time
- **RUL Estimation**: MAE (Mean Absolute Error), RMSE
- **Edge Performance**: Inference time (<100ms), Model size (<10MB)

## 🎓 Learning Resources

This project demonstrates:
- Time-series data analysis
- Feature engineering for sensor data
- Machine learning for predictive maintenance
- Model optimization for edge deployment
- Real-time monitoring systems
- Industrial IoT concepts

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **scikit-learn**: Classical ML algorithms
- **TensorFlow/Keras**: Deep learning models
- **Pandas & NumPy**: Data manipulation
- **Plotly/Dash**: Interactive dashboards
- **ONNX**: Model optimization and deployment
- **PyYAML**: Configuration management

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to extend this project with:
- Additional sensor types
- More sophisticated models
- Integration with real IoT devices
- Cloud connectivity
- Advanced visualization

## 📧 Support

For questions or issues, please refer to the code documentation and comments.

---

**Built with ❤️ for Industrial Edge AI Applications**
