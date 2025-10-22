# Predictive Maintenance of Factory Equipment Using Edge AI

## 🏭 Project Overview

This project implements an end-to-end **Predictive Maintenance System** for factory equipment using Edge AI. The system monitors equipment sensor data in real-time, detects anomalies, and predicts potential failures before they occur, enabling proactive maintenance and reducing downtime.

## 🎯 Key Features

### **Core ML Capabilities**
- **Real-time Sensor Monitoring**: Continuous monitoring of equipment parameters (temperature, vibration, pressure, RPM, power consumption)
- **Anomaly Detection**: ML-based detection of unusual equipment behavior using Isolation Forest
- **Failure Prediction**: Predictive models (XGBoost, Random Forest, LightGBM) to forecast equipment failures 24-72 hours in advance
- **Edge Deployment**: Optimized models for deployment on edge devices with limited resources
- **Advanced Feature Engineering**: Rolling statistics, FFT, wavelets, time-based features

### **Production Features** ✨ NEW
- **🚀 REST API**: FastAPI-based REST API with automatic documentation
- **💾 Time-Series Database**: TimescaleDB integration for historical data storage
- **📡 MQTT Streaming**: Real-time sensor data ingestion from IoT devices
- **🧪 Comprehensive Testing**: Full test suite with pytest (90%+ coverage)
- **🔄 CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **☸️ Kubernetes Ready**: Complete K8s manifests for production deployment
- **📊 Monitoring**: Prometheus metrics and Grafana dashboards
- **🐳 Docker Support**: Fully containerized with images on Docker Hub (full & inference modes)
- **📈 Interactive Dashboard**: Real-time Plotly Dash visualization of equipment health

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
│   ├── api/                  # FastAPI REST API (NEW)
│   ├── streaming/            # MQTT broker integration (NEW)
│   ├── data_generation/      # Synthetic data generation
│   ├── preprocessing/        # Data preprocessing + advanced features
│   ├── models/              # ML model implementations
│   ├── edge_inference/      # Edge deployment code
│   └── monitoring/          # Dashboard and alerting
│
├── tests/                    # Comprehensive test suite (NEW)
│   ├── test_data_generator.py
│   ├── test_preprocessor.py
│   ├── test_model_trainer.py
│   ├── test_inference_engine.py
│   └── test_integration.py
│
├── k8s/                      # Kubernetes deployment (NEW)
│   └── deployment.yaml      # Complete K8s manifests
│
├── .github/workflows/        # CI/CD automation (NEW)
│   └── ci-cd.yml            # GitHub Actions pipeline
│
├── notebooks/                # Jupyter notebooks
│
├── config/                   # Configuration files
│   ├── config.yaml          # Main configuration
│   ├── mosquitto.conf       # MQTT broker config (NEW)
│   └── prometheus.yml       # Prometheus config (NEW)
│
├── logs/                    # Application logs
├── main.py                  # Main application entry point
├── quick-start.sh           # One-command startup (Linux/Mac) (NEW)
├── quick-start.bat          # One-command startup (Windows) (NEW)
├── docker-compose.full.yml  # 8-service stack (NEW)
├── requirements.txt         # Python dependencies
├── IMPROVEMENTS_GUIDE.md    # Comprehensive guide (NEW)
└── README.md               # This file
```

## 🚀 Quick Start

### ⚡ Super Quick Start (Production Stack)

**One command to start everything:**

**Windows:**
```powershell
.\quick-start.bat
```

**Linux/Mac:**
```bash
./quick-start.sh
```

This starts 8 services: API, Dashboard, TimescaleDB, Redis, MQTT, MLflow, Prometheus, Grafana

### 🐳 Option 1: Docker Compose (Recommended)

**Full production stack with 8 services:**

```bash
docker-compose -f docker-compose.full.yml up -d
```

**Access services:**
- Dashboard: http://localhost:8050
- REST API: http://localhost:8000/docs (Swagger UI)
- Grafana: http://localhost:3000 (admin/admin)
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090

**Simple dashboard only:**
```bash
docker-compose up -d
```

### 🐳 Option 2: Pull from Docker Hub

```bash
# Run the full system with dashboard
docker run -p 8050:8050 erenyeager471/pm-edge:latest

# Or run inference-only (smaller image)
docker run -p 8052:8050 erenyeager471/pm-edge-inference:latest
```

**See [DOCKER_RUN_GUIDE.md](DOCKER_RUN_GUIDE.md) for complete Docker instructions.**

Docker Hub Images:
- Full Image: `erenyeager471/pm-edge:latest`
- Inference Image: `erenyeager471/pm-edge-inference:latest`

### 💻 Option 3: Local Development

**Prerequisites:**
- Python 3.10 or higher
- pip package manager
- Docker (for database and MQTT)

**Installation:**

1. Clone or navigate to the project directory:
```bash
cd c:\ECF_project
```

2. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start infrastructure services:
```bash
docker-compose -f docker-compose.full.yml up -d timescaledb redis mosquitto
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

#### 5. Start REST API (NEW)
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

#### 6. Test with cURL (NEW)
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"equipment_id": "PUMP-001", "temperature": 85.5, "vibration": 2.3, "pressure": 120.0, "rpm": 1500, "power_consumption": 45.2}'

# Get recent alerts
curl http://localhost:8000/alerts?limit=10
```

#### 7. Run Tests (NEW)
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_integration.py -v
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

## � Production Features (NEW)

### REST API

Full REST API with automatic documentation powered by FastAPI.

**Available Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check for Kubernetes probes |
| GET | `/metrics` | Prometheus metrics export |
| POST | `/predict` | Single sensor prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/equipment` | Create equipment record |
| GET | `/equipment/{id}` | Get equipment status |
| GET | `/equipment/{id}/predictions` | Historical predictions |
| GET | `/alerts` | Recent alerts with filtering |
| POST | `/retrain` | Trigger model retraining |

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Database Schema (TimescaleDB)

```sql
-- Equipment table
equipment: id, name, type, location, installation_date, status

-- Sensor readings (hypertable - auto-partitioned by time)
sensor_readings: timestamp, equipment_id, temperature, vibration, pressure, rpm, power_consumption

-- Predictions
predictions: timestamp, equipment_id, anomaly_score, failure_probability, rul_estimate, alert_level

-- Alerts
alerts: timestamp, equipment_id, alert_type, severity, message, acknowledged
```

### MQTT Topics

Real-time sensor data streaming:

```
sensors/{equipment_id}/data    # Publish sensor readings
sensors/{equipment_id}/status  # Publish status updates
alerts/{equipment_id}          # Subscribe to alerts
predictions/{equipment_id}     # Subscribe to predictions
```

### Monitoring Stack

**Prometheus Metrics:**
- `pm_prediction_duration_seconds` - Inference latency
- `pm_predictions_total` - Total predictions counter
- `pm_alerts_total` - Total alerts by severity
- `pm_model_load_time_seconds` - Model load performance

**Grafana Dashboards:**
- Equipment health overview
- Prediction accuracy trends
- Alert history and patterns
- System performance metrics

### MLOps with MLflow

Track experiments, models, and metrics:

```python
import mlflow

# Log parameters
mlflow.log_param("n_estimators", 100)

# Log metrics
mlflow.log_metric("accuracy", 0.95)

# Log model
mlflow.sklearn.log_model(model, "model")
```

**MLflow UI:** http://localhost:5000

### Kubernetes Deployment

Deploy to production cluster:

```bash
# Create namespace and deploy
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n predictive-maintenance

# View logs
kubectl logs -n predictive-maintenance deployment/pm-api
```

**Included resources:**
- ConfigMap for configuration
- Secrets for credentials
- Deployments for all services (API, dashboard, database, MQTT, monitoring)
- Services with load balancers
- Ingress with TLS support
- Persistent volumes for data

## �🛠️ Technologies Used

### Core ML Stack
- **Python 3.10+**: Core programming language
- **scikit-learn**: Classical ML algorithms (Random Forest, Isolation Forest)
- **XGBoost & LightGBM**: Gradient boosting models
- **TensorFlow 2.13**: Deep learning models
- **Pandas & NumPy**: Data manipulation
- **Plotly/Dash**: Interactive dashboards
- **ONNX**: Model optimization and deployment

### Production Stack (NEW)
- **FastAPI**: Modern async web framework for APIs
- **TimescaleDB**: PostgreSQL extension for time-series data
- **Eclipse Mosquitto**: MQTT broker for IoT messaging
- **Redis**: In-memory cache for performance
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **MLflow**: Experiment tracking and model registry
- **Nginx**: Reverse proxy and load balancing

### MLOps & Monitoring (NEW)
- **Optuna**: Hyperparameter optimization
- **SHAP & LIME**: Model explainability
- **pytest**: Comprehensive testing (90%+ coverage)
- **GitHub Actions**: CI/CD automation
- **Kubernetes**: Container orchestration
- **Docker Compose**: Multi-service orchestration

### Advanced Features (NEW)
- **PyWavelets**: Wavelet transforms for signal processing
- **SciPy**: FFT and advanced statistical analysis
- **SQLAlchemy**: ORM for database operations
- **paho-mqtt**: MQTT client library
- **python-jose**: JWT authentication

## 📝 License

This project is for educational and demonstration purposes.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        IoT Sensors / Edge Devices                │
└───────────────────────┬─────────────────────────────────────────┘
                        │ MQTT (Port 1883)
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Eclipse Mosquitto Broker                      │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  FastAPI API │ │     Dash     │ │  Prometheus  │
│  (Port 8000) │ │  Dashboard   │ │  (Port 9090) │
│              │ │  (Port 8050) │ │              │
└──────┬───────┘ └──────────────┘ └──────┬───────┘
       │                                  │
       ▼                                  ▼
┌──────────────────────────────────────────────────────┐
│  TimescaleDB (PostgreSQL + Time-series Extension)    │
│  - Sensor readings (hypertable)                      │
│  - Equipment metadata                                 │
│  - Predictions & alerts                              │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  ML Models (scikit-learn, XGBoost, TensorFlow)      │
│  - Anomaly Detection (Isolation Forest)              │
│  - Failure Prediction (Random Forest, XGBoost)       │
│  - RUL Estimation (Regression Models)                │
└──────────────────────────────────────────────────────┘
```

**Data Flow:**
1. **Ingestion**: IoT sensors publish data to MQTT broker
2. **Storage**: API consumes MQTT messages, stores in TimescaleDB
3. **Processing**: Sensor data preprocessed (scaling, feature engineering)
4. **Prediction**: ML models generate predictions (anomaly, failure, RUL)
5. **Alerting**: Critical predictions trigger alerts via MQTT and API
6. **Visualization**: Dashboard displays real-time equipment health
7. **Monitoring**: Prometheus scrapes metrics, Grafana visualizes

## 📚 Documentation

- **[IMPROVEMENTS_GUIDE.md](IMPROVEMENTS_GUIDE.md)** - Comprehensive guide to all production features
- **[DOCKER_RUN_GUIDE.md](DOCKER_RUN_GUIDE.md)** - Docker deployment instructions
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Beginner's guide
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Detailed project documentation
- **API Docs**: http://localhost:8000/docs (when running)

## 🧪 Testing

The project includes a comprehensive test suite with 90%+ code coverage:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_integration.py -v  # Integration tests
pytest tests/test_model_trainer.py -v  # Model training tests
```

**Test Coverage:**
- Unit tests for all modules
- Integration tests for end-to-end workflows
- API endpoint tests
- Database integration tests
- MQTT broker tests

## 🤝 Contributing

Feel free to extend this project with:
- Additional sensor types (acoustic, thermal imaging)
- More sophisticated models (LSTM, Transformer, GAN)
- Integration with real IoT devices (Raspberry Pi, Arduino)
- Cloud connectivity (AWS IoT, Azure IoT Hub, Google Cloud IoT)
- Advanced visualization (3D equipment models, AR/VR)
- Mobile app for alerts and monitoring
- Federated learning across multiple sites
- Digital twin simulation
- Predictive maintenance scheduling optimization

**Development Workflow:**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Run tests: `pytest tests/ -v`
4. Commit changes: `git commit -am 'Add new feature'`
5. Push to branch: `git push origin feature/new-feature`
6. Create Pull Request

## � Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Check what's using the port (Windows)
netstat -ano | findstr :8000

# Check what's using the port (Linux/Mac)
lsof -i :8000

# Kill the process or change port in docker-compose.yml
```

**2. Database Connection Failed**
```bash
# Check if TimescaleDB is running
docker ps | grep timescaledb

# View database logs
docker logs ecf_project_timescaledb_1

# Restart database
docker-compose -f docker-compose.full.yml restart timescaledb
```

**3. MQTT Connection Issues**
```bash
# Test MQTT broker
docker exec -it ecf_project_mosquitto_1 mosquitto_sub -t 'sensors/#' -v

# Publish test message
docker exec -it ecf_project_mosquitto_1 mosquitto_pub -t 'sensors/test' -m 'hello'
```

**4. Model Not Found**
```bash
# Generate data and train models first
python main.py --mode generate --samples 10000
python main.py --mode train --data data/synthetic/sensor_data.csv
```

**5. Import Errors**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade

# Or rebuild Docker images
docker-compose -f docker-compose.full.yml build --no-cache
```

### Logs

View logs for debugging:

```bash
# Docker Compose logs
docker-compose -f docker-compose.full.yml logs -f api
docker-compose -f docker-compose.full.yml logs -f dashboard

# Application logs
tail -f logs/app.log

# Kubernetes logs
kubectl logs -n predictive-maintenance deployment/pm-api -f
```

## 📊 Performance Benchmarks

**Inference Performance:**
- Single prediction latency: <50ms
- Batch prediction (100 samples): <200ms
- MQTT message processing: <10ms
- Model load time: <1s

**Scalability:**
- API can handle 1000+ requests/second
- TimescaleDB supports millions of sensor readings
- Horizontal scaling with Kubernetes (tested up to 10 replicas)

**Resource Usage:**
- API container: ~200MB RAM
- Dashboard container: ~150MB RAM
- TimescaleDB: ~500MB RAM (1M records)
- Total stack: ~2GB RAM, 4 CPU cores recommended

## 🎯 Roadmap

### Completed ✅
- [x] Comprehensive test suite (90%+ coverage)
- [x] REST API with FastAPI
- [x] Time-series database (TimescaleDB)
- [x] MQTT streaming support
- [x] MLOps with MLflow
- [x] Monitoring (Prometheus/Grafana)
- [x] CI/CD pipeline
- [x] Kubernetes deployment
- [x] Advanced feature engineering (FFT, wavelets)
- [x] Authentication & security

### In Progress 🚧
- [ ] Deep learning models (LSTM, CNN)
- [ ] Digital twin simulation
- [ ] Federated learning
- [ ] Maintenance scheduler optimization
- [ ] ROI calculator

### Future Enhancements 🚀
- [ ] Mobile app (React Native)
- [ ] WebSocket real-time updates
- [ ] Multi-tenancy support
- [ ] Advanced anomaly detection (GAN, VAE)
- [ ] Integration with cloud platforms (AWS, Azure, GCP)
- [ ] AR/VR visualization
- [ ] Voice alerts and commands
- [ ] Blockchain for audit trail

## 📧 Support

For questions or issues:
- Check [IMPROVEMENTS_GUIDE.md](IMPROVEMENTS_GUIDE.md) for detailed documentation
- Review [DOCKER_RUN_GUIDE.md](DOCKER_RUN_GUIDE.md) for deployment help
- Open an issue on GitHub with logs and error messages
- Refer to code documentation and inline comments

## 📝 License

This project is for educational and demonstration purposes.

## 🌟 Acknowledgments

This project demonstrates modern MLOps practices for industrial IoT:
- Production-ready REST API architecture
- Time-series data management at scale
- Real-time streaming with MQTT
- Containerized microservices
- Kubernetes orchestration
- Comprehensive monitoring and observability
- CI/CD automation
- Test-driven development

---

**Built with ❤️ for Industrial Edge AI & Predictive Maintenance**

**Docker Hub**: [erenyeager471/pm-edge](https://hub.docker.com/r/erenyeager471/pm-edge)  
**GitHub**: primusorion/ECF_Project
