# 🐳 Running Your Predictive Maintenance System in Docker

## ✅ Images Successfully Pushed to Docker Hub!

Your Docker images are now publicly available on Docker Hub:

- **Full Image (Dashboard + Training)**: `erenyeager471/pm-edge:latest`
- **Inference-Only Image (Optimized)**: `erenyeager471/pm-edge-inference:latest`

---

## 🚀 Quick Start - Run From Any Machine

### Option 1: Run the Full Dashboard (Recommended for Development)

```powershell
# Pull and run the dashboard
docker run --rm -p 8050:8050 --name pm-edge erenyeager471/pm-edge:latest
```

Then open your browser to: **http://localhost:8050**

**What this does:**
- Downloads the image from Docker Hub (first time only)
- Starts the monitoring dashboard
- Maps port 8050 on your machine to the container

---

### Option 2: Run the Inference-Only Image (Smaller, Optimized)

**⚠️ Important:** This image needs trained models to work. You have two options:

#### A. Mount Local Models (If you have them)

```powershell
# Navigate to your project folder
cd C:\ECF_project

# Run with models mounted
docker run --rm -p 8052:8050 --name pm-edge-infer `
  -v ${PWD}\models\saved_models:/app/models/saved_models:ro `
  erenyeager471/pm-edge-inference:latest
```

#### B. Generate Models First

```powershell
# Step 1: Run the full image to generate and train models
docker run --rm `
  -v ${PWD}\models:/app/models `
  -v ${PWD}\data:/app/data `
  erenyeager471/pm-edge:latest python main.py --mode full

# Step 2: Now run inference with the generated models
docker run --rm -p 8052:8050 `
  -v ${PWD}\models\saved_models:/app/models/saved_models:ro `
  erenyeager471/pm-edge-inference:latest
```

---

## 📦 All Available Run Modes

### Mode 1: Data Generation Only
```powershell
docker run --rm -v ${PWD}\data:/app/data `
  erenyeager471/pm-edge:latest python main.py --mode generate --samples 10000
```

### Mode 2: Model Training Only
```powershell
docker run --rm `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\models:/app/models `
  erenyeager471/pm-edge:latest python main.py --mode train
```

### Mode 3: Run Inference Demo
```powershell
docker run --rm `
  -v ${PWD}\models:/app/models `
  erenyeager471/pm-edge:latest python main.py --mode infer
```

### Mode 4: Full Pipeline (Generate + Train + Infer)
```powershell
docker run --rm `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\models:/app/models `
  erenyeager471/pm-edge:latest python main.py --mode full
```

### Mode 5: Monitoring Dashboard
```powershell
docker run --rm -p 8050:8050 `
  -v ${PWD}\models:/app/models `
  erenyeager471/pm-edge:latest python main.py --mode monitor
```

---

## 🌐 Run From ANY Computer (Without Local Files)

Anyone can run your system without cloning the repo:

### Just the Dashboard (No Data/Models)
```bash
docker run --rm -p 8050:8050 erenyeager471/pm-edge:latest
```

### Full Pipeline (Generates everything inside container)
```bash
docker run --rm -p 8050:8050 erenyeager471/pm-edge:latest python main.py --mode full
```

**Note:** Models and data will be created inside the container and lost when it stops (unless you mount volumes).

---

## 🔧 Using Docker Compose (Recommended)

### Option A: Development with Full Image
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    image: erenyeager471/pm-edge:latest
    ports:
      - "8050:8050"
    volumes:
      - .:/app
    command: python main.py --mode monitor
```

Run:
```powershell
docker-compose up
```

### Option B: Inference with Models
Create `docker-compose.inference.yml`:
```yaml
version: '3.8'

services:
  inference:
    image: erenyeager471/pm-edge-inference:latest
    ports:
      - "8052:8050"
    volumes:
      - ./models/saved_models:/app/models/saved_models:ro
    command: python main.py --mode infer
    restart: unless-stopped
```

Run:
```powershell
docker-compose -f docker-compose.inference.yml up
```

---

## 📊 Image Sizes

- **Full Image** (`pm-edge`): ~2.8 GB (includes TensorFlow, XGBoost, LightGBM, Dash, all training libraries)
- **Inference Image** (`pm-edge-inference`): ~800 MB (minimal runtime dependencies only)

**Use the inference image for production/edge deployment!**

---

## 🛠️ Advanced Usage

### Run in Detached Mode (Background)
```powershell
docker run -d -p 8050:8050 --name pm-edge-bg erenyeager471/pm-edge:latest
```

### Check Logs
```powershell
docker logs pm-edge-bg
```

### Stop Container
```powershell
docker stop pm-edge-bg
```

### Interactive Shell (Debug)
```powershell
docker run -it --entrypoint /bin/bash erenyeager471/pm-edge:latest
```

---

## 🔐 Environment Variables

You can pass configuration via environment variables:

```powershell
docker run --rm -p 8050:8050 `
  -e PYTHONUNBUFFERED=1 `
  -e DEBUG_MODE=false `
  erenyeager471/pm-edge:latest
```

---

## 🌍 Deploy to Cloud

### Deploy to Azure Container Instances
```bash
az container create `
  --resource-group myResourceGroup `
  --name pm-edge-app `
  --image erenyeager471/pm-edge:latest `
  --dns-name-label pm-edge-demo `
  --ports 8050
```

### Deploy to AWS ECS/Fargate
```bash
# Use AWS CLI or Console to create a task definition
# pointing to: erenyeager471/pm-edge:latest
```

### Deploy to Google Cloud Run
```bash
gcloud run deploy pm-edge `
  --image erenyeager471/pm-edge:latest `
  --platform managed `
  --allow-unauthenticated `
  --port 8050
```

---

## 📝 Pull Images Manually

```powershell
# Pull full image
docker pull erenyeager471/pm-edge:latest

# Pull inference image
docker pull erenyeager471/pm-edge-inference:latest

# List your images
docker images | Select-String "erenyeager471"
```

---

## 🎯 Quick Reference

| Command | Description |
|---------|-------------|
| `docker pull erenyeager471/pm-edge:latest` | Download full image |
| `docker pull erenyeager471/pm-edge-inference:latest` | Download inference image |
| `docker run -p 8050:8050 erenyeager471/pm-edge:latest` | Run dashboard |
| `docker ps` | List running containers |
| `docker logs <container-id>` | View container logs |
| `docker stop <container-id>` | Stop a container |
| `docker system prune -a` | Clean up unused images/containers |

---

## 🎉 Success!

Your predictive maintenance system is now available on Docker Hub and can be run anywhere with Docker installed!

**Docker Hub URLs:**
- https://hub.docker.com/r/erenyeager471/pm-edge
- https://hub.docker.com/r/erenyeager471/pm-edge-inference

**Share these images with anyone:**
```
erenyeager471/pm-edge:latest
erenyeager471/pm-edge-inference:latest
```

---

## 💡 Tips

1. **First Run**: The first `docker run` will download the image (~1-3 GB). Subsequent runs are instant.
2. **Data Persistence**: Use `-v` volume mounts to persist data/models outside containers.
3. **Production**: Use the inference image with proper orchestration (Kubernetes, Docker Swarm, etc.).
4. **Updates**: When you update code, rebuild and push: `docker build -t erenyeager471/pm-edge:latest . && docker push erenyeager471/pm-edge:latest`

---

## 🐛 Troubleshooting

**Port already in use?**
```powershell
# Use a different port
docker run -p 8051:8050 erenyeager471/pm-edge:latest
```

**Out of disk space?**
```powershell
docker system prune -a --volumes
```

**Permission denied on Windows?**
- Make sure Docker Desktop is running
- Try running PowerShell as Administrator

---

🚀 **You're all set! Your project is now Dockerized and deployable anywhere!**
