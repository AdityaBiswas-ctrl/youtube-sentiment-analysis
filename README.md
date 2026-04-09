# YouTube Sentiment Analysis — Full MLOps Project

An end-to-end YouTube comment sentiment analysis platform with a complete MLOps stack: **DVC** data/model versioning, **MLflow** experiment tracking, **Docker** containerization, and **GitHub Actions CI/CD**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-yellowgreen)
![MLflow](https://img.shields.io/badge/MLflow-2.21-blue)
![DVC](https://img.shields.io/badge/DVC-3.59-purple)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        CI/CD (GitHub Actions)                    │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │ Lint/Test │ → │ DVC Pipeline │ → │ Docker Build & Push  │    │
│  └──────────┘   └──────────────┘   └──────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘

┌──────────── ML Pipeline (DVC) ────────────┐
│                                            │
│  fetch_data → preprocess → train → evaluate│
│       │            │         │        │    │
│    YouTube      Clean &   XGBoost  Metrics │
│     API        Split     LightGBM   JSON   │
│                           + MLflow         │
└────────────────────────────────────────────┘

┌──────────── Serving (Docker) ─────────────┐
│                                            │
│  ┌─────────────┐    ┌────────────────┐    │
│  │ Flask App   │    │ MLflow Server  │    │
│  │ :5000       │    │ :5001          │    │
│  │ Dashboard + │    │ Experiment     │    │
│  │ REST API    │    │ Tracking       │    │
│  └─────────────┘    └────────────────┘    │
└────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 
- YouTube Data API v3 key ([Get one here](https://console.cloud.google.com/))

### 1. Project Setup & Installation

```bash
git clone https://github.com/your-username/youtube-sentiment-analysis.git
cd youtube-sentiment-analysis

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Key

Create a file named `.env` in the root of your project folder and add your key:

```text
YOUTUBE_API_KEY="your_api_key_here"
```

Or export it directly in your terminal:

```bash
# Windows PowerShell
$env:YOUTUBE_API_KEY="your_api_key_here"

# Linux/macOS
export YOUTUBE_API_KEY="your_api_key_here"
```

### 3. Initialize DVC & Run Pipeline

```bash
# Initialize DVC
dvc init

# Run the full ML pipeline
dvc repro

# View metrics
dvc metrics show
```

### 4. Start MLflow (Experiment Tracking)

```bash
# Run the MLflow server locally
mlflow server --host 127.0.0.1 --port 5001
```
*Open MLflow Dashboard at: http://127.0.0.1:5001*

### 5. Start the Application

In a new terminal (make sure to activate your virtual environment first):

```bash
# Start Flask server
python -m src.api.app
```
*Open web dashboard at: http://localhost:5000*

---

## 🐳 Docker

### Run with Docker Compose

```bash
# Set your API key
echo "YOUTUBE_API_KEY=your_key" > .env

# Build and start containers
docker-compose up --build

# App:    http://localhost:5000
# MLflow: http://localhost:5001
```

### Build Only

```bash
docker build -t youtube-sentiment-analysis .
docker run -p 5000:5000 -e YOUTUBE_API_KEY=your_key youtube-sentiment-analysis
```

---

## 📊 DVC Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| `fetch_data` | `src/data/fetch_comments.py` | Fetches YouTube comments via API, generates VADER pseudo-labels |
| `preprocess` | `src/data/preprocess.py` | Cleans text, filters, deduplicates, train/test split |
| `train` | `src/models/train.py` | Trains XGBoost or LightGBM, logs to MLflow |
| `evaluate` | `src/models/evaluate.py` | Computes metrics, logs to MLflow, saves JSON |

### Switch Models

Edit `params.yaml`:
```yaml
train:
  model_type: "lightgbm"   # Change from "xgboost" to "lightgbm"
```

Then re-run:
```bash
dvc repro
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short
```

---

## 📁 Project Structure

```
youtube-sentiment-analysis/
├── .github/workflows/ci-cd.yml    # CI/CD pipeline
├── configs/config.yaml            # Centralized config
├── data/raw/                      # Raw comments (DVC tracked)
├── data/processed/                # Processed data (DVC tracked)
├── models/                        # Model artifacts (DVC tracked)
├── metrics/                       # Evaluation metrics
├── src/
│   ├── data/                      # Data ingestion & preprocessing
│   ├── features/                  # Feature extraction (TF-IDF)
│   ├── models/                    # Training & evaluation
│   └── api/                       # Flask REST API
├── static/                        # Dashboard UI (HTML/CSS/JS)
├── tests/                         # Unit & API tests
├── dvc.yaml                       # DVC pipeline definition
├── params.yaml                    # Pipeline parameters
├── Dockerfile                     # App container
├── docker-compose.yml             # Multi-service setup
└── requirements.txt               # Python dependencies
```

---

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `YOUTUBE_API_KEY` | YouTube Data API v3 key | Yes |
| `MLFLOW_TRACKING_URI` | MLflow server URL (default: `http://localhost:5001`) | No |

---

## 📈 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard UI |
| `POST` | `/api/analyze` | Analyze video comments |
| `GET` | `/api/model-info` | Current model info & metrics |
| `GET` | `/api/health` | Health check |

---

## License

MIT
