# Movie Recommendation System

A sophisticated movie recommendation system built with PyTorch, FastAPI, and MLflow. This system provides personalized movie recommendations using collaborative filtering and neural networks, enhanced with real-time serving capabilities and comprehensive monitoring.

## 🌟 Features

- **Neural Collaborative Filtering** with user and item embeddings
- **Real-time recommendations** via FastAPI endpoints
- **Interactive web interface** for demonstration
- **MLflow integration** for experiment tracking and model management
- **Comprehensive evaluation metrics** and visualizations
- **Automated hyperparameter tuning**
- **Production-ready model serving**

## 🏗️ Architecture

### Components
- **Frontend**: HTML/CSS/JavaScript interface for demonstration
- **Backend API**: FastAPI server for real-time serving
- **Model**: PyTorch-based neural collaborative filtering
- **Monitoring**: MLflow for experiment tracking and model management
- **Data Pipeline**: Efficient data processing and feature engineering

### Tech Stack
- **ML Framework**: PyTorch
- **API Framework**: FastAPI
- **Experiment Tracking**: MLflow
- **Data Processing**: Pandas, NumPy
- **Text Processing**: Sentence Transformers
- **Frontend**: HTML5, CSS3, JavaScript

## 📁 Project Structure

```
movie-recommender/
├── models/                 # Saved model files
├── static/                 # Frontend files
│   ├── index.html         # Main demo page
│   ├── architecture.html  # System architecture page
│   ├── style.css         # Styling
│   └── script.js         # Frontend logic
├── data/                  # Dataset directory
├── mlruns/               # MLflow experiment logs
├── models.py             # Model architecture
├── model_serving.py      # Model serving logic
├── online_serving.py     # FastAPI server
├── data_pipeline.py      # Data processing
├── utils.py             # Utility functions
├── train.py             # Training script
├── init_mlflow.py       # MLflow configuration
└── eval_model_utils.py   # Evaluation utilities


```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional)
- Git

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the System

1. **Run the Docker container**:
```bash
docker-compose up --build
```

2. **Access the web interface**:
- Open `http://localhost:8000/static/index.html` in your browser
- Try the demo by creating a new user and getting recommendations

3. **Access the MLflow dashboard**:
- Open `http://localhost:5000` in your browser
- View experiments and model metrics

### Monitoring & Management

1. **MLflow Dashboard**:
```bash
mlflow ui
```
- Visit `http://localhost:5000` to view experiments

2. **TensorBoard** (during training):
```bash
tensorboard --logdir=runs/recommendation_experiment
```
- Visit `http://localhost:6006` for training visualizations

## 📊 Model Performance

The system uses several metrics to evaluate performance:
- Recommendation accuracy
- User engagement metrics
- Model inference time
- System latency

Models achieving >85% accuracy are automatically saved as production candidates.

## 🔄 API Endpoints

### Create New User
```http
POST /users/new
Content-Type: application/json

{
    "age": 25,
    "occupation": 4,
    "gender": "F"
}
```

### Get Recommendations
```http
POST /recommendations/{user_id}?batch_size=20
```

### Train Model
```http
POST /model/train
Content-Type: application/json

{
    "epochs": 2,
    "batch_size": 512,
    "learning_rate": 0.0001,
    "user_emb_size": 32,
    "item_emb_size": 32
}
```

### Get Model Status
```http
GET /model/status
```

### Download Model
```http
GET /model/download
```

## 📈 Monitoring & Logging

The system provides comprehensive monitoring through:
- MLflow experiment tracking
- Real-time API metrics
- Model performance monitoring