from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from data_pipeline import DataPipeline
from model_serving import ModelServer
from train import train_model, read_dataset, preprocess_dataset, prepare_data, download_and_extract_dataset
from utils import save_production_model, load_production_model
import mlflow
from datetime import datetime
import os
from fastapi.responses import FileResponse
import tempfile
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Initialize services
data_pipeline = DataPipeline()
model_server = ModelServer(
    model_path='models/modelRec.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files at /static instead of root
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

class UserFeatures(BaseModel):
    age: int
    occupation: int
    gender: str

class RecommendationResponse(BaseModel):
    items: List[Dict]
    next_cursor: str

class TrainingConfig(BaseModel):
    epochs: int = 2
    batch_size: int = 512
    learning_rate: float = 1e-4
    user_emb_size: int = 32
    item_emb_size: int = 32

class TrainingResponse(BaseModel):
    status: str
    experiment_id: Optional[str]
    metrics: Dict[str, float]
    model_path: str

@app.post("/model/train")
async def train_new_model(config: TrainingConfig) -> TrainingResponse:
    """Endpoint to train a new model"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Download and prepare dataset
        download_and_extract_dataset()
        df_movies, df_ratings, df_users = read_dataset()
        df_combined = preprocess_dataset(df_movies, df_ratings, df_users)
        
        # Calculate user and item counts
        max_user_id = df_combined["user_id"].max()
        max_item_id = df_combined["item_id"].max()
        userCount = max_user_id + 1
        itemCount = max_item_id + 1
        
        # Prepare train and test datasets
        df_train, df_test = prepare_data(df_combined, df_movies, df_users)
        
        # Convert config to dict for train_model
        train_config = {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "user_embSize": config.user_emb_size,
            "item_embSize": config.item_emb_size
        }
        
        # Train model
        model = train_model(df_train, df_test, userCount, itemCount, device, train_config)
        
        # Save model and get metrics
        model_path = 'models/modelRec.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_accuracy': model.val_accuracy if hasattr(model, 'val_accuracy') else 0.0,
            'train_loss': model.train_loss if hasattr(model, 'train_loss') else 0.0
        }, model_path)
        
        # Save to MLflow with metadata
        metadata = {
            'training_date': datetime.now().strftime("%Y-%m-%d"),
            'model_version': '1.0',
            'config': train_config,
            'val_accuracy': model.val_accuracy if hasattr(model, 'val_accuracy') else 0.0,
            'train_loss': model.train_loss if hasattr(model, 'train_loss') else 0.0
        }
        
        experiment_id = save_production_model(model, "modelRec", metadata)
        
        # Return response
        return TrainingResponse(
            status="success",
            experiment_id=str(experiment_id) if experiment_id else None,
            metrics={
                "final_val_accuracy": float(model.val_accuracy) if hasattr(model, 'val_accuracy') else 0.0,
                "final_train_loss": float(model.train_loss) if hasattr(model, 'train_loss') else 0.0
            },
            model_path=model_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def get_model_status():
    """Get the current model status and information"""
    try:
        # Get MLflow client
        client = mlflow.tracking.MlflowClient()
        
        # Get experiment by name
        experiment_name = "recommendation_experiment"
        experiment = client.get_experiment_by_name(experiment_name)
                
        if experiment is None:
            return {
                "status": "no_model_found",
                "message": "No experiment found in MLflow"
            }
        
        # Get latest run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attribute.status = 'FINISHED'",
            order_by=["attribute.start_time DESC"],
            max_results=1
        )
        
        if runs:
            latest_run = runs[0]
            
            # Get metrics and format them
            metrics = {
                "final_val_accuracy": float(latest_run.data.metrics.get("final_val_accuracy", 0.0)),
                "final_train_loss": float(latest_run.data.metrics.get("final_train_loss", 0.0)),
                "val_accuracy": float(latest_run.data.metrics.get("val_accuracy", 0.0)),
                "train_loss": float(latest_run.data.metrics.get("train_loss", 0.0))
            }
            
            return {
                "status": "active",
                "model_version": latest_run.data.tags.get("model_version", "unknown"),
                "training_date": latest_run.data.tags.get("training_date", "unknown"),
                "metrics": metrics,
                "run_id": latest_run.info.run_id,
                "experiment_id": experiment.experiment_id
            }
        else:
            return {
                "status": "no_model_found",
                "message": "No trained model found in MLflow"
            }
            
    except Exception as e:
        print(f"MLflow error: {str(e)}")  # Add debug print
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations/{user_id}")
async def get_recommendations(
    user_id: int,
    cursor: Optional[str] = None,
    batch_size: int = 20
) -> RecommendationResponse:
    try:
        # Get user features
        user_features = data_pipeline.get_user_features(user_id)
        
        # Add gender encoding to user features
        user_features['gen_F'] = 1 if user_features['gender'] == 'F' else 0
        user_features['gen_M'] = 1 if user_features['gender'] == 'M' else 0
        
        # Get user embedding
        user_embedding = model_server.get_user_embedding(user_id, user_features)
        
        # Get recommendations
        recommended_items = model_server.get_recommendations(user_embedding, k=batch_size)
        
        # Enrich recommendations with item details
        enriched_recommendations = []
        for item_id in recommended_items:
            item_features = data_pipeline.get_item_features(item_id)
            enriched_recommendations.append({
                'item_id': item_id,
                'title': item_features['title'],
                'genres': [genre for genre, value in item_features.items() 
                          if value == 1 and genre not in ['item_id', 'title']],
            })
        
        # Generate next cursor (you might want to implement your own cursor logic)
        next_cursor = str(int(cursor or '0') + batch_size)
        
        return RecommendationResponse(
            items=enriched_recommendations,
            next_cursor=next_cursor
        )
    
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def root():
    return {"message": "Welcome to the Movie Recommender API"}

@app.post("/users/new")
async def create_new_user(user_features: UserFeatures):
    """Endpoint to handle new users"""
    try:
        # Convert gender to one-hot encoding
        gen_F = 1 if user_features.gender == 'F' else 0
        gen_M = 1 if user_features.gender == 'M' else 0
        
        # Create user features dict
        features = {
            'age': user_features.age,
            'occupation': user_features.occupation,
            'gen_F': gen_F,
            'gen_M': gen_M
        }
        
        # Generate new user ID
        user_id = model_server.get_next_user_id()
        
        # Get initial recommendations based on demographics
        similar_users = data_pipeline.get_similar_demographic_users(features)
        popular_items = data_pipeline.get_popular_items_for_demographic(similar_users)
        
        # Store user data
        data_pipeline.store_new_user(user_id, features)
        
        return {
            "user_id": user_id,
            "initial_recommendations": popular_items,
            "status": "success"
        }
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/download")
async def download_model(background_tasks: BackgroundTasks, run_id: Optional[str] = None):
    """Download a model from MLflow. If no run_id is provided, downloads the latest model."""
    try:
        # Get MLflow client
        client = mlflow.tracking.MlflowClient()
        
        # Get experiment
        experiment_name = "recommendation_experiment"
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise HTTPException(status_code=404, detail="No experiment found")
        
        # If no run_id provided, get the latest run
        if run_id is None:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="attribute.status = 'FINISHED'",
                order_by=["attribute.start_time DESC"],
                max_results=1
            )
            if not runs:
                raise HTTPException(status_code=404, detail="No runs found")
            run_id = runs[0].info.run_id
        
        # Create a temporary directory that will persist until the file is sent
        temp_dir = tempfile.mkdtemp()
        try:
            # Download the model file
            model_path = client.download_artifacts(
                run_id=run_id,
                path="modelRec.pth",
                dst_path=temp_dir
            )
            
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail="Model file not found")
            
            # Add cleanup task to background tasks
            def cleanup_temp_dir():
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Error cleaning up temp directory: {e}")
            
            background_tasks.add_task(cleanup_temp_dir)
            
            # Return the file
            return FileResponse(
                path=model_path,
                filename="modelRec.pth",
                media_type="application/octet-stream",
                headers={"Content-Disposition": "attachment; filename=modelRec.pth"}
            )
            
        except Exception as e:
            # Clean up in case of error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e
            
    except Exception as e:
        print(f"Error downloading model: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)