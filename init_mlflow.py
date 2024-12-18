import mlflow
import os
import yaml
from datetime import datetime
import time

def init_mlflow():
    """Initialize MLflow tracking"""
    # Get tracking URI from environment or use default
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Set experiment name
    experiment_name = "recommendation_experiment"
    mlruns_dir = tracking_uri.replace("file://", "").replace("file:", "")
    
    # Ensure directories exist with proper permissions
    os.makedirs(mlruns_dir, exist_ok=True)
    os.chmod(mlruns_dir, 0o777)
    
    # Wait for MLflow server to be ready
    max_retries = 5
    for i in range(max_retries):
        try:
            # Try to create or get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=os.path.join(mlruns_dir, experiment_name)
                )
            else:
                experiment_id = experiment.experiment_id
                
            mlflow.set_experiment(experiment_name)
            print(f"MLflow initialized with experiment ID: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            if i < max_retries - 1:
                print(f"Retrying MLflow initialization ({i+1}/{max_retries}): {e}")
                time.sleep(2)  # Wait before retrying
            else:
                print(f"Failed to initialize MLflow after {max_retries} attempts: {e}")
                return None

if __name__ == "__main__":
    init_mlflow()