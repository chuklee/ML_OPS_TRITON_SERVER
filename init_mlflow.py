import mlflow
import os
import yaml
from datetime import datetime

def init_mlflow():
    """Initialize MLflow tracking"""
    # Get tracking URI from environment or use default
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Set experiment name and base directory
    experiment_name = "recommendation_experiment"
    mlruns_dir = tracking_uri.replace("file://", "").replace("file:", "")
    
    try:
        # Ensure base mlruns directory exists
        os.makedirs(mlruns_dir, exist_ok=True)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Create experiment directory
            experiment_dir = os.path.join(mlruns_dir, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Create meta.yaml for the experiment
            meta = {
                "artifact_location": os.path.abspath(experiment_dir),
                "creation_time": int(datetime.now().timestamp() * 1000),
                "experiment_id": "1",  # Fixed ID for consistency
                "last_update_time": int(datetime.now().timestamp() * 1000),
                "lifecycle_stage": "active",
                "name": experiment_name,
                "tags": {}
            }
            
            # Write experiment meta.yaml
            meta_path = os.path.join(experiment_dir, "meta.yaml")
            with open(meta_path, "w") as f:
                yaml.safe_dump(meta, f)
            
            # Create root meta.yaml if it doesn't exist
            root_meta_path = os.path.join(mlruns_dir, "meta.yaml")
            if not os.path.exists(root_meta_path):
                root_meta = {
                    "experiments": {
                        "1": {
                            "artifact_location": os.path.abspath(experiment_dir),
                            "creation_time": meta["creation_time"],
                            "experiment_id": "1",
                            "last_update_time": meta["last_update_time"],
                            "lifecycle_stage": "active",
                            "name": experiment_name
                        }
                    }
                }
                with open(root_meta_path, "w") as f:
                    yaml.safe_dump(root_meta, f)
            
            experiment_id = "1"
        else:
            experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(experiment_name)
        return experiment_id
        
    except Exception as e:
        print(f"Warning during MLflow initialization: {e}")
        return None

if __name__ == "__main__":
    init_mlflow()