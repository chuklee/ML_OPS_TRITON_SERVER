import mlflow
import os
import shutil
import yaml
from datetime import datetime

def init_mlflow():
    # Clean the existing environment
    if os.path.exists("mlruns"):
        shutil.rmtree("mlruns")
    
    # Create the mlruns directory
    os.makedirs("mlruns", exist_ok=True)
    
    # Configure MLflow
    tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create the experiment
    experiment_name = "recommendation_experiment"
    
    # Set the ID explicitly
    experiment_id = "1"  # We use "1" because "0" is reserved for the default experiment
    
    # Create the experiment directory
    exp_dir = os.path.join("mlruns", experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create the meta.yaml file
    meta = {
        "artifact_location": os.path.abspath(exp_dir),
        "experiment_id": experiment_id,
        "lifecycle_stage": "active",
        "name": experiment_name,
        "creation_time": int(datetime.now().timestamp() * 1000)
    }
    
    with open(os.path.join(exp_dir, "meta.yaml"), "w") as f:
        yaml.dump(meta, f)
    
    # Create also the meta.yaml file in the root directory
    root_meta = {
        "experiments": {
            experiment_id: {
                "artifact_location": os.path.abspath(exp_dir),
                "experiment_id": experiment_id,
                "lifecycle_stage": "active",
                "name": experiment_name,
                "creation_time": int(datetime.now().timestamp() * 1000)
            }
        }
    }
    
    with open(os.path.join("mlruns", "meta.yaml"), "w") as f:
        yaml.dump(root_meta, f)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID: {experiment_id}")
    print("MLflow initialization completed successfully!")
    
    return experiment_name

if __name__ == "__main__":
    experiment_name = init_mlflow()