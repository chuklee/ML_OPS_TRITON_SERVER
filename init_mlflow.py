import mlflow
import os
import shutil
import yaml
from datetime import datetime

def init_mlflow():
    # Nettoyer l'environnement existant
    if os.path.exists("mlruns"):
        shutil.rmtree("mlruns")
    
    # Créer le répertoire mlruns
    os.makedirs("mlruns", exist_ok=True)
    
    # Configurer MLflow
    tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Créer l'expérience
    experiment_name = "recommendation_experiment"
    
    # Définir l'ID explicitement
    experiment_id = "1"  # On utilise "1" car "0" est réservé pour l'expérience par défaut
    
    # Créer le répertoire de l'expérience
    exp_dir = os.path.join("mlruns", experiment_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Créer le fichier meta.yaml
    meta = {
        "artifact_location": os.path.abspath(exp_dir),
        "experiment_id": experiment_id,
        "lifecycle_stage": "active",
        "name": experiment_name,
        "creation_time": int(datetime.now().timestamp() * 1000)
    }
    
    with open(os.path.join(exp_dir, "meta.yaml"), "w") as f:
        yaml.dump(meta, f)
    
    # Créer aussi le fichier meta.yaml dans le répertoire principal
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