import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
import numpy as np
from pprint import pprint
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from sentence_transformers import SentenceTransformer
import multiprocessing

import faiss
from models import cls_model
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.store.db.utils import _upgrade_db


import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, BigInteger, Float

from sqlalchemy.orm import declarative_base
from sqlalchemy.exc import MovedIn20Warning
from datetime import datetime
from utils import (cls_dataset, FocalLoss, debug_shapes, 
                  read_dataset, preprocess_dataset, prepare_data,
                  download_and_extract_dataset, encode_title, 
                  create_and_log_plots, save_production_model, load_production_model)
from eval_model_utils import evaluation, detailed_user_recommendations, analyze_multiple_users
import warnings
warnings.filterwarnings('ignore', category=MovedIn20Warning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from typing import Callable


def train_model(df_train, df_test, userCount, itemCount, device, config, progress_callback: Callable = None):
    """
    Train the recommendation model with MLflow tracking
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    experiment_name = setup_mlflow()
    mlflow.set_experiment(experiment_name)
    os.makedirs("mlruns", exist_ok=True)
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow run ID: {run.info.run_id}")
        print("\nTracking interfaces:")
        print("- TensorBoard: tensorboard --logdir=runs/recommendation_experiment")
        print("  Then visit: http://localhost:6006")
        print("- MLflow UI: mlflow ui")
        print("  Then visit: http://localhost:5000\n")
        # Log parameters
        mlflow.log_params({
            "userCount": userCount,
            "itemCount": itemCount,
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "user_embSize": config["user_embSize"],
            "item_embSize": config["item_embSize"],
            "model_architecture": "cls_model"
        })

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir='runs/recommendation_experiment')

        # Model initialization
        modelRec = cls_model(userCount, itemCount, user_embSize=config["user_embSize"], item_embSize=config["item_embSize"])
        if torch.cuda.device_count() > 1:
            modelRec = nn.DataParallel(modelRec)
        modelRec = modelRec.to(device)

        # Dataset preparation
        all_unique_titles = pd.concat([df_train, df_test])["title"].unique()
        title_to_idx = {title: idx for idx, title in enumerate(all_unique_titles)}
        dict_titlEmb = encode_title(df_train, df_test)
        list_titlEmb = [dict_titlEmb[title] for title in title_to_idx]
        title_emb_tensor = torch.stack(list_titlEmb)

        # Dataset and DataLoader
        ds_train = cls_dataset(df_train, userCount, itemCount, title_emb_tensor, title_to_idx)
        ds_test = cls_dataset(df_test, userCount, itemCount, title_emb_tensor, title_to_idx)

        num_workers = multiprocessing.cpu_count()
        ds_trainLoader = DataLoader(ds_train, batch_size=config["batch_size"], num_workers=num_workers, pin_memory=True)
        ds_testLoader = DataLoader(ds_test, batch_size=1000, num_workers=num_workers, pin_memory=True)

        # Training setup
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean').to(device)
        optim = torch.optim.Adam(modelRec.parameters(), lr=config["learning_rate"])
        scaler = torch.GradScaler()

        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        # Training loop
        total_batches = len(ds_trainLoader)
        best_val_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(config["epochs"]):
            # Training
            for batch_idx, (x1, x2, y, _) in enumerate(ds_trainLoader):
                # Update progress if callback is provided
                if progress_callback:
                    progress_callback(epoch, config["epochs"], batch_idx, total_batches)
                    
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                
                optim.zero_grad()
                with torch.autocast(device_type='cuda'):
                    logits = modelRec(x1, x2)
                    loss = criterion(logits, y.squeeze())
                
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                
                history['train_loss'].append(loss.item())
                preds = torch.argmax(logits, dim=1)
                history['val_accuracy'].append(accuracy_score(y.squeeze().cpu().numpy(), preds.cpu().numpy()))
                
                # TensorBoard logging
                writer.add_scalar('Train/Loss', loss.item(), epoch * total_batches + batch_idx)
                writer.add_scalar('Validation/Accuracy', history['val_accuracy'][-1], epoch * total_batches + batch_idx)

                print(f"Epoch {epoch+1} - Train Loss: {loss.item():.4f} - Val Accuracy: {history['val_accuracy'][-1]:.4f}")

                # Save best model
                if history['val_accuracy'][-1] > best_val_accuracy:
                    best_val_accuracy = history['val_accuracy'][-1]
                    best_model_state = modelRec.state_dict()

            # Validation
            val_loss, val_accuracy = validate_epoch(modelRec, ds_testLoader, criterion, device)

            history['train_loss'].append(history['train_loss'][-1])
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            # Log metrics
            mlflow.log_metrics({
                "train_loss": history['train_loss'][-1],
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)

            create_and_log_plots(history, epoch)

            # TensorBoard logging
            writer.add_scalar('Train/Loss', history['train_loss'][-1], epoch)
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)

            print(f"Epoch {epoch+1} - Train Loss: {history['train_loss'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

        # Store final metrics in the model for easy access
        modelRec.val_accuracy = best_val_accuracy
        modelRec.train_loss = history['train_loss'][-1]
        
        # Save final model with MLflow
        final_metrics = {
            'final_val_accuracy': float(best_val_accuracy),
            'final_train_loss': float(history['train_loss'][-1]),
            'best_val_accuracy': float(best_val_accuracy)
        }
        mlflow.log_metrics(final_metrics)
        
        # Log of the model metadata
        metadata = {
            'training_date': datetime.now().strftime("%Y-%m-%d"),
            'model_version': '1.0',
            'framework_versions': {
                'pytorch': str(torch.__version__),
                'numpy': str(np.__version__)
            }
        }

        for key, value in metadata.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_param(f"{key}.{sub_key}", str(sub_value))
            else:
                mlflow.log_param(key, value)
        writer.close()

        if best_model_state is not None:
            modelRec.load_state_dict(best_model_state)
        return modelRec
    
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Training epoch"""
    model.train()
    total_loss = 0
    for x1, x2, y, _ in tqdm(train_loader, desc="Training"):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            logits = model(x1, x2)
            loss = criterion(logits, y.squeeze())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Validation epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x1, x2, y, _ in val_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            with torch.autocast(device_type='cuda'):
                logits = model(x1, x2)
                loss = criterion(logits, y.squeeze())
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.squeeze().cpu().numpy())
    
    val_loss = total_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    
    return val_loss, val_accuracy


def setup_mlflow():
    """Configure MLflow tracking"""
    from init_mlflow import init_mlflow
    experiment_id = init_mlflow()
    return "recommendation_experiment"

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Configuration
    config = {
        "epochs": 2,
        "batch_size": 512,
        "learning_rate": 1e-4,
        "user_embSize": 32,
        "item_embSize": 32,
    }
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Download dataset if not exists
    print("Checking and downloading dataset if needed...")
    download_and_extract_dataset()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    
    # Load and preprocess data
    df_movies, df_ratings, df_users = read_dataset()
    df_combined = preprocess_dataset(df_movies, df_ratings, df_users)
    
 
    # Determine userCount and itemCount
    max_user_id = df_combined["user_id"].max()
    max_item_id = df_combined["item_id"].max()
    userCount = max_user_id + 1
    itemCount = max_item_id + 1

    NEGATIVE_SAMPLES_PER_USER = 80
    # Prepare train and test datasets
    df_train, df_test = prepare_data(df_combined, df_movies, df_users)

    # Train model
    model = train_model(df_train, df_test, userCount, itemCount, device, config)

    # Save model
    dict_titlEmb = encode_title(df_train,df_test)
    all_unique_titles = pd.concat([df_train, df_test])["title"].unique()
    title_to_idx = {title: idx for idx, title in enumerate(all_unique_titles)}
    list_titlEmb = [dict_titlEmb[title] for title in title_to_idx]
    title_emb_tensor = torch.stack(list_titlEmb)
    # Evaluation of the performances
    user_stats = analyze_multiple_users(100, 42, df_test, model, device, df_combined, title_to_idx,title_emb_tensor)

    mean_accuracy = np.mean([stat['overall_accuracy'] for stat in user_stats])
    if mean_accuracy > 85:
        metadata = {
            'mean_accuracy': str(mean_accuracy),
            'training_date': datetime.now().strftime("%Y-%m-%d"),
            'model_version': '1.0'
        }
        
        # Sauvegarder et récupérer l'experiment_id
        experiment_id = save_production_model(model, "modelRec", metadata)
        
        if experiment_id:
            # Utiliser l'experiment_id pour charger le modèle
            prod_model = load_production_model("modelRec", userCount, itemCount, experiment_id)
            
            if prod_model is None:
                print("Using current model as production model")
                prod_model = model
            else:
                print("Production model loaded successfully")
                
            prod_model = prod_model.to(device)
            
            detailed_user_recommendations(
                user_id=10,
                modelRec=prod_model,
                df_combined=df_combined,
                device=device,
                df_train=df_train,
                df_test=df_test,
                title_to_idx=title_to_idx,
                title_emb_tensor=title_emb_tensor
            )

