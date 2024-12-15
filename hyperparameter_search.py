import mlflow
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from train import cls_model, train_epoch, validate_epoch, FocalLoss
from utils import prepare_data, create_datasets, read_dataset, preprocess_dataset, create_and_log_plots, create_comparison_visualizations

def train_with_params(df_train, df_test, params, device, title_to_idx, title_emb_tensor):
    """Train with parameters"""
    with mlflow.start_run(run_name=f"hp_search_lr{params['learning_rate']}_bs{params['batch_size']}"):
        mlflow.log_params(params)
        scaler = torch.GradScaler()
        
        modelRec = cls_model(
            userCount=params['userCount'], 
            itemCount=params['itemCount'],
            user_embSize=32,
            item_embSize=32
        ).to(device)
        
        optimizer = torch.optim.Adam(modelRec.parameters(), lr=params['learning_rate'])
        criterion = FocalLoss(alpha=1, gamma=2).to(device)
        
        ds_train, ds_testLoader, _, _ = create_datasets(
            df_train, df_test, 
            params['userCount'], 
            params['itemCount']
        )
        
        train_loader = DataLoader(
            ds_train, 
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_accuracy = 0
        
        for epoch in range(params['epochs']):
            train_loss = train_epoch(modelRec, train_loader, criterion, optimizer, scaler, device)
            val_loss, val_accuracy = validate_epoch(modelRec, ds_testLoader, criterion, device)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # Sauvegarder le meilleur modèle
                model_path = f"model_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': modelRec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                }, model_path)
                mlflow.log_artifact(model_path)
                os.remove(model_path)
            
            create_and_log_plots(history, epoch)
            
        # Log the final model with simple metadata
        metadata = {
            'final_val_accuracy': float(history['val_accuracy'][-1]),
            'final_train_loss': float(history['train_loss'][-1]),
            'framework_versions': {
                'pytorch': f"{torch.__version__}",  # Conversion explicite en string
                'numpy': f"{np.__version__}"        # Conversion explicite en string
            }
        }
        
        # Save the model with MLflow
        try:
            # Save the model locally first
            model_path = f"model_lr{params['learning_rate']}_bs{params['batch_size']}.pth"
            torch.save(modelRec.state_dict(), model_path)
            
            # Log the model as an artifact
            mlflow.log_artifact(model_path)
            os.remove(model_path)
            
            # Log the metadata as parameters
            for key, value in metadata.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        mlflow.log_param(f"{key}.{sub_key}", str(sub_value))
                else:
                    mlflow.log_param(key, value)
                    
        except Exception as e:
            print(f"Warning: Error while saving model: {e}")
            
        return history

def hyperparameter_search(df_train, df_test, userCount, itemCount, device):
    # Define the hyperparameters to test
    param_grid = {
        'learning_rate': [0.0001, 0.001],
        'batch_size': [256, 512],
        'epochs': [2],
        'userCount': [userCount],
        'itemCount': [itemCount]
    }
    
    combinations = [dict(zip(param_grid.keys(), v)) 
                   for v in itertools.product(*param_grid.values())]
    
    # Create the datasets once
    _, _, title_to_idx, title_emb_tensor = create_datasets(
        df_train, df_test, userCount, itemCount
    )
    
    results = []
    for params in combinations:
        history = train_with_params(
            df_train, df_test, params, device, 
            title_to_idx, title_emb_tensor
        )
        results.append({
            **params,
            'final_val_accuracy': history['val_accuracy'][-1],
            'final_train_loss': history['train_loss'][-1]
        })
    
    create_comparison_visualizations(results)
    return results




if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # MLflow configuration
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("hyperparameter_search")
    
    # Configuration device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the data
    df_movies, df_ratings, df_users = read_dataset()
    df_combined = preprocess_dataset(df_movies, df_ratings, df_users)
    
 
    # Determine userCount and itemCount
    max_user_id = df_combined["user_id"].max()
    max_item_id = df_combined["item_id"].max()
    userCount = max_user_id + 1
    itemCount = max_item_id + 1
    df_train, df_test = prepare_data(df_combined, df_movies, df_users)
    
    # Launch the hyperparameter search
    results = hyperparameter_search(df_train, df_test, userCount, itemCount, device)
    
    # Save the results
    results_df = pd.DataFrame(results)
    results_df.to_csv("hyperparameter_search_results.csv", index=False)
    print("\nBest hyperparameters:")
    best_idx = results_df['final_val_accuracy'].idxmax()
    print(results_df.iloc[best_idx])